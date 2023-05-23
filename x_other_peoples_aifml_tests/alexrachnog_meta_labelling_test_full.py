import sys
import os

sys.path.append("..")
cwd = os.getcwd()

import matplotlib
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import time
import io
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import LabelBinarizer
import pprint
from scipy.stats import entropy
from scipy import stats

from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GaussianNoise,
    Conv1D,
    MaxPool1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.layers import (
    Input,
    concatenate,
    SpatialDropout1D,
    Flatten,
    AvgPool1D,
    LSTM,
    Lambda,
    Reshape,
    GaussianDropout,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2, l1_l2

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

# import library.spartan_stats
# from spartan_stats import check_different_frac_diffs

from support_files.locations import DB_USER, DB_HOST, DB_PASSWORD, DB_NAME

# import frac_diff
# from frac_diff import frac_diff_ffd_panda
import library.features
from library.features import make_features_from_window, get_class_weights
import library.labelling
from library.labelling import (
    fixed_horizon_labelling,
    volatility_horizon_labelling,
    get_meta_barrier,
)
from library.splitting import horizon_splitting
import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
from collections import Counter


def make_features_from_window(X_train_b, X_val, X_test, features, WINDOW_LONG):

    X_train_normts = np.array(
        [(x.close - x.close.iloc[0]) / np.std(x.close) for x in X_train_b]
    ).reshape((len(X_train_b), WINDOW_LONG, 1))
    X_val_normts = np.array(
        [(x.close - x.close.iloc[0]) / np.std(x.close) for x in X_val]
    ).reshape((len(X_val), WINDOW_LONG, 1))
    X_test_normts = np.array(
        [(x.close - x.close.iloc[0]) / np.std(x.close) for x in X_test]
    ).reshape((len(X_test), WINDOW_LONG, 1))

    X_train_normv = np.array(
        [(x.volume - x.volume.iloc[0]) / np.std(x.volume) for x in X_train_b]
    ).reshape((len(X_train_b), WINDOW_LONG, 1))
    X_val_normvv = np.array(
        [(x.volume - x.volume.iloc[0]) / np.std(x.volume) for x in X_val]
    ).reshape((len(X_val), WINDOW_LONG, 1))
    X_test_normv = np.array(
        [(x.volume - x.volume.iloc[0]) / np.std(x.volume) for x in X_test]
    ).reshape((len(X_test), WINDOW_LONG, 1))

    X_train = np.array([x[features].fillna(0.0).values.tolist() for x in X_train_b])
    X_val = np.array([x[features].fillna(0.0).values.tolist() for x in X_val])
    X_test = np.array([x[features].fillna(0.0).values.tolist() for x in X_test])

    X_train = np.concatenate((X_train, X_train_normts, X_train_normv), axis=-1)
    X_val = np.concatenate((X_val, X_val_normts, X_val_normvv), axis=-1)
    X_test = np.concatenate((X_test, X_test_normts, X_test_normv), axis=-1)

    return X_train, X_val, X_test


def get_class_weights(y):
    y = [np.argmax(x) for x in y]
    counter = Counter(y)
    majority = max(counter.values())
    return {
        cls: round(float(majority) / float(count), 2) for cls, count in counter.items()
    }


import numpy as np
import pandas as pd

from mlfinlab.util.multiprocess import mp_pandas_obj


# Fixed Horizon
def fixed_horizon_labelling(tick_bars, WINDOW_LONG, N_BARS, HORIZON, T):
    X, labels = [], []
    for i in range(WINDOW_LONG, N_BARS, 1):

        window = tick_bars.iloc[i - WINDOW_LONG : i]
        #     window = tick_bars.iloc[i]
        now = tick_bars.close[i]
        future = tick_bars.close[i + HORIZON]
        ret = (future - now) / now

        X.append(window)
        if ret > T:
            labels.append(1)
        elif ret < -T:
            labels.append(-1)
        else:
            labels.append(0)
    return X, labels


def volatility_horizon_labelling(tick_bars, WINDOW_LONG, N_BARS, HORIZON):
    X, labels = [], []
    for i in range(WINDOW_LONG, N_BARS, 1):
        window = tick_bars.iloc[i - WINDOW_LONG : i]
        now = tick_bars.close[i]
        future = tick_bars.close[i + HORIZON]
        ret = (future - now) / now

        window_abs_returns = np.abs(window.close.pct_change())
        Ti = np.std(window_abs_returns) + np.mean(window_abs_returns)

        X.append(window)
        if ret > Ti:
            labels.append(1)
        elif ret < -Ti:
            labels.append(-1)
        else:
            labels.append(0)
    return X, labels


def get_meta_barrier(future_window, last_close, min_ret, tp, sl, vertical_zero=False):
    """
        XXX
    """
    if vertical_zero:
        min_ret_situation = [0, 0, 0]
    else:
        min_ret_situation = [0, 0]

    differences = np.array([(fc - last_close) / last_close for fc in future_window])

    # Are there gonna be fluctuations within min_ret???
    min_ret_ups = np.where((differences >= min_ret) == True)[0]
    min_ret_downs = np.where((differences < -min_ret) == True)[0]

    if (len(min_ret_ups) == 0) and (len(min_ret_downs) == 0):
        if vertical_zero:
            min_ret_situation[2] = 1
        else:
            if differences[-1] > 0:
                min_ret_situation[0] = 1
            else:
                min_ret_situation[1] = 1
    else:
        if len(min_ret_ups) == 0:
            min_ret_ups = [np.inf]
        if len(min_ret_downs) == 0:
            min_ret_downs = [np.inf]
        if min_ret_ups[0] > min_ret_downs[0]:
            min_ret_situation[0] = 1
        else:
            min_ret_situation[1] = 1

    #  Take profit and stop losses indices
    take_profit = np.where((differences >= tp) == True)[0]
    stop_loss = np.where((differences < sl) == True)[0]

    # Fluctuation directions coincide with take profit / stop loss actions?
    if min_ret_situation[0] == 1 and len(take_profit) != 0:
        take_action = 1
    elif min_ret_situation[1] == 1 and len(stop_loss) != 0:
        take_action = 1
    else:
        take_action = 0

    return min_ret_situation, take_action


data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()
head = 100000
if head > 0:
    data = data.head(head)

data = data.set_index("date_time")
data.index = pd.to_datetime(data.index)


plt.figure()
plt.plot(data.close[: int(len(data) * 0.5)])
plt.plot(data.close[int(len(data) * 0.5) : int(len(data) * 0.7)])
plt.show()

# fracs = frac_diff_ffd_panda(close, d=0.175, thres=1e-5)

# plt.figure()
# plt.plot(fracs['close'][:int(len(fracs) * 0.5)])
# plt.plot(fracs['close'][int(len(fracs) * 0.5):int(len(fracs) * 0.7)])
# plt.show()

# d = 0.175
# ts1 = np.log(close)
# ts2 = frac_diff_ffd_panda(ts1, d)
# ts2 = ts2.set_index('timestamp')  #  corr = 0.8994240996886402
# ts2 = ts2.rank(pct=True)  #  corr = 0.8689590225352614

# corr = np.corrcoef(ts1.loc[ts2.index, 'close'], ts2['close'])[0, 1]
# adf = adfuller(ts2.close)[0]

# # The Some stats
# H = 10
# print(pd.Series.autocorr(np.clip(ts2.close.pct_change(), -H, H).dropna()))
# print(stats.jarque_bera(ts2.close.values))
# print(stats.shapiro(ts2.close.values))

# # Checking different fractional differentiation levels
# # effect on correlation and stationarity
# stats, plot = check_different_frac_diffs(0.0, 1.5, 0.5, ts1)
# plt.show()

# d = 0.175
# frac_diff_bars = frac_diff_ffd_panda(np.log(close), d).set_index("timestamp")

# dollar_bars["pct_rank_frac"] = frac_diff_bars

# dollar_bars.reset_index(inplace=True)

FEATURES = ["close"]
WINDOW_LONG = 100
HORIZON = 25
N_BARS = len(data) - HORIZON
T = 0.01

# META LABELLING
TP = T
SL = -T

import importlib

importlib.reload(library.labelling)

X, Y, Y2, Y_min_ret_numbers = [], [], [], []
for i in range(WINDOW_LONG, N_BARS, 1):
    window = data.iloc[i - WINDOW_LONG : i]
    now = data.close[i]
    future_window = data.close[i : i + HORIZON]

    # how has the price been deviating over the window t-100:t
    # use this as measure of volatility which you can compare to the future window
    window_abs_returns = np.abs(window.close.pct_change())
    Ti = np.std(window_abs_returns) + np.mean(window_abs_returns)

    min_ret_situation, take_action = get_meta_barrier(
        future_window, now, Ti, TP, SL, True
    )
    X.append(window)
    if min_ret_situation[0] == 1:
        min_ret_outcome = 0
    elif min_ret_situation[1] == 1:
        min_ret_outcome = 1
    elif min_ret_situation[2] == 1:
        min_ret_outcome = 2
    Y_min_ret_numbers.append(min_ret_outcome)
    Y.append(
        min_ret_situation
    )  # is the price changing more than the required volatility measure
    Y2.append(take_action)  # would it of made profit ?

unique, counts = np.unique(Y_min_ret_numbers, return_counts=True)
unique2, counts2 = np.unique(Y2, return_counts=True)


# plt.figure()
# plt.hist([np.argmax(x) for x in Y], alpha=0.5, bins=5)
# plt.show()

# plt.figure()
# plt.hist([np.argmax(x) for x in Y2], alpha=0.5, bins=5)
# plt.show()


X_train, X_val, X_test = (
    X[: int(len(X) * 0.5)],
    X[int(len(X) * 0.6) : int(len(X) * 0.7)],
    X[int(len(X) * 0.8) :],
)
Y_train, Y_val, Y_test = (
    Y[: int(len(X) * 0.5)],
    Y[int(len(X) * 0.6) : int(len(X) * 0.7)],
    Y[int(len(X) * 0.8) :],
)

X_train, X_val, X_test = make_features_from_window(
    X_train, X_val, X_test, FEATURES, WINDOW_LONG
)

Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)


def cnn_one(shape):
    main_input = Input(shape=shape, name="main_input")
    x = Flatten()(main_input)
    x = Dropout(0.25)(x)
    output = Dense(3, activation="softmax")(x)

    final_model = Model(inputs=[main_input], outputs=[output])
    return final_model


model = cnn_one((WINDOW_LONG, len(X_train[0][0])))
model.summary()

model.compile(
    optimizer=Adam(lr=0.01), loss=["categorical_crossentropy"], metrics=["accuracy"]
)

checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(monitor="val_loss", patience=5)

history = model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=16,
    verbose=True,
    validation_data=(X_val, Y_val),
    callbacks=[checkpointer, es],
    shuffle=True,
    class_weight=get_class_weights(np.concatenate((Y_train, Y_val))),
)

model.load_weights("test.hdf5")

pred = model.predict(X_val)

print(
    classification_report([np.argmax(y) for y in Y_val], [np.argmax(y) for y in pred])
)

print(confusion_matrix([np.argmax(y) for y in Y_val], [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_val], bins=5, alpha=0.5, label="Test data")
plt.hist([np.argmax(y) for y in pred], bins=5, alpha=0.5, label="Predicted data")
plt.legend()
plt.show()

pred = model.predict(X_test)

print(
    classification_report([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
)

print(confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_test], bins=5, alpha=0.5, label="Test data")
plt.hist([np.argmax(y) for y in pred], bins=5, alpha=0.5, label="Predicted data")
plt.legend()
plt.show()

X_train, X_val, X_test = (
    X[: int(len(X) * 0.5)],
    X[int(len(X) * 0.6) : int(len(X) * 0.7)],
    X[int(len(X) * 0.8) :],
)
Y_train, Y_val, Y_test = (
    Y2[: int(len(X) * 0.5)],
    Y2[int(len(X) * 0.6) : int(len(X) * 0.7)],
    Y2[int(len(X) * 0.8) :],
)

X_train, X_val, X_test = make_features_from_window(
    X_train, X_val, X_test, FEATURES, WINDOW_LONG
)
P_train, P_val, P_test = (
    model.predict(X_train),
    model.predict(X_val),
    model.predict(X_test),
)

Y_train = np.array([[1, 0] if x == 1 else [0, 1] for x in Y_train])
Y_val = np.array([[1, 0] if x == 1 else [0, 1] for x in Y_val])
Y_test = np.array([[1, 0] if x == 1 else [0, 1] for x in Y_test])


def cnn_two(shape):
    main_input = Input(shape=shape, name="main_input")
    aux_input = Input((3,), name="meta")
    x = Flatten()(main_input)
    x = Dropout(0.25)(x)
    x = concatenate([x, aux_input])
    output = Dense(2, activation="softmax")(x)
    final_model = Model(inputs=[main_input, aux_input], outputs=[output])
    return final_model


model = cnn_two((WINDOW_LONG, len(X_train[0][0])))
model.summary()

model.compile(
    optimizer=Adam(lr=0.01), loss=["categorical_crossentropy"], metrics=["accuracy"]
)

checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(monitor="val_loss", patience=5)

history = model.fit(
    [X_train, P_train],
    Y_train,
    epochs=100,
    batch_size=16,
    verbose=True,
    validation_data=([X_test, P_test], Y_test),
    callbacks=[checkpointer, es],
    shuffle=True,
    class_weight=get_class_weights(np.concatenate((Y_train, Y_val))),
)

model.load_weights("test.hdf5")

pred = model.predict([X_val, P_val])

print(
    classification_report([np.argmax(y) for y in Y_val], [np.argmax(y) for y in pred])
)

print(confusion_matrix([np.argmax(y) for y in Y_val], [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_val], bins=5, alpha=0.5, label="Test data")
plt.hist([np.argmax(y) for y in pred], bins=5, alpha=0.5, label="Predicted data")
plt.legend()
plt.show()

pred = model.predict([X_test, P_test])

print(
    classification_report([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])
)

print(confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_test], bins=5, alpha=0.5, label="Test data")
plt.hist([np.argmax(y) for y in pred], bins=5, alpha=0.5, label="Predicted data")
plt.legend()
plt.show()

