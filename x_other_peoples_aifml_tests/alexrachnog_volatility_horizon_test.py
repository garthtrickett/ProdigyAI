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

import spartan_stats
from spartan_stats import check_different_frac_diffs
import database
from database import connect_to_database, read_sql_tmpfile
from bars import (
    BarSeries,
    TickBarSeries,
    VolumeBarSeries,
    DollarBarSeries,
    ImbalanceTickBarSeries,
)
from support_files.locations import DB_USER, DB_HOST, DB_PASSWORD, DB_NAME
import frac_diff
from frac_diff import frac_diff_ffd_panda
import features
from features import make_features_from_window, get_class_weights
import labelling
from labelling import (
    fixed_horizon_labelling,
    volatility_horizon_labelling,
    get_meta_barrier,
)
from splitting import horizon_splitting

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


# Start Script
database_details = [DB_HOST, DB_PASSWORD, DB_USER, DB_NAME]

cur, engine, conn = connect_to_database(
    database_details[0], database_details[1], database_details[2], database_details[3]
)
table_names = engine.table_names()
table_name = table_names[3]

# Alternate way of reading the database
#  frames = pd.read_sql_table(table_name, engine, chunksize=1000)
query = "SELECT aggregate_trade_id, price, quantity, timestamp FROM {0} ORDER BY aggregate_trade_id asc".format(
    table_name
)

# Read from sql to panda with tempfile
df = read_sql_tmpfile(query, engine)
df = df.apply(pd.to_numeric)

# # TICK BARS
bars = TickBarSeries(df)
tick_bars = bars.process_ticks(frequency=1000)

# # # #  VOLUME BARS
# # bars = VolumeBarSeries(df)
# # volume_bars = bars.process_ticks(frequency=100)
# # # DOLLAR BARS
# # bars = DollarBarSeries(df)
# # dollar_bars = bars.process_ticks(frequency=100)
# # # IMBALANCED TICK BARS
# # bars = ImbalanceTickBarSeries(df)
# # imbtick_bars = bars.process_ticks(init=100, min_bar=10, max_bar=1000)
# # # # TIME BARS
# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', origin='unix')
# df = df.set_index('timestamp')
# bars = BarSeries(df)
# time_bars = bars.process_ticks(frequency='1Min')
# time_bars = time_bars.dropna(subset=['close'])

plt.figure()
plt.plot(tick_bars.close[: int(len(tick_bars) * 0.5)])
plt.plot(tick_bars.close[int(len(tick_bars) * 0.5) : int(len(tick_bars) * 0.7)])
plt.show()

# tickbar_no_index = tick_bars.reset_index()
close = tick_bars[["close"]]

fracs = frac_diff_ffd_panda(close, d=0.175, thres=1e-5)

plt.figure()
plt.plot(fracs["close"][: int(len(fracs) * 0.5)])
plt.plot(fracs["close"][int(len(fracs) * 0.5) : int(len(fracs) * 0.7)])
plt.show()

d = 0.175
ts1 = np.log(close)
ts2 = frac_diff_ffd_panda(ts1, d)
ts2 = ts2.set_index("timestamp")  #  corr = 0.8994240996886402
ts2 = ts2.rank(pct=True)  #  corr = 0.8689590225352614

corr = np.corrcoef(ts1.loc[ts2.index, "close"], ts2["close"])[0, 1]
adf = adfuller(ts2.close)[0]

# The Some stats
H = 10
print(pd.Series.autocorr(np.clip(ts2.close.pct_change(), -H, H).dropna()))
print(stats.jarque_bera(ts2.close.values))
print(stats.shapiro(ts2.close.values))

# Checking different fractional differentiation levels
# effect on correlation and stationarity
stats, plot = check_different_frac_diffs(0.0, 1.5, 0.5, ts1)
plt.show()

d = 0.175
frac_diff_bars = (
    frac_diff_ffd_panda(np.log(close), d).set_index("timestamp").rank(pct=True)
)

tick_bars["pct_rank_frac"] = frac_diff_bars

tick_bars.reset_index(inplace=True)

FEATURES = ["pct_rank_frac"]
WINDOW_LONG = 100
WINDOW_SHORT = 50
HORIZON = 25
N_BARS = len(tick_bars) - HORIZON
T = 0.01
H = 0.05

# Volitility Horizon Labelling
X, labels = volatility_horizon_labelling(tick_bars, WINDOW_LONG, N_BARS, HORIZON)

plt.figure()
plt.hist(labels, bins=5, alpha=0.7)
plt.show()

X_train, X_val, X_test, Y_train, Y_val, Y_test = horizon_splitting(X, labels)
X_train, X_val, X_test = make_features_from_window(
    X_train, X_val, X_test, FEATURES, WINDOW_LONG
)


def cnn(shape):
    main_input = Input(shape=shape, name="main_input")
    x = Flatten()(main_input)
    x = Dropout(0.25)(x)
    output = Dense(3, activation="softmax")(x)

    final_model = Model(inputs=[main_input], outputs=[output])
    return final_model


model = cnn((WINDOW_LONG, len(X_train[0][0])))
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

