import sys
import os

sys.path.append("..")
cwd = os.getcwd()
import importlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Nadam
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
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
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GaussianNoise,
    Conv1D,
    MaxPool1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.preprocessing import LabelBinarizer
import pyarrow as pa
import pyarrow.parquet as pq

import time
import pyfolio as pf
import pandas as pd
import numpy as np
from numpy import array
import arviz
import pymc3 as pm
import h5py


import library.blackarbsceo_bars as brs
import library.plotting_and_statistics
from library.plotting_and_statistics import plot_learning_curves
import third_party_libraries.mlfinlab as ml
from third_party_libraries.mlfinlab.filters import filters
from third_party_libraries.mlfinlab.labeling import labeling
from third_party_libraries.mlfinlab.util import utils
from third_party_libraries.mlfinlab.features import fracdiff
import library.snippets as snp
import library.labelling
from library.labelling import (
    fixed_horizon_labelling,
    volatility_horizon_labelling,
    get_meta_barrier,
)
import library.features
from library.features import make_features_from_window, get_class_weights
import third_party_libraries.finance_ml

from third_party_libraries.mlfinlab.util.multiprocess import mp_pandas_obj

from third_party_libraries.finance_ml.stats.vol import *

import multiprocessing as mp
from multiprocessing import cpu_count

cpus = cpu_count()

import sklearn

assert sklearn.__version__ >= "0.20"


from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"

# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

from third_party_libraries.mlfinlab.filters.filters import cusum_filter

#
# FROM HERE: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb

# Check for duplicates
# duplicates_index_ = data.index.duplicated(keep='first')
# unique, counts = np.unique(duplicates_index_, return_counts=True)

# thoughts ... filtering events with cusum filter vses giving rows where its activated a 1,-1 and others a 0 and keeping all rows


# spartan labelling
# Write the code similar to honchar that gets side and size
# if the previous price is x% higher/lower than the price before it also multiply this by the time between the timestamps by some thresholdactivate the cusum filter
# add in some way where when the volatility of the window is higher the percentage change needs to be higher
# add these events into a list
# check for these rows which part of triple barrier is hit first and label that the side 0,0,0 (lower, end, higher)
# check these points to see whether there was actually enough up or down to gain profit. (feed side prediction as input into size)


# META LABELLING
print("starting data load")
# read parquet file of dollar bars
data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()
print("data load finished")
head = 8000
if head > 0:
    data = data.head(head)

data = data.set_index("date_time")
data.index = pd.to_datetime(data.index)
data.reset_index(inplace=True)


def total_seconds_wrapper(window):
    return window.total_seconds()


import importlib


def triple_barrier(future_window, last_close, tp, sl, vertical_zero=False):
    count = 0

    differences = np.array(
        [(fc - last_close) / last_close for fc in future_window.close]
    )
    if max(differences) > tp:
        outcome = 1
    elif min(differences) < sl:
        outcome = -1
    else:
        outcome = 0

    return outcome


TIME_MODIFIER = 1
WINDOW_LONG = 100
HORIZON = 25
PROFIT_THRESHOLD = 0.0012


def labelling(WINDOW_LONG, data, HORIZON, TIME_MODIFIER, PROFIT_THRESHOLD, molecule):
    print(str(molecule[0]) + "___" + str(molecule[-1]))
    if molecule[0] == 0:
        x = 2
        start = WINDOW_LONG
    elif molecule[0] > 0 and molecule[-1] != data.index[-1]:
        start = molecule[0]
        split_molecule = pd.RangeIndex(
            start=molecule[0] - WINDOW_LONG, stop=molecule[-1] + HORIZON, step=1
        )
        molecule = split_molecule
    elif molecule[-1] == data.index[-1] and molecule[0] != data.index[0]:
        x = 1
        start = molecule[0]
        split_molecule = pd.RangeIndex(
            start=molecule[0] - WINDOW_LONG, stop=molecule[-1], step=1
        )
        molecule = split_molecule

    data = data.loc[molecule]
    data = data.to_frame()

    last_bar = data.index[-1] - HORIZON
    TP = PROFIT_THRESHOLD
    SL = -PROFIT_THRESHOLD
    X, Y = [], []

    for i in range((start), (last_bar), 1):
        future_window = data.loc[i : i + HORIZON]
        now = data.close[i]

        outcome = triple_barrier(future_window, now, TP, SL, vertical_zero=True)
        Y.append(outcome)

    scaler = MinMaxScaler()  # normalization
    scaler = StandardScaler()  # standardization
    data[["close"]] = scaler.fit_transform(data[["close"]])

    for i in range((start), (last_bar), 1):
        window = data.loc[i - WINDOW_LONG : i]

        # print(len(window))
        # window = window.pct_change()
        # window = window.replace([np.inf, -np.inf], np.nan)
        # window.dropna(inplace=True)  # remove the nas created by norming
        # print(len(window))

        # window_abs_returns = np.abs(window.close.pct_change())
        # # Volatility measure
        # window_volatility = np.std(window_abs_returns)
        # window_time_differences = (
        #     window["date_time"] - window["date_time"].shift()
        # ).fillna(0)
        # window_time_differences_in_seconds = window_time_differences.apply(
        #     total_seconds_wrapper
        # )
        # window_time_adjusted_volatility = window_volatility * (
        #     TIME_MODIFIER / window_time_differences_in_seconds
        # )
        X.append(window)

    return X, Y


# data[col] = data[col].pct_change()
# data = data.replace([np.inf, -np.inf], np.nan)
# data.dropna(inplace=True)  # remove the nas created by norming
# data.close = preprocessing.scale(data.close.values)
# data.dropna(inplace=True)

# X_and_y.reset_index(inplace=True)

# Make the number of threads a factor of the total number of rows ie 4 and 1600
num_threads = (cpus * 2) - 1
df1 = mp_pandas_obj(
    func=labelling,
    pd_obj=("molecule", data["close"].index),
    num_threads=num_threads,
    data=data["close"],
    WINDOW_LONG=WINDOW_LONG,
    HORIZON=HORIZON,
    TIME_MODIFIER=TIME_MODIFIER,
    PROFIT_THRESHOLD=PROFIT_THRESHOLD,
)

import itertools

X_list = [x for x, y in df1]
X = list(itertools.chain.from_iterable(X_list))
Y_list = [y for x, y in df1]
Y = list(itertools.chain.from_iterable(Y_list))


X_train, X_val, X_test = (
    X[: int(len(X) * 0.5)],
    X[int(len(X) * 0.6) : int(len(X) * 0.7)],
    X[int(len(X) * 0.8) :],
)

WINDOW_SHAPE = WINDOW_LONG + 1

X_train = np.array([np.array(x[:]) for x in X_train]).reshape(
    (len(X_train), WINDOW_SHAPE, 1)
)
X_val = np.array([np.array(x[:]) for x in X_val]).reshape((len(X_val), WINDOW_SHAPE, 1))
X_test = np.array([np.array(x[:]) for x in X_test]).reshape(
    (len(X_test), WINDOW_SHAPE, 1)
)

unique, counts = np.unique(Y, return_counts=True)

Y_train, Y_val, Y_test = (
    Y[: int(len(X) * 0.5)],
    Y[int(len(X) * 0.6) : int(len(X) * 0.7)],
    Y[int(len(X) * 0.8) :],
)


# sm = SMOTE()
sm = ADASYN()  # adds in randomness on top of smote
X_train = X_train.reshape(len(X_train), len(X_train[0]))
Y_train = np.asarray(Y_train)
X_train, Y_train = sm.fit_sample(X_train, Y_train)
unique, counts = np.unique(Y_val, return_counts=True)

lbr = LabelBinarizer()
Y_train = lbr.fit_transform((Y_train))
X_train = np.array(X_train).reshape((len(X_train), WINDOW_SHAPE, 1))

Y_val = np.asarray(Y_val)
Y_val = lbr.fit_transform((Y_val))
Y_test = np.asarray(Y_test)
Y_test = lbr.fit_transform((Y_test))


# # # Fractional differentiation
# # data_series = data['close'].to_frame()
# # # # Log the prices
# # log_prices = np.log(data_series)
# # d = 0.4
# # fd_series = fracdiff.frac_diff_ffd(log_prices, diff_amt=d, thresh=1e-5)
# # # fd_series.head(1480)
# # fd_series.dropna()

# bar_name = "btcusdt_agg_trades_10_tick_bars"

# # Triple Barrier Parameters
# vol_span = 100
# vol_seconds = 3600
# vol_modifier = 0.2
# # lower vol_modifier = more cusum events
# # cant see any pattern for how changing vol_modifier affects percentages of -1,0,1 bins
# vertical_num_days = 1
# pt_sl = [1, 1]
# min_ret = 0.0011

# triple_barrier_file_name = (
#     "bar_name="
#     + bar_name
#     + "head="
#     + str(head)
#     + "_vol_span="
#     + str(vol_span)
#     + "_vol_seconds="
#     + str(vol_seconds)
#     + "_vol_modifier="
#     + str(vol_modifier)
#     + "_vertical_num_days="
#     + str(vertical_num_days)
#     + "_pt="
#     + str(pt_sl[0])
#     + "_pt="
#     + str(pt_sl[1])
#     + "_pt="
#     + str(min_ret)
# )

# # Get volatility

# # vol = get_vol(data["close"], span=vol_span, seconds=vol_seconds)

# # alternative measure of volatility
# # vol = data['close'].std()

# # third volatility measure
# # Compute daily volatility
# vol = ml.util.get_daily_vol(close=data["close"], lookback=vol_span)

# # # modulate the profit taking margin #
# vol *= vol_modifier


# # Generate cusum event filter # takes a long tie but less than triple barrier
# start = time.time()
# num_threads = 6
# df1 = mp_pandas_obj(
#     func=cusum_filter,
#     pd_obj=("molecule", data["close"].index),
#     num_threads=num_threads,
#     raw_time_series=data["close"],
#     threshold=vol.mean(),
# )
# count = 0
# for dt_index in df1:
#     if count != 0:
#         last_index = last_index.append(dt_index)
#     else:
#         last_index = dt_index
#     count = count + 1

# cusum_events = last_index
# end = time.time()
# print(end - start)

# # Compute vertical barrier

# start = time.time()
# vertical_barriers = ml.labeling.add_vertical_barrier(
#     t_events=cusum_events, close=data["close"], num_days=vertical_num_days
# )
# end = time.time()
# print(end - start)

# # Triple barrier ## takes ages (can up the threads)
# start = time.time()
# triple_barrier_events = ml.labeling.get_events(
#     close=data["close"],
#     t_events=cusum_events,
#     pt_sl=pt_sl,
#     target=vol,
#     min_ret=min_ret,
#     num_threads=6,
#     vertical_barrier_times=vertical_barriers,
# )

# end = time.time()
# print(end - start)


# # # Saving some data so we don't have to run the triple barrier labelling every time
# # table = pa.Table.from_pandas(triple_barrier_events)
# # pq.write_table(
# #     table,
# #     "data/barrier_events/" + triple_barrier_file_name + ".parquet",
# #     use_dictionary=True,
# #     compression="snappy",
# # )

# # # Reading the triple barrier events
# # table = pq.read_table("data/barrier_events/" + triple_barrier_file_name + ".parquet")
# # triple_barrier_events = table.to_pandas()

# # create labels
# labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
# clean_labels = ml.labeling.drop_labels(labels)
# clean_labels.bin.value_counts()


# start = time.time()
# X_and_y = clean_labels.join(data)

# # X_and_y['bin'].shift(1) # shifting the label by one

# X_and_y.drop(["open", "high", "low", "ret", "trgt"], axis=1, inplace=True)

# for col in X_and_y.columns:  # go through all of the columns
#     if col == "close":  #
#         X_and_y[col] = X_and_y[col].pct_change()
#         X_and_y = X_and_y.replace([np.inf, -np.inf], np.nan)
#         X_and_y.dropna(inplace=True)  # remove the nas created by norming

#         X_and_y[col] = preprocessing.scale(X_and_y[col].values)
#         X_and_y.dropna(inplace=True)

# X_and_y.reset_index(inplace=True)


# WINDOW_LONG = 100
# X, Y, timestamp_index = [], [], []
# for i, row in X_and_y.iterrows():
#     one_hot_list = []
#     if i > WINDOW_LONG:
#         window = X_and_y.iloc[i - WINDOW_LONG : i]
#         X.append(window)
#         if row["bin"] == -1:
#             one_hot_list = [1, 0, 0]
#         elif row["bin"] == 0:
#             one_hot_list = [0, 1, 0]
#         elif row["bin"] == 1:
#             one_hot_list = [0, 0, 1]
#         Y.append(one_hot_list)


# X_train, X_val, X_test = (
#     X[: int(len(X) * 0.5)],
#     X[int(len(X) * 0.6) : int(len(X) * 0.7)],
#     X[int(len(X) * 0.8) :],
# )

# X_train = np.array([x.close for x in X_train]).reshape((len(X_train), WINDOW_LONG, 1))

# X_val = np.array([x.close for x in X_val]).reshape((len(X_val), WINDOW_LONG, 1))

# X_test = np.array([x.close for x in X_test]).reshape((len(X_test), WINDOW_LONG, 1))

# Y_train, Y_val, Y_test = (
#     Y[: int(len(X) * 0.5)],
#     Y[int(len(X) * 0.6) : int(len(X) * 0.7)],
#     Y[int(len(X) * 0.8) :],
# )

# Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
triple_barrier_file_name = "spartan_made"
preprocessed_name = (
    "meta_labelling_" + triple_barrier_file_name + "_window_long=" + str(WINDOW_LONG)
)
# Writing preprocessed X,y
h5f = h5py.File("data/preprocessed/" + preprocessed_name + ".h5", "w")
h5f.create_dataset("X_train", data=X_train)
h5f.create_dataset("X_val", data=X_val)
h5f.create_dataset("X_test", data=X_test)
h5f.create_dataset("Y_train", data=Y_train)
h5f.create_dataset("Y_val", data=Y_val)
h5f.create_dataset("Y_test", data=Y_test)
h5f.close()
