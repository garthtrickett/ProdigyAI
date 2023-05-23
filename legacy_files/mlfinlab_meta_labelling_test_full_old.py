import sys
import os

sys.path.append("..")
import importlib
import plotnine as pn
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

cpus = cpu_count() - 1

# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"

from third_party_libraries.mlfinlab.filters.filters import cusum_filter

#
# FROM HERE: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb

# read parquet file of dollar bars
data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()
# 0.12980103492736816 much much faster using parquets should replace the csv parts of the third_party_libraries.mlfinlab

data = data.head(100000)
data = data.set_index("date_time")
data.index = pd.to_datetime(data.index)
# Check for duplicates
# duplicates_index_ = data.index.duplicated(keep='first')
# unique, counts = np.unique(duplicates_index_, return_counts=True)


# # Fractional differentiation
# data_series = data['close'].to_frame()
# # # Log the prices
# log_prices = np.log(data_series)
# d = 0.4
# fd_series = fracdiff.frac_diff_ffd(log_prices, diff_amt=d, thresh=1e-5)
# # fd_series.head(1480)
# fd_series.dropna()

bar_name = "btcusdt_agg_trades_10_tick_bars"

# Triple Barrier Parameters
vol_span = 100
vol_seconds = 3600
vol_modifier = 1
vertical_num_days = 1
pt_sl = [2, 2]
min_ret = 0.0001

triple_barrier_file_name = (
    "bar_name="
    + bar_name
    + "_vol_span="
    + str(vol_span)
    + "_vol_seconds="
    + str(vol_seconds)
    + "_vol_modifier="
    + str(vol_modifier)
    + "_vertical_num_days="
    + str(vertical_num_days)
    + "_pt="
    + str(pt_sl[0])
    + "_pt="
    + str(pt_sl[1])
    + "_pt="
    + str(min_ret)
)

# Get volatility

vol = get_vol(data["close"], span=vol_span, seconds=vol_seconds)

# alternative measure of volatility
# vol = data['close'].std()

# third volatility measure
# Compute daily volatility
# vol = ml.util.get_daily_vol(close=data['close'], lookback=100)

# # modulate the profit taking margin #
vol *= vol_modifier


start = time.time()
num_threads = 6
df1 = mp_pandas_obj(
    func=cusum_filter,
    pd_obj=("molecule", data["close"].index),
    num_threads=num_threads,
    raw_time_series=data["close"],
    threshold=vol.mean(),
)
count = 0
for dt_index in df1:
    if count != 0:
        last_index = last_index.append(dt_index)
    else:
        last_index = dt_index
    count = count + 1

cusum_events = last_index
end = time.time()
print(end - start)

# Generate cusum event filter # takes a long tie but less than triple barrier

# Compute vertical barrier

start = time.time()
vertical_barriers = ml.labeling.add_vertical_barrier(
    t_events=cusum_events, close=data["close"], num_days=vertical_num_days
)
end = time.time()
print(end - start)

# Triple barrier ## takes ages (can up the threads)
start = time.time()
triple_barrier_events = ml.labeling.get_events(
    close=data["close"],
    t_events=cusum_events,
    pt_sl=pt_sl,
    target=vol,
    min_ret=min_ret,
    num_threads=6,
    vertical_barrier_times=vertical_barriers,
)

end = time.time()
print(end - start)

# Saving some data so we don't have to run the triple barrier labelling every time
table = pa.Table.from_pandas(triple_barrier_events)
pq.write_table(
    table,
    "data/barrier_events/" + triple_barrier_file_name + ".parquet",
    use_dictionary=True,
    compression="snappy",
)

# Reading the triple barrier events
table = pq.read_table("data/barrier_events/" + triple_barrier_file_name + ".parquet")
triple_barrier_events = table.to_pandas()

# create labels
labels = ml.labeling.get_bins(triple_barrier_events, data["close"])
clean_labels = ml.labeling.drop_labels(labels)
clean_labels.bin.value_counts()


start = time.time()
X_and_y = clean_labels.join(data)

# X_and_y['bin'].shift(1) # shifting the label by one

X_and_y.drop(["open", "high", "low", "ret", "trgt"], axis=1, inplace=True)

for col in X_and_y.columns:  # go through all of the columns
    if col == "close":  #
        X_and_y[col] = X_and_y[col].pct_change()
        X_and_y = X_and_y.replace([np.inf, -np.inf], np.nan)
        X_and_y.dropna(inplace=True)  # remove the nas created by norming

        X_and_y[col] = preprocessing.scale(X_and_y[col].values)
        X_and_y.dropna(inplace=True)

X_and_y.reset_index(inplace=True)


WINDOW_LONG = 100
X, Y, timestamp_index = [], [], []
for i, row in X_and_y.iterrows():
    one_hot_list = []
    if i > WINDOW_LONG:
        window = X_and_y.iloc[i - WINDOW_LONG : i]
        X.append(window)
        if row["bin"] == -1:
            one_hot_list = [1, 0, 0]
        elif row["bin"] == 0:
            one_hot_list = [0, 1, 0]
        elif row["bin"] == 1:
            one_hot_list = [0, 0, 1]
        Y.append(one_hot_list)


X_train, X_val, X_test = (
    X[: int(len(X) * 0.5)],
    X[int(len(X) * 0.6) : int(len(X) * 0.7)],
    X[int(len(X) * 0.8) :],
)

X_train = np.array([x.close for x in X_train]).reshape((len(X_train), WINDOW_LONG, 1))

X_val = np.array([x.close for x in X_val]).reshape((len(X_val), WINDOW_LONG, 1))

X_test = np.array([x.close for x in X_test]).reshape((len(X_test), WINDOW_LONG, 1))

Y_train, Y_val, Y_test = (
    Y[: int(len(X) * 0.5)],
    Y[int(len(X) * 0.6) : int(len(X) * 0.7)],
    Y[int(len(X) * 0.8) :],
)

Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)

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


# Reading preprocessed X,y
h5f = h5py.File("data/preprocessed/" + preprocessed_name + ".h5", "r")
X_train = h5f["X_train"][:]
X_val = h5f["X_val"][:]
X_test = h5f["X_test"][:]
Y_train = h5f["Y_train"][:]
Y_val = h5f["Y_val"][:]
Y_test = h5f["Y_test"][:]
h5f.close()


end = time.time()
print(end - start)

# 1. read in the dollar bars
# 2. frac diff
# 3. do the sampling tricks (seq boostrap + time decay)
# 4  make a model that takes in feature window and outputs buy/sell
# 5 make a copy of the dataframe and then filter out the rows where neither buy nor sell is given
# 7  put the side in as an input into the second model and then it goes throuugh
#  cusum event filter => vertical barrier => triple_barrier. This will output a bin of 0 or 1 which indicates
# wether that side trigger should actually be taken (would it of made profit above some threshold)
# run a second model with the X being the features from model one + the side and the Y being the bin result of get_bins
# also make sure the classes are equal (could use oversampling)


# NEXT:
# implement the sentdex model saving and tensorflow
# get this working with cpu/gpu preemptible automatic creation/stopping

# to make this notebook's output stable across runs

run_name = preprocessed_name + "model_type=LSTM"

np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# LSTM
model = keras.models.Sequential(
    [
        keras.layers.LSTM(128, return_sequences=True, input_shape=[100, 1]),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.LSTM(256, return_sequences=True),
        keras.layers.LSTM(128),
        keras.layers.Dense(3),
        keras.layers.Activation(activation="softmax"),
    ]
)

# # GRU
# model = keras.models.Sequential(
#     [
#         keras.layers.GRU(20, return_sequences=True, input_shape=[100, 1]),
#         keras.layers.GRU(20, return_sequences=True),
#         keras.layers.GRU(20),
#         keras.layers.Dense(3),
#         keras.layers.Activation(activation="softmax"),
#     ]
# )

# # Simpliefied wavenet
# model = keras.models.Sequential()
# model.add(keras.layers.InputLayer(input_shape=[100, 1]))
# for rate in (1, 2, 4, 8) * 2:
#     model.add(
#         keras.layers.Conv1D(
#             filters=20,
#             kernel_size=2,
#             padding="causal",
#             activation="relu",
#             dilation_rate=rate,
#         )
#     )
# model.add(keras.layers.Conv1D(filters=10, kernel_size=1))

# Same for each model LSTM, GRU, Wavenet
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(monitor="val_loss", patience=2)
tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=16,
    verbose=True,
    validation_data=(X_val, Y_val),
    callbacks=[checkpointer, es],
)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# Metrics
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

# 100 thousands bars
# try different window sizes
# 10 epochs of patience (magnitudes higher epoch total)
# add recurrent dropout, add recurrent batch normalization, l1/l2 reg,
# add time duration feature (throw in everything in the dataset) make sure u have good regularization
# accuracy needs to be high enough that there is profit after fees


# add in config variables that are saved in the names of the data files.
