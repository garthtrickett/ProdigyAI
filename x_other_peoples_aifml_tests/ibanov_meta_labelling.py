import sys
import os

sys.path.append("..")
import importlib
import plotnine as pn
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

import library.blackarbsceo_bars as brs
import mlfinlab as ml
from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from mlfinlab.features import fracdiff
import library.snippets as snp
import library.labelling
from library.labelling import (
    fixed_horizon_labelling,
    volatility_horizon_labelling,
    get_meta_barrier,
)
import library.features
from library.features import make_features_from_window, get_class_weights
import finance_ml

from finance_ml.stats.vol import *

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

from imblearn.over_sampling import SMOTE

#
# FROM HERE: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb

# read parquet file of dollar bars
data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()

data_ohlc = data.head(10000)

# Computing Dynamic Thresholds


def get_vol(prices, span=100, delta=pd.Timedelta(hours=1)):
    # 1. compute returns of the form p[t]/p[t-1] - 1
    # 1.1 find the timestamps of p[t-1] values
    df0 = prices.index.searchsorted(prices.index - delta)
    df0 = df0[df0 > 0]
    # 1.2 align timestamps of p[t-1] to timestamps of p[t]
    df0 = pd.Series(
        prices.index[df0 - 1], index=prices.index[prices.shape[0] - df0.shape[0] :]
    )
    # 1.3 get values by timestamps, then compute returns
    df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1
    # 2. estimate rolling standard deviation
    df0 = df0.ewm(span=span).std()
    return df0


# Adding Path Dependency: Triple-Barrier Method


def get_horizons(prices, delta=pd.Timedelta(minutes=60)):

    t1 = prices.index.searchsorted(prices.index + delta)
    t1 = t1[t1 < prices.shape[0]]
    t1 = prices.index[t1]
    t1 = pd.Series(t1, index=prices.index[: t1.shape[0]])
    return t1


def get_touches(prices, events, factors=[1, 1]):
    """
  events: pd dataframe with columns
    t1: timestamp of the next horizon
    threshold: unit height of top and bottom barriers
    side: the side of each bet
  factors: multipliers of the threshold to set the height of 
           top/bottom barriers
  """
    out = events[["t1"]].copy(deep=True)
    if factors[0] > 0:
        thresh_uppr = factors[0] * events["threshold"]
    else:
        thresh_uppr = pd.Series(index=events.index)  # no uppr thresh
    if factors[1] > 0:
        thresh_lwr = -factors[1] * events["threshold"]
    else:
        thresh_lwr = pd.Series(index=events.index)  # no lwr thresh
    for loc, t1 in events["t1"].iteritems():
        df0 = prices[loc:t1]  # path prices
        df0 = (df0 / prices[loc] - 1) * events.side[loc]  # path returns
        out.loc[loc, "stop_loss"] = df0[
            df0 < thresh_lwr[loc]
        ].index.min()  # earliest stop loss
        out.loc[loc, "take_profit"] = df0[
            df0 > thresh_uppr[loc]
        ].index.min()  # earliest take profit
    return out


def get_labels(touches):
    out = touches.copy(deep=True)
    # pandas df.min() ignores NaN values
    first_touch = touches[["stop_loss", "take_profit"]].min(axis=1)
    for loc, t in first_touch.iteritems():
        if pd.isnull(t):
            out.loc[loc, "label"] = 0
        elif t == touches.loc[loc, "stop_loss"]:
            out.loc[loc, "label"] = -1
        else:
            out.loc[loc, "label"] = 1
    return out


data_ohlc = data_ohlc.set_index("date_time")
data_ohlc = data_ohlc.apply(pd.to_numeric)
data_ohlc.index = pd.to_datetime(data_ohlc.index)
data_ohlc = data_ohlc.assign(threshold=get_vol(data_ohlc.close)).dropna()

data_ohlc = data_ohlc.assign(t1=get_horizons(data_ohlc)).dropna()
events = data_ohlc[["t1", "threshold"]]
events = events.assign(side=pd.Series(1.0, events.index))  # long only
touches = get_touches(data_ohlc.close, events, [1, 1])
touches = get_labels(touches)
data_ohlc = data_ohlc.assign(label=touches.label)

# Define X and y
X = data_ohlc[["open", "close", "high", "low"]].values
y = np.squeeze(data_ohlc[["label"]].values)
# X_train, y_train = X[:4500], y[:4500]
# X_test, y_test = X[4500:], y[4500:]

data_ohlc[["label"]].values
np.unique(data_ohlc[["label"]].values, return_counts=True)

# Split training data
X_train, X_test = X[: int(len(X) * 0.8)], X[int(len(X) * 0.8) :]
y_train, y_test = y[: int(len(y) * 0.8)], y[int(len(y) * 0.8) :]

# Even out training samples bin counts  (oversampling)
sm = SMOTE()
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
unique, counts = np.unique(y_train_res, return_counts=True)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)


def true_binary_label(y_pred, y_test):
    bin_label = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        if y_pred[i] != 0 and y_pred[i] * y_test[i] > 0:
            bin_label[i] = 1  # true positive
    return bin_label


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(true_binary_label(y_pred, y_test), y_pred != 0)

# generate predictions for training set
y_train_pred = clf.predict(X_train)
# add the predictions to features
X_train_meta = np.hstack([y_train_pred[:, None], X_train])
X_test_meta = np.hstack([y_pred[:, None], X_test])
# generate true meta-labels
y_train_meta = true_binary_label(y_train_pred, y_train)
# rebalance classes again
sm = SMOTE()
X_train_meta_res, y_train_meta_res = sm.fit_sample(X_train_meta, y_train_meta)
model_secondary = LogisticRegression().fit(X_train_meta_res, y_train_meta_res)
y_pred_meta = model_secondary.predict(X_test_meta)
# use meta-predictions to filter primary predictions
cm = confusion_matrix(true_binary_label(y_pred, y_test), (y_pred * y_pred_meta) != 0)

