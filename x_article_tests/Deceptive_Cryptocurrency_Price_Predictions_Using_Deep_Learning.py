import sys
import os
sys.path.append("..")
import importlib
import plotnine as pn
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg") try this is x11 fails
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
from tensorflow.keras.layers import Input, concatenate, SpatialDropout1D, Flatten, AvgPool1D, LSTM, Lambda, Reshape, GaussianDropout
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, GaussianNoise, Conv1D, MaxPool1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
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
import mlfinlab as ml
from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from library.splitting import *
from mlfinlab.features import fracdiff
import library.snippets as snp
import library.labelling
from library.labelling import fixed_horizon_labelling, volatility_horizon_labelling, get_meta_barrier
import library.features
from library.features import make_features_from_window, get_class_weights
import finance_ml

from finance_ml.stats.vol import *

import multiprocessing as mp
from multiprocessing import cpu_count
cpus = cpu_count() - 1

# Scikit-Learn ≥0.20 is required #
import sklearn
assert sklearn.__version__ >= "0.20"
# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

import os
cwd = os.getcwd()

data = pq.read_pandas(
    'data/btcusdt_agg_trades_10_tick_bars.parquet').to_pandas()

del data['volume']
data.columns = ['time', 'open', 'high', 'low', 'close']

hist = data.head(10000)


# Train-Test Split
def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


def line_plot(line1, line2, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)


train, test = train_test_split(hist, test_size=0.1)
line_plot(train.close, test.close, 'training', 'test', 'BTC')


# Building the window
def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with
        respect to first entry.
    """
    return df / df.iloc[0] - 1


def extract_window_data(df, window=7, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of
        length `window`.
    """
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx:(idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, window=7, zero_base=True, test_size=0.1):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size)

    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)

    # extract targets
    y_train = train_data.close[window:].values
    y_test = test_data.close[window:].values
    if zero_base:
        y_train = y_train / train_data.close[:-window].values - 1
        y_test = y_test / test_data.close[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test


hist = hist.set_index('time')
hist = hist.apply(pd.to_numeric)
train, test, X_train, X_test, y_train, y_test = prepare_data(hist)


# Model
def build_lstm_model(input_data,
                     output_size,
                     neurons=20,
                     activ_func='linear',
                     dropout=0.25,
                     loss='mae',
                     optimizer='adam'):
    model = Sequential()
    model.add(
        LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


model = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=5, batch_size=4)

# Correlation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
# actual correlation
corr = np.corrcoef(actual_returns, predicted_returns)[0][1]
ax1.scatter(actual_returns, predicted_returns, color='k')
ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)
# shifted correlation
shifted_actual = actual_returns[:-1]
shifted_predicted = predicted_returns.shift(-1).dropna()
corr = np.corrcoef(shifted_actual, shifted_predicted)[0][1]
ax2.scatter(shifted_actual, shifted_predicted, color='k')
ax2.set_title('r = {:.2f}'.format(corr))
