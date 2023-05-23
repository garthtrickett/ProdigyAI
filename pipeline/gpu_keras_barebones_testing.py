# -*- coding: utf-8 -*-
print("script started")

import argparse
import datetime
import os
import sys
sys.path.append("..")
import time
import warnings
from multiprocessing import cpu_count

import h5py
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import shap

# Scikit-Learn ≥0.20 is required
import sklearn
import skopt

# TensorFlow ≥2.0-preview is required
import tensorflow as tf
import tensorflow.keras.backend as K
from eli5.permutation_importance import get_score_importances
from hanging_threads import start_monitoring
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from kerastuner.tuners import Hyperband, RandomSearch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import (
    LabelBinarizer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.utils import class_weight
from skopt import gbrt_minimize, gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.layers import (
    LSTM,
    AvgPool1D,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GaussianDropout,
    GaussianNoise,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Lambda,
    MaxPool1D,
    Reshape,
    SpatialDropout1D,
    concatenate,
)
from tensorflow.keras.models import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from third_party_libraries.finance_ml.model_selection import PurgedKFold
from third_party_libraries.finance_ml.stats.vol import *
from library.core import *
from third_party_libraries.cross_validation import *
from timeseriesAI.fastai_timeseries import *
from timeseriesAI.torchtimeseries.models import *

import torch
import torch.nn as nn

print('pytorch:', torch.__version__)
print('fastai :', fastai.__version__)

monitoring_thread = start_monitoring(seconds_frozen=360, test_interval=100)
cwd = os.getcwd()
cpus = cpu_count() - 1

# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True
except Exception:
    check_if_ipython = False
    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

with open("temp/data_name.txt", "r") as text_file:
    preprocessed_name = text_file.read()

# Reading preprocessed X,y
preprocessed_name = preprocessed_name.replace("second_stage", "")
h5f = h5py.File("data/preprocessed/" + preprocessed_name + ".h5", "r")
sample_weights = h5f["sample_weights"][:]
X = h5f["X"][:]
y = h5f["y"][:]
unique, counts = np.unique(y, return_counts=True)
sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
h5f.close()

data = pq.read_pandas("data/preprocessed/" + preprocessed_name +
                      "_data.parquet").to_pandas()

print("data load finished")
# FAST AI ?SHit

# extract data

# Using a portion of the data for testing purposes
# X = X[0:500]
# y = y[0:500]

unique, counts = np.unique(y, return_counts=True)
run_name = preprocessed_name + "model_type=LSTM"

np.random.seed(42)
tf.random.set_seed(42)

### BAREBONES TESTING
batch_size = 10
loss_function = "categorical_crossentropy"

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    shuffle=False)

# Check to see how balanced classes are
unique, counts = np.unique(Y_train, return_counts=True)
# Data Normalization
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train, Y_train = makeOverSamplesADASYN(X_train, Y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
Y_train = binarize_y_side(Y_train)


def one_d_wave_net_model(X, loss_function, batch_size):
    main_input = Input(batch_shape=(batch_size, X.shape[1], 1),
                       name="main_input")
    inputs = main_input
    no_conv_layer = True
    for rate in (1, 2, 4, 8) * 2:
        if no_conv_layer == True:
            conv_input = inputs
            no_conv_layer = False
        else:
            conv_input = conv_output
        conv_output = Conv1D(
            filters=20,
            kernel_size=2,
            padding="causal",
            activation="relu",
            dilation_rate=rate,
        )(conv_input)
    flattened_output = Flatten()(conv_output)
    predictions = Dense(3, activation="softmax")(flattened_output)
    model = Model(inputs=[main_input], outputs=predictions)
    model.compile(loss=loss_function,
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    return model


def lstm_model(X, loss_function, batch_size):
    main_input = Input(batch_shape=(batch_size, X.shape[1], 1),
                       name="main_input")
    inputs = main_input
    auxiliary_input = Input(batch_shape=(batch_size, 1), name="aux_input")
    x = LSTM(units=16, return_sequences=True, input_shape=[inputs.shape[1],
                                                           1])(inputs)
    x = LSTM(units=8)(x)
    predictions = Dense(3)(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=loss_function,
                  optimizer="adadelta",
                  metrics=["accuracy"])
    return model


def gru_model(X, loss_function, batch_size):
    main_input = Input(batch_shape=(batch_size, X.shape[1], 1),
                       name="main_input")
    inputs = main_input

    layer_one_output = GRU(units=10, return_sequences=True)(inputs)
    layer_two_output = GRU(units=10, return_sequences=True)(layer_one_output)
    layer_three_output = GRU(units=10,
                             return_sequences=False)(layer_two_output)
    # layer_four_output = GRU(5, return_sequences=False)(layer_three_output)
    predictions = Dense(3, activation="softmax")(layer_three_output)
    model = Model(inputs=[main_input], outputs=predictions)
    model.compile(loss=loss_function, optimizer="adam", metrics=["accuracy"])

    return model


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)

model = lstm_model(X_train, loss_function, batch_size)
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    validation_split=0.1,
    callbacks=[tensorboard_callback],
)

model.evaluate(
    X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
    binarize_y_side(Y_test),
    batch_size=batch_size,
)

# END BAREBONES TESTING