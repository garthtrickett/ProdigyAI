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

loss_function = "categorical_crossentropy"
scoring = "neg_log_loss"

t1_data = data.dropna(subset=["t1"])
t1 = t1_data["t1"]

unindex_data = t1_data.reset_index()
unindex_data["x"] = np.random.randint(0, 5, size=len(unindex_data))
unindex_data["y"] = y
if len(sample_weights) > 1:
    unindex_data["sample_weights"] = sample_weights

x_and_y = unindex_data[["x", "y"]]
pred_times = unindex_data["date_time"]
eval_times = unindex_data["t1"]

# Hyperparameters
# dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
#                          name='learning_rate')
# dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=511, high=513, name="num_input_nodes")
# dim_num_dense_nodes = Integer(low=1, high=28, name='num_dense_nodes')
# dim_activation = Categorical(categories=['relu', 'sigmoid'],
#                              name='activation')
dim_batch_size = Integer(low=31, high=32, name="batch_size")
# dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")

dimensions = [
    #   dim_learning_rate,
    #   dim_num_dense_layers,
    dim_num_input_nodes,
    #   dim_num_dense_nodes,
    #   dim_activation,
    dim_batch_size
    # dim_adam_decay,
]

default_parameters = [512, 31]

# checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
# es = EarlyStopping(monitor="val_loss", patience=2)
# tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))


def gru_model(num_input_nodes, batch_size):

    main_input = Input(batch_shape=(batch_size, X.shape[1], 1),
                       name="main_input")
    inputs = main_input

    x = GRU(units=20, return_sequences=True)(inputs)
    x = GRU(20, return_sequences=True)(x)
    x = GRU(20, return_sequences=False)(x)
    # x = GRU(5, return_sequences=False)(x)
    predictions = Dense(3, activation="softmax")(x)
    model = Model(inputs=[main_input], outputs=predictions)
    model.compile(loss=loss_function,
                  optimizer="rmsprop",
                  metrics=["accuracy"])

    return model


def lstm_model(num_input_nodes, batch_size):
    main_input = Input(batch_shape=(batch_size, X.shape[1], 1),
                       name="main_input")
    inputs = main_input
    auxiliary_input = Input(batch_shape=(batch_size, 1), name="aux_input")
    x = LSTM(units=128,
             return_sequences=True,
             input_shape=[inputs.shape[1], 1])(inputs)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128)(x)
    predictions = Dense(3)(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=loss_function,
                  optimizer="rmsprop",
                  metrics=["accuracy"])
    return model


scaler = StandardScaler()
# scaler = MinMaxScaler()

fold_loss_list = []
fold_accuracy_list = []
best_models_list = []

# Create instance of PurgedWalkForwardCV class
cv = PurgedWalkForwardCV(n_splits=3,
                         n_test_splits=1,
                         min_train_splits=1,
                         max_train_splits=1)


@use_named_args(dimensions=dimensions)
def fitness(num_input_nodes, batch_size):

    model = gru_model(num_input_nodes=num_input_nodes, batch_size=batch_size)

    # named blackbox becuase it represents the structure
    blackbox = model.fit(
        x=features_train,
        y=Y_train,
        epochs=1,
        batch_size=batch_size,
        sample_weight=oversampled_sample_weights,
        validation_data=(features_val, Y_val, fold_sample_weights_validation),
    )
    # return the validation accuracy for the last epoch.

    accuracy = blackbox.history["val_accuracy"][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tf.compat.v1.reset_default_graph

    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy


## NEURAL NETWORKS
row = 1
# loop through each fold and tune hyper parameters on training set and evaluate on test sets
print("cv started")
for train_set, test_set in cv.split(X=x_and_y,
                                    y=x_and_y.y,
                                    pred_times=pred_times,
                                    eval_times=eval_times):
    print(row)
    print((train_set))
    print((test_set))
    X = X.reshape(X.shape[0], X.shape[1])
    X_train = make_x_from_set(train_set, X)
    X_train = X_train[~np.isnan(X_train).any(axis=1)]

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    Y_train = y[train_set]

    if len(sample_weights) > 1:
        X_train = make_x_and_sample_weights(sample_weights[train_set], X_train)

    # Doesn't seem to be a way to input class weights for either evaluate or predict keras functions
    # Options
    # 1. only use class weights at training time (is this ok?!?)
    # 2. Don't use class weights and instead use smote (oversampling)
    # which means I would need to oversample the sample weights aswell (doesn't seem possible)
    # 3. Don't use sample weights or class weights and just use smote instead
    # 4. Configure the preprocessing in a certain way so that the
    # classes come out equally for both side and size and then using only sample weights
    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(Y_train),
                                                      Y_train)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                                                      Y_train,
                                                      test_size=0.2,
                                                      random_state=1)

    # sm = SMOTE()
    sm = ADASYN()  # adds in randomness on top of smote
    Y_train = np.asarray(Y_train)
    unique, counts = np.unique(Y_train, return_counts=True)
    try:
        X_train, Y_train = sm.fit_sample(X_train, Y_train)
    except Exception as e:
        print("Either already balanced or couldn't balance")

    if len(sample_weights) > 1:
        oversampled_sample_weights = np.zeros(len(X_train))
        oversampled_sample_weights[:] = np.nan
        new_X_train = np.zeros((X_train.shape[0], (X_train.shape[1] - 1)))
        new_X_train[:] = np.nan
        for i in range(len(X_train)):
            oversampled_sample_weights[i] = X_train[i][-1]
            new_X_train[i] = X_train[i][0:-1]

        X_train = new_X_train

        new_X_val = np.zeros((X_val.shape[0], (X_val.shape[1] - 1)))
        new_X_val[:] = np.nan
        for i in range(len(X_val)):
            new_X_val[i] = X_val[i][0:-1]

        X_val = new_X_val

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    if len(sample_weights) > 1:
        fold_sample_weights = sample_weights[train_set]
        fold_sample_weights_validation = fold_sample_weights[-len(Y_val):]

    features_train = {"main_input": X_train}
    features_val = X_val
    unique, thecount = np.unique(Y_train, return_counts=True)
    Y_train = binarize_y_side(Y_train)
    Y_val = binarize_y_side(Y_val)

    if len(sample_weights) == 1:
        oversampled_sample_weights = None
        fold_sample_weights_validation = None

    print("starting search for row " + str(row))
    # # run the trials gradient boosted regression trees
    gbrt_result = gbrt_minimize(
        func=fitness,
        dimensions=dimensions,
        n_calls=11,
        n_jobs=-1,
        x0=default_parameters,
    )
    # # run the trials gaussian process
    # gp_result = gp_minimize(
    #     func=fitness,
    #     dimensions=dimensions,
    #     n_calls=12,
    #     noise=0.01,
    #     n_jobs=-1,
    #     kappa=5,
    #     x0=default_parameters,
    # )
    print("finished search for row " + str(row))

    # Get the best model
    model = gru_model(gbrt_result.x[0], gbrt_result.x[1])
    print("best model picked for row " + str(row))

    X_test = make_x_from_set(test_set, X)
    X_test = X_test[~np.isnan(X_test).any(axis=1)]
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_test = y[test_set]

    features_test = {"main_input": X_test}
    Y_test = binarize_y_side(Y_test)

    print("evaluating best model for row " + str(row))
    if len(sample_weights) > 1:
        fold_sample_weights = sample_weights[test_set]
    else:
        fold_sample_weights = None
    evaluation = model.evaluate(features_test,
                                Y_test,
                                sample_weight=fold_sample_weights)

    fold_loss_list.append(evaluation[0])
    fold_accuracy_list.append(evaluation[1])
    best_models_list.append(model)

    row = row + 1

best_model_index_number = fold_accuracy_list.index(min(fold_accuracy_list))
best_model = best_models_list[best_model_index_number]

# save the sklearn scaler
from sklearn.externals import joblib

scaler_filename = "data/gpu_output/scaler_one.save"
joblib.dump(scaler, scaler_filename)

average_loss = sum(fold_loss_list) / (row - 1)
average_accuracy = sum(fold_accuracy_list) / (row - 1)

print("cv finished")
# Do you we want to pass probabilities to the second model or classes?
model = best_model
X = X.reshape(X.shape[0], X.shape[1])
normed_x = scaler.transform(X)
normed_x = normed_x.reshape(normed_x.shape[0], normed_x.shape[1], 1)
X = X.reshape(X.shape[0], X.shape[1], 1)
P = model.predict(normed_x)
P = P.argmax(axis=-1)
unique, counts = np.unique(P, return_counts=True)
unique, counts = np.unique(y, return_counts=True)
# Writing preprocessed X,y
h5f = h5py.File("data/gpu_output/" + "gpu_temp_name" + ".h5", "w")
h5f.create_dataset("X", data=X)
h5f.create_dataset("P", data=P)
h5f.create_dataset("sample_weights", data=sample_weights)
h5f.create_dataset("sampled_idx_epoch", data=sampled_idx_epoch)
h5f.close()

table = pa.Table.from_pandas(data)
pq.write_table(
    table,
    "data/gpu_output/" + "gpu_temp_name" + "_data.parquet",
    use_dictionary=True,
    compression="snappy",
    use_deprecated_int96_timestamps=True,
)

model.save("data/gpu_output/my_model_stage_one.h5"
           )  # creates a HDF5 file 'my_model.h5'

with open("temp/data_name_gpu.txt", "w+") as text_file:
    text_file.write("gpu_temp_name")

print("mlfinlab_gpu_finished")

print("stage one finished")