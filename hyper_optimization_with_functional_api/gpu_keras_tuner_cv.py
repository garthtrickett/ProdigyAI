# -*- coding: utf-8 -*-
print("script started")

from hanging_threads import start_monitoring

monitoring_thread = start_monitoring(seconds_frozen=360, test_interval=100)

import sys
import os
import argparse
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from sklearn.model_selection import train_test_split

sys.path.append("..")
cwd = os.getcwd()
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

import tensorflow as tf

import pyarrow as pa
import pyarrow.parquet as pq

import time
import numpy as np
import h5py


from finance_ml.stats.vol import *


from imblearn.over_sampling import SMOTE, ADASYN

from multiprocessing import cpu_count


cpus = cpu_count() - 1

# Scikit-Learn ≥0.20 is required
import sklearn

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight

assert sklearn.__version__ >= "0.20"
# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from time import time
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


from finance_ml.model_selection import PurgedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

from library.core import *
from library.cross_validation import *

from imblearn.pipeline import Pipeline

from sklearn.base import TransformerMixin, BaseEstimator


arg_parse_stage = None

# Can set the model type here for if your using ipython
model_type = "two_model"


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

    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("-o", "--stage", type=str, help="Stage of Preprocesssing")
    parser.add_argument("-m", "--model", type=str, help="one_model or two_model")
    args = parser.parse_args()
    if args.stage != None:
        arg_parse_stage = 1
    if args.model != None:
        model_type = args.model

with open("temp/data_name.txt", "r") as text_file:
    preprocessed_name = text_file.read()

try:
    print(args.stage)
    arg_stage_given = 1

except Exception:
    print("No stage parsearg given")
    arg_stage_given = 0

if "second_stage" not in preprocessed_name or (
    arg_stage_given == 1 and args.stage == 1
):
    stage = 1
else:
    stage = 2


if arg_parse_stage == 1:
    stage = int(args.stage)

# Overides
# preprocessed_name = "model_name=one_model&WINDOW_LONG=100&pt=1&sl=1&min_ret=0.001&vertical_barrier_seconds=120&head=1000&volatility_max=0.0012&volatility_min=0.0011"
stage = 1
model_type = "one_model"

print("starting data load")

if stage == 1:
    # Reading preprocessed X,y
    preprocessed_name = preprocessed_name.replace("second_stage", "")
    h5f = h5py.File("data/preprocessed/" + preprocessed_name + ".h5", "r")
    sample_weights = h5f["sample_weights"][:]
    X = h5f["X"][:]
    y = h5f["y"][:]
    sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
    h5f.close()

    data = pq.read_pandas(
        "data/preprocessed/" + preprocessed_name + "_data.parquet"
    ).to_pandas()

else:
    h5f = h5py.File("data/preprocessed/" + preprocessed_name + ".h5", "r")
    X = h5f["X"][:]
    P = h5f["P"][:]

    sample_weights = h5f["sample_weights"][:]
    y = h5f["y"][:]
    data = pq.read_pandas(
        "data/preprocessed/" + preprocessed_name + "_data.parquet"
    ).to_pandas()
    h5f.close()

print("data load finished")

run_name = preprocessed_name + "model_type=LSTM"


np.random.seed(42)
tf.random.set_seed(42)


if stage == 1:
    loss_function = "categorical_crossentropy"
    scoring = "neg_log_loss"
else:
    loss_function = "sparse_categorical_crossentropy"
    scoring = "f1"


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


def gru_model(hp):
    if stage == 1:
        main_input = Input(batch_shape=(32, X.shape[1], 1), name="main_input")
        inputs = main_input
    if stage == 2:
        main_input = Input(batch_shape=(32, X.shape[1], 1), name="main_input")
        inputs = main_input
        auxiliary_input = Input(batch_shape=(32, 1), name="aux_input")
    x = GRU(units=20, return_sequences=True)(inputs)
    x = GRU(20, return_sequences=True)(x)
    x = GRU(20, return_sequences=False)(x)
    # x = GRU(5, return_sequences=False)(x)
    if stage == 2:
        x = keras.layers.concatenate([x, auxiliary_input])
    predictions = Dense(3, activation="softmax")(x)
    if stage == 1:
        model = Model(inputs=[main_input], outputs=predictions)
    elif stage == 2:
        model = Model(inputs=[main_input, auxiliary_input], outputs=predictions)
    model.compile(loss=loss_function, optimizer="rmsprop", metrics=["accuracy"])

    return model


def lstm_model(hp):
    if stage == 1:
        main_input = Input(batch_shape=(32, X.shape[1], 1), name="main_input")
        inputs = main_input
    if stage == 2:
        main_input = Input(batch_shape=(32, X.shape[1], 1), name="main_input")
        inputs = main_input
        auxiliary_input = Input(batch_shape=(32, 1), name="aux_input")
    x = LSTM(units=128, return_sequences=True, input_shape=[inputs.shape[1], 1])(inputs)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128)(x)
    if stage == 2:
        x = keras.layers.concatenate([x, auxiliary_input])
    predictions = Dense(3)(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss=loss_function, optimizer="rmsprop", metrics=["accuracy"])
    return model


# checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
# es = EarlyStopping(monitor="val_loss", patience=2)
# tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))


scaler = StandardScaler()
# scaler = MinMaxScaler()

fold_loss_list = []
fold_accuracy_list = []
best_models_list = []

# Create instance of PurgedWalkForwardCV class
cv = PurgedWalkForwardCV(
    n_splits=3, n_test_splits=1, min_train_splits=1, max_train_splits=1
)

row = 1
# loop through each fold and tune hyper parameters on training set and evaluate on test sets

# # testing gpu utilization without keras tuner / CV
# hp = 1
# model = lstm_model(hp)
# model.summary()
# X = X.reshape(X.shape[0], X.shape[1], 1)
# y = binarize_y_side(y)
# history = model.fit(
#     X,
#     y,
#     epochs=5,
#     verbose=True,
#     validation_split=0.1
#     # callbacks=[checkpointer, es],
# )
# # tuner = RandomSearch(  # Hyperband
# #     lstm_model,
# #     objective="val_accuracy",
# #     max_trials=1,
# #     executions_per_trial=1,
# #     directory="my_dir",
# #     project_name="helloworld",
# # )
# # tuner.search_space_summary()

# # tuner.search(
# #     X,  # [X_train, P_train]
# #     y,
# #     epochs=1,
# #     validation_split=0.1
# #     # class_weight=class_weights,
# #     # callbacks=[checkpointer, es, tensorboard],
# # )


print("cv started")
for train_set, test_set in cv.split(
    X=x_and_y, y=x_and_y.y, pred_times=pred_times, eval_times=eval_times
):
    print(row)
    print((train_set))
    print((test_set))
    tuner = Hyperband(
        gru_model,
        objective="val_accuracy",
        max_trials=1,
        executions_per_trial=1,
        directory="my_dir",
        project_name="helloworld",
    )
    tuner.search_space_summary()
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
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(Y_train), Y_train
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=1
    )

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
        fold_sample_weights_validation = fold_sample_weights[-len(Y_val) :]

    if stage == 1:
        features_train = {"main_input": X_train}
        features_val = X_val
        unique, thecount = np.unique(Y_train, return_counts=True)
        Y_train = binarize_y_side(Y_train)
        Y_val = binarize_y_side(Y_val)

    if stage == 2:
        P_train = P[train_set]
        fold_P_train = P_train[0 : len(X_train)]
        fold_P_val = P_train[-len(Y_val) :]
        # fold_P_train = binarize_p(fold_P_train)
        # fold_P_val = binarize_p(fold_P_val)

        features_train = {"main_input": X_train, "aux_input": fold_P_train}
        features_val = {"main_input": X_val, "aux_input": fold_P_val}

    if len(sample_weights) == 1:
        oversampled_sample_weights = None
        fold_sample_weights_validation = None

    print("starting search for row " + str(row))
    tuner.search(
        features_train,  # [X_train, P_train]
        Y_train,
        epochs=1,
        sample_weight=oversampled_sample_weights,
        validation_data=(features_val, Y_val, fold_sample_weights_validation),
        # class_weight=class_weights,
        # callbacks=[checkpointer, es, tensorboard],
    )
    print("finished search for row " + str(row))

    models = tuner.get_best_models(num_models=1)
    model = models[0]
    print("best model picked for row " + str(row))

    X_test = make_x_from_set(test_set, X)
    X_test = X_test[~np.isnan(X_test).any(axis=1)]
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_test = y[test_set]

    if stage == 1:
        features_test = {"main_input": X_test}
        Y_test = binarize_y_side(Y_test)

    if stage == 2:
        if len(sample_weights) > 1:
            fold_P_test = sample_weights[test_set]
        features_test = {"main_input": X_test, "aux_input": fold_P_test}

    print("evaluating best model for row " + str(row))
    if len(sample_weights) > 1:
        fold_sample_weights = sample_weights[test_set]
    else:
        fold_sample_weights = None
    evaluation = model.evaluate(
        features_test, Y_test, sample_weight=fold_sample_weights
    )

    fold_loss_list.append(evaluation[0])
    fold_accuracy_list.append(evaluation[1])
    best_models_list.append(model)

    row = row + 1

best_model_index_number = fold_accuracy_list.index(min(fold_accuracy_list))
best_model = best_models_list[best_model_index_number]


average_loss = sum(fold_loss_list) / (row - 1)
average_accuracy = sum(fold_accuracy_list) / (row - 1)


model = best_model
normed_x = scaler.transform(X)
normed_x = normed_x.reshape(normed_x.shape[0], normed_x.shape[1], 1)
X = X.reshape(X.shape[0], X.shape[1], 1)
P = model.predict(normed_x)
P = P.argmax(axis=-1)
unique, counts = np.unique(P, return_counts=True)

print("cv finished")

if stage == 1 and model_type != None and model_type == "two_model":
    # Do you we want to pass probabilities to the second model or classes?
    model = best_model
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

    with open("temp/data_name_gpu.txt", "w+") as text_file:
        text_file.write("gpu_temp_name")

print("mlfinlab_gpu_finished")

if stage == 1:
    print("stage one finished")

if stage == 2:
    print("stage two finished")


# time finished
finished_time = time.time() + 28800
with open("temp/gpu_finished.txt", "w+") as text_file:
    text_file.write(finished_time)

