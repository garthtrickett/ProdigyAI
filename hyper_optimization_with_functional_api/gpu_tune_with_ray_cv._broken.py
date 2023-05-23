print("script started")
import sys
import os
import argparse
from ray import tune
from ray.tune.integration.keras import TuneReporterCallback
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


smt = SMOTE()
# smt = ADASYN()  # adds in randomness on top of smote
# adasyn doesnt work with less than like 5000 samples


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
# stage = 1
# model_type = "two_model"

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
    X_and_P = h5f["X_and_P"][:]
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
    inputs = X
    scoring = "neg_log_loss"
else:
    loss_function = "sparse_categorical_crossentropy"
    inputs = X_and_P
    scoring = "f1"


# MinMaxScaler()  # normalization
# StandardScaler()  # standardization

t1_data = data.dropna(subset=["t1"])
t1 = t1_data["t1"]

unindex_data = t1_data.reset_index()
unindex_data["x"] = np.random.randint(0, 5, size=len(unindex_data))
unindex_data["y"] = y

x_and_y = unindex_data[["x", "y"]]
pred_times = unindex_data["date_time"]
eval_times = unindex_data["t1"]

# Create instance of PurgedWalkForwardCV class
cv = PurgedWalkForwardCV(
    n_splits=10, n_test_splits=1, min_train_splits=2, max_train_splits=9
)


def get_data():
    train_x = X_train
    train_y = Y_train
    test_x = X_test
    test_y = Y_test
    return train_x, train_y, test_x, test_y


class custom_callback(TuneReporterCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def on_train_batch_begin(self, *args, **kwargs):
        pass

    def on_train_batch_end(self, *args, **kwargs):
        pass


def gru_model(config):

    train_x, train_y, test_x, test_y = get_data()

    inputs = Input(batch_shape=(10, X.shape[1], 1))
    x = GRU(units=32, return_sequences=True)(inputs)
    x = GRU(20, return_sequences=True)(x)
    x = GRU(20, return_sequences=False)(x)
    predictions = Dense(3, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=predictions)

    # Same for each model LSTM, GRU, Wavenet
    checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
    es = EarlyStopping(monitor="val_loss", patience=2)
    tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))

    model.compile(loss=loss_function, optimizer="rmsprop", metrics=["accuracy"])

    history = model.fit(
        X_train,
        Y_train,
        epochs=5,
        batch_size=10,
        verbose=True,
        validation_data=(X_val, Y_val),
        callbacks=[checkpointer, es, tensorboard],
        # callbacks=[checkpointer, es, tensorboard, custom_callback(freq="epoch")],
    )


# loop through each fold and tune hyper parameters on training set and evaluate on test sets
row = 1
for train_set, test_set in cv.split(
    X=x_and_y, y=x_and_y.y, pred_times=pred_times, eval_times=eval_times
):

    X_train = make_x_from_set(train_set, X)
    X_train = X_train[~np.isnan(X_train).any(axis=1)]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    Y_train = y[train_set]

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=1
    )

    Y_train = binarize_y_side(Y_train)
    Y_val = binarize_y_side(Y_val)

    X_test = make_x_from_set(test_set, X)
    X_test = X_test[~np.isnan(X_test).any(axis=1)]
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_test = y[test_set]
    Y_test = binarize_y_side(Y_test)

    print("starting search for row " + str(row))
    results = tune.run(
        gru_model, config={"dense_1": 128, "dense_2": 256}, num_samples=1
    )
    print("finished search for row " + str(row))

    results.get_best_config(metric="mean_accuracy")
    print("best model picked for row " + str(row))

    print("evaluating best model for row " + str(row))

    all_configs = results.get_all_configs()
    rows = results._retrieve_rows(metric="validation_loss", mode="max")

    model.evaluate(X_test, Y_test)

    row = row + 1


# class_weights = class_weight.compute_class_weight("balanced", np.unique(y), y)


def create_model_lstm(optimizer="rmsprop", init_mode="uniform"):
    model = Sequential()
    model.add(
        LSTM(
            128,
            kernel_initializer=init_mode,
            return_sequences=True,
            input_shape=[inputs.shape[1], 1],
        )
    )
    model.add(LSTM(256, kernel_initializer=init_mode, return_sequences=True))
    model.add(LSTM(256, kernel_initializer=init_mode, return_sequences=True))
    model.add(LSTM(128, kernel_initializer=init_mode))
    model.add(Dense(3, kernel_initializer=init_mode))
    model.add(Activation(activation="softmax"))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    return model


# Same for each model LSTM, GRU, Wavenet
checkpointer = ModelCheckpoint(filepath="test.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(monitor="val_loss", patience=2)
tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))


print("clf_hyper_fit finished")

if stage == 2:
    size_predictions = clf.predict(X_and_P)
    unique, counts = np.unique(size_predictions, return_counts=True)
    unique, counts = np.unique(y, return_counts=True)

if stage == 1 and model_type != None and model_type == "two_model":
    P = clf.predict(X)
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
    )

    with open("temp/data_name_gpu.txt", "w+") as text_file:
        text_file.write("gpu_temp_name")

print("mlfinlab_gpu_finished")


if stage == 2:
    print("stage two finished")

