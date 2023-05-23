print("script started")
import argparse
import os
import sys
import time
from multiprocessing import cpu_count
from time import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
# Scikit-Learn ≥0.20 is required
import sklearn
# TensorFlow ≥2.0-preview is required
import tensorflow as tf
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.layers import (LSTM, AvgPool1D, Conv1D, Dense, Dropout,
                                     Flatten, GaussianDropout, GaussianNoise,
                                     GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Input, Lambda,
                                     MaxPool1D, Reshape, SpatialDropout1D,
                                     concatenate)
from tensorflow.keras.models import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from third_party_libraries.finance_ml.model_selection import PurgedKFold
from third_party_libraries.finance_ml.stats.vol import *
from library.core import *

cwd = os.getcwd()


cpus = cpu_count() - 1


assert sklearn.__version__ >= "0.20"

assert tf.__version__ >= "2.0"


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


# GRU
def create_model_gru(optimizer="rmsprop"):
    model = Sequential()
    model.add(GRU(20, return_sequences=True, input_shape=[inputs.shape[1], 1]))
    model.add(GRU(20, return_sequences=True))
    model.add(GRU(20))
    model.add(Dense(3))
    model.add(Activation(activation="softmax"))
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])
    return model


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


# MinMaxScaler()  # normalization
# StandardScaler()  # standardization

start = time()
model = KerasClassifier(build_fn=create_model_lstm)
pipe_clf = MyPipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        # ("smt", smt), # If I want to use smote with sample weights i will
        # need to also oversample the sample weights somehow (somehwere along the pipeline)
        ("x_in_three_dims", XInThreeDims()),
        ("clf", model),
    ]
)


optimizers = ["Nadam"]
# optimizers = ["Adam"]
epochs = np.array([1])
batches = np.array([10])
# # init_mode = [
# #     "uniform",
#     "lecun_uniform",
# #     "normal",
# #     "zero",
#     "glorot_normal",
#     "glorot_uniform",
#     "he_normal",
#     "he_uniform"
# ]
init_mode = ["glorot_uniform"]
param_grid = dict(
    clf__optimizer=optimizers,
    clf__nb_epoch=epochs,
    clf__batch_size=batches,
    clf__init_mode=init_mode,
)

t1_data = data.dropna(subset=["t1"])
t1 = t1_data["t1"]

# Possible things to add to fit_params
# x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
# validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
# sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)

validation_split = 0.1

class_weights = class_weight.compute_class_weight("balanced", np.unique(y), y)
# class_weights = None
# sample_weights = None
callbacks_list = [checkpointer, es, tensorboard]
fit_params = {
    "clf__callbacks": callbacks_list,
    "clf__sample_weight": sample_weights,
    "clf__validation_split": validation_split,
    "clf__class_weight": class_weights,
    # "scaler__sampling_strategy": "not majority",
}
print("clf_hyper_fit starting")

clf = clf_hyper_fit(
    inputs,
    y,
    t1=t1,
    pipe_clf=pipe_clf,
    scoring=scoring,
    search_params=param_grid,
    n_splits=3,
    bagging=[1, 0.8, 0.8],  # set the first value to 0 to turn off bagging
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0.01,
    **fit_params
)

print("clf_hyper_fit finished")

# clf.get_params()
# clf.score(X[-100:-1], y[-100:-1])

# plot_learning_curves(history.history["loss"], history.history["val_loss"])
# plt.show()

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
