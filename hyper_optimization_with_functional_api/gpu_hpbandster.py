print("script started")
import sys
import os
import argparse
from kerastuner.tuners import RandomSearch
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

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)

X = X.reshape(X.shape[0], X.shape[1], 1)

lbr = LabelBinarizer()
y = lbr.fit_transform((y))

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=1
)


def get_data():
    x_train = X_train
    y_train = Y_train
    x_val = X_val
    y_val = Y_val
    x_test = X_test
    y_test = Y_test
    return x_train, y_train, x_val, y_val, x_test, y_test


class KerasWorker(Worker):
    def __init__(self, N_train=8192, N_valid=1024, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 10

        # the data, split between train and test sets

        x_train, y_train, x_val, y_val, x_test, y_test = get_data()

        self.x_train, self.y_train = x_train, y_train
        self.x_validation, self.y_validation = x_val, y_val
        self.x_test, self.y_test = x_test, y_test

    def compute(self, config, budget, working_directory, *args, **kwargs):

        inputs = Input(batch_shape=(10, X.shape[1], 1))
        x = GRU(units=32, return_sequences=True)(inputs)
        x = GRU(20, return_sequences=True)(x)
        x = GRU(20, return_sequences=False)(x)
        predictions = Dense(3, activation="softmax")(x)
        model = Model(inputs=inputs, outputs=predictions)

        checkpointer = ModelCheckpoint(
            filepath="test.hdf5", verbose=0, save_best_only=True
        )
        es = EarlyStopping(monitor="val_loss", patience=2)
        tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(run_name))

        if config["optimizer"] == "Adam":
            optimizer = keras.optimizers.Adam(lr=config["lr"])
        else:
            optimizer = keras.optimizers.SGD(
                lr=config["lr"], momentum=config["sgd_momentum"]
            )

        model.compile(loss=loss_function, optimizer="rmsprop", metrics=["accuracy"])

        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=5,
            batch_size=self.batch_size,
            verbose=True,
            validation_data=(self.x_validation, self.y_validation),
            callbacks=[checkpointer, es, tensorboard],
        )

        train_score = model.evaluate(self.x_train, self.y_train, verbose=0)
        val_score = model.evaluate(self.x_validation, self.y_validation, verbose=0)
        test_score = model.evaluate(self.x_test, self.y_test, verbose=0)

        return {
            "loss": 1 - val_score[1],  # remember: HpBandSter always minimizes!
            "info": {
                "test accuracy": test_score[1],
                "train accuracy": train_score[1],
                "validation accuracy": val_score[1],
                "number of parameters": model.count_params(),
            },
        }

    @staticmethod
    def get_configspace():
        """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            "lr", lower=1e-6, upper=1e-1, default_value="1e-2", log=True
        )

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter("optimizer", ["Adam", "SGD"])

        sgd_momentum = CSH.UniformFloatHyperparameter(
            "sgd_momentum", lower=0.0, upper=0.99, default_value=0.9, log=False
        )

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, "SGD")
        cs.add_condition(cond)

        return cs


if __name__ == "__main__":
    worker = KerasWorker(run_id="0")
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory=".")
    print(res)
