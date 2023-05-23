# load packages
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    Activation,
    Input,
    Reshape,
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras.backend import set_session
from keras.utils import np_utils
import matplotlib.pyplot as plt
import math
from numba import njit, prange
import yaml

# set random seeds
np.random.seed(1)
tf.random.set_seed(2)

import keras
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import h5py
import wandb

# check if using gpu
gpus = tf.config.list_physical_devices()
any_gpus = [s for s in gpus if "GPU" in s[0]]

import sys
import os
import argparse

sys.path.append("..")
cwd = os.getcwd()
from pathlib import Path

home = str(Path.home())
sys.path.append(home + "/ProdigyAI")

# Sorting out whether we are using the ipython kernel or not
try:
    resuming = "NA"
    get_ipython()
    check_if_ipython = True
    path_adjust = "../"

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("-s",
                        "--stage",
                        type=str,
                        help="Stage of Preprocesssing")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="one_model or two_model")
    parser.add_argument("-f",
                        "--resuming",
                        type=str,
                        help="Is this a continuation of preempted instance?")
    args = parser.parse_args()
    if args.resuming != None:
        resuming = args.resuming
    else:
        resuming = "NA"
    if args.stage != None:
        arg_parse_stage = 1
        if int(args.stage) == 1:
            if os.path.exists(path_adjust + "temp/data_name_gpu.txt"):
                os.remove(path_adjust + "temp/data_name_gpu.txt")
                print("removed temp/data_name_gpu.txt")
            else:
                print("The file does not exist")

    if args.model != None:
        model = args.model
    path_adjust = ""

if cwd == home + "/":
    cwd = cwd + "/ProdigyAI"
    path_adjust = cwd

# Init wandb
yaml_path = path_adjust + "yaml/deeplob.yaml"
with open(yaml_path) as file:
    yaml_dict = yaml.load(file, Loader=yaml.FullLoader)

config_dictionary = dict(yaml=yaml_path, params=yaml_dict)
wandb.init(dir="~/ProdigyAI/",
           project="prodigyai",
           config=config_dictionary,
           resume=True)

window_length = wandb.config['params']['window_length']['value']
num_features = wandb.config['params']['num_features']['value']
epochs = wandb.config['params']['epochs']['value']
batch_size = wandb.config['params']['batch_size']['value']
number_of_lstm = wandb.config['params']['number_of_lstm']['value']

# limit gpu usage for keras
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
import tensorflow.python.keras.backend as K

sess = K.get_session()

## lob FI-2010 DATA PREPERATION
# The first 40 columns of the FI-2010 dataset are 10 levels ask and bid information
# for a limit order book and we only use these 40 features in our network.
# The last 5 columns of the FI-2010 dataset are the labels with different prediction horizons.

# def prepare_x(data):
#     df1 = data[:40, :].T
#     return np.array(df1)

# def get_label(data):
#     lob = data[-5:, :].T
#     return lob

# def data_classification(X, Y, T):
#     [N, D] = X.shape
#     df = np.array(X)

#     dY = np.array(Y)

#     dataY = dY[T - 1 : N]

#     dataX = np.zeros((N - T + 1, T, D))
#     for i in range(T, N + 1):
#         dataX[i - T] = df[i - T : i, :]

#     return dataX.reshape(dataX.shape + (1,)), dataY

# # please change the data_path to your local path
# data_path = (
#     path_adjust
#     + "third_party_libraries/gam_rhn/95-FI2010/data/BenchmarkDatasets/NoAuction"
# )

# dec_train = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Training/Train_Dst_NoAuction_DecPre_CF_7.txt"
# )
# dec_test1 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_7.txt"
# )
# dec_test2 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_8.txt"
# )
# dec_test3 = np.loadtxt(
#     data_path
#     + "/3.NoAuction_DecPre/NoAuction_DecPre_Testing/Test_Dst_NoAuction_DecPre_CF_9.txt"
# )

# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

# # extract limit order book data from the FI-2010 dataset
# train_lob = prepare_x(dec_train)
# test_lob = prepare_x(dec_test)

# mins, maxes, means, stds = get_feature_stats(train_lob)

# # extract label from the FI-2010 dataset
# train_label = get_label(dec_train)
# test_label = get_label(dec_test)

# # prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
# trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=100)
# trainY_CNN = trainY_CNN[:,3] - 1
# trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

# # prepare test data.
# testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=100)
# testY_CNN = testY_CNN[:,3] - 1
# testY_CNN = np_utils.to_categorical(testY_CNN, 3)

# h5f = h5py.File(path_adjust + "data/lob_2010/train_and_test_deeplob.h5", "w")
# h5f.create_dataset("trainX_CNN", data=trainX_CNN)
# h5f.create_dataset("trainY_CNN", data=trainY_CNN)
# h5f.create_dataset("testX_CNN", data=testX_CNN)
# h5f.create_dataset("testY_CNN", data=testY_CNN)
# h5f.close()

# path = path_adjust + "data/lob_2010/train_and_test_deeplob.h5"
# h5f = h5py.File(path, "r")
# trainX_CNN = h5f["trainX_CNN"][:]
# trainY_CNN = h5f["trainY_CNN"][:]
# testX_CNN = h5f["testX_CNN"][:]
# testY_CNN = h5f["testY_CNN"][:]
# h5f.close()
# trainX_CNN.shape

# testX_CNN.shape (139488, 100, 40, 1)
# trainX_CNN.shape (254651, 100, 40, 1)
# np.mean(trainX_CNN) == 0.60001
# np.min(trainX_CNN) == 0
# np.mean(trainX_CNN) == 0.12389889685674725
# np.std(trainX_CNN) ==
# np.std(trainX_CNN) == 0.12757670786242864
# each 40 contains the 10 top bid and ask price and volumes for a timestep

## PRODIGY AI HOCKUS POCKUS START
from pathlib import Path
import h5py

home = str(Path.home())

file_name = "esugj36b.h5"
wandb.config.update({'dataset': file_name})
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
prices_for_window_index_array_train = h5f[
    "prices_for_window_index_array_train"][:]
prices_for_window_index_array_val = h5f["prices_for_window_index_array_val"][:]
prices_for_window_index_array_test = h5f[
    "prices_for_window_index_array_test"][:]
close_index_array_train = h5f["close_index_array_train"][:]
close_index_array_val = h5f["close_index_array_val"][:]
close_index_array_test = h5f["close_index_array_test"][:]
input_features_normalized_train = h5f["input_features_normalized_train"][:]
input_features_normalized_val = h5f["input_features_normalized_val"][:]
input_features_normalized_test = h5f["input_features_normalized_test"][:]
y_train = h5f["y_train"][:]
y_val = h5f["y_val"][:]
y_test = h5f["y_test"][:]
h5f.close()

for i in range(len(close_index_array_train)):
    if close_index_array_train[i] == prices_for_window_index_array_train[0]:
        offset = i
        break


@njit
def generate_x_numba(batch_size, dim, list_IDs_temp,
                     input_features_normalized):
    X = np.empty((batch_size, *dim))
    for i, ID in enumerate(list_IDs_temp):
        for k in prange(dim[0]):
            for l in prange(len(input_features_normalized)):
                X[i][k][l] = input_features_normalized[l][ID + k]

    # X[i, ] = self.input_features_normalized[ID+self.window_length, ]
    return X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))


@njit
def generate_y_numba(batch_size, n_classes, list_IDs_temp, y_data):

    y = np.empty((batch_size, n_classes))

    for i, ID in enumerate(list_IDs_temp):

        y[i, :] = y_data[ID]

    return y


class DataGenerator(tf.compat.v2.keras.utils.Sequence):
    def __init__(self,
                 checkpoint_path,
                 prices_for_window_index_array,
                 input_features_normalized,
                 y_data,
                 batch_size,
                 dim,
                 n_classes,
                 to_fit,
                 shuffle=True):
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path
        self.prices_for_window_index_array = prices_for_window_index_array
        self.input_features_normalized = input_features_normalized
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.prices_for_window_index_array))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):

        wandb.save(self.checkpoint_path)

        self.indexes = np.arange(len(self.prices_for_window_index_array))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):

        dim = self.dim
        batch_size = self.batch_size
        input_features_normalized = self.input_features_normalized

        X = generate_x_numba(batch_size, dim, list_IDs_temp,
                             input_features_normalized)

        return X

    def _generate_y(self, list_IDs_temp):

        batch_size = self.batch_size
        n_classes = self.n_classes
        y_data = self.y_data

        y = generate_y_numba(batch_size, n_classes, list_IDs_temp, y_data)

        return y


n_classes = 3
dim = (window_length, num_features)
to_fit = True

# for X_batch, y_batch in DataGenerator(prices_for_window_index_array_train,
#                                       input_features_normalized_train,
#                                       y_train,
#                                       batch_size,
#                                       dim,
#                                       n_classes,
#                                       to_fit,
#                                       shuffle=True):
#     import pdb
#     pdb.set_trace()

# for X_batch, y_batch in generator(X_train, y_train, 64):
#     print("ok")

## PRODIGY AI HOCKUS POCKUS END

checkpoint_path = path_adjust + "temp/cp.ckpt"
# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                              save_weights_only=True,
                              verbose=1)

check_point_file = Path(checkpoint_path)
if check_point_file.exists() and resuming == "resuming":
    print("weights loaded")
    model.load_weights(checkpoint_path)

train_generator = DataGenerator(checkpoint_path,
                                prices_for_window_index_array_train,
                                input_features_normalized_train,
                                y_train,
                                batch_size,
                                dim,
                                n_classes,
                                to_fit,
                                shuffle=True)

val_generator = DataGenerator(checkpoint_path,
                              prices_for_window_index_array_val,
                              input_features_normalized_val,
                              y_val,
                              batch_size,
                              dim,
                              n_classes,
                              to_fit,
                              shuffle=True)

steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

if wandb.run.resumed and False:
    wandb.restore("model-best.h5",
                  run_path="garthtrickett/prodigyai/" + wandb.run.id)
    pdb.set_trace()
    # restore the best model
    deeplob = keras.models.load_model(wandb.restore("model-best.h5").name)
    adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
    deeplob.compile(optimizer=adam,
                    loss="categorical_crossentropy",
                    metrics=["accuracy"])
else:
    ### Model Architecture
    def create_deeplob(T, NF, number_of_lstm):
        input_lmd = Input(shape=(T, NF, 1))

        # build the convolutional block
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        conv_first1 = Conv2D(32, (1, 10))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # build the inception module
        convsecond_1 = Conv2D(64, (1, 1), padding="same")(conv_first1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_1 = Conv2D(64, (3, 1), padding="same")(convsecond_1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

        convsecond_2 = Conv2D(64, (1, 1), padding="same")(conv_first1)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_2 = Conv2D(64, (5, 1), padding="same")(convsecond_2)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

        convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1),
                                    padding="same")(conv_first1)
        convsecond_3 = Conv2D(64, (1, 1), padding="same")(convsecond_3)
        convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

        convsecond_output = concatenate(
            [convsecond_1, convsecond_2, convsecond_3], axis=3)

        # use the MC dropout here
        conv_reshape = Reshape(
            (int(convsecond_output.shape[1]),
             int(convsecond_output.shape[3])))(convsecond_output)

        conv_lstm = layers.LSTM(number_of_lstm)(conv_reshape)

        # build the output layer
        out = Dense(3, activation="softmax")(conv_lstm)
        model = Model(inputs=input_lmd, outputs=out)
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        return model

    deeplob = create_deeplob(window_length, num_features, number_of_lstm)
deeplob.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=100,
    verbose=2,
    validation_data=val_generator,
    callbacks=[cp_callback,
               WandbCallback(save_model=True, monitor="loss")])

finished_weights_path = path_adjust + "temp/cp_end.ckpt"
deeplob.save_weights(finished_weights_path)

print("script finished")

# deeplob = create_deeplob(200, 40, 64)
# deeplob.fit(
#     X_train,
#     y_train,
#     epochs=1000,
#     batch_size=64,
#     verbose=2,
#     validation_data=(X_val, y_val),
# )

## THIS SHOULD BE COMBINED WITH LSTM LAYERS WITH stateful=True
# deeplob = create_deeplob(200, 40, 64)
# for i in range(200):
#     deeplob.fit(
#         X_train,
#         y_train,
#         epochs=1,
#         batch_size=64,
#         verbose=2,
#         validation_data=(X_val, y_val),
#         shuffle=False,
#     )
#     deeplob.reset_states()
