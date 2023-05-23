### LOAD PACKAGES
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import set_random_seed
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.layers import (
    Flatten,
    Dense,
    Dropout,
    Activation,
    Input,
    CuDNNLSTM,
    LSTM,
    Reshape,
    Conv2D,
    MaxPooling2D,
)
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.backend.tensorflow_backend import set_session
from keras.utils import np_utils
import matplotlib.pyplot as plt

# set random seeds
np.random.seed(1)
set_random_seed(2)

# limit gpu usage for keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


### DATA PREPERATION
# The first 40 columns of the FI-2010 dataset are 10 levels ask and bid information
# for a limit order book and we only use these 40 features in our network.
# The last 5 columns of the FI-2010 dataset are the labels with different prediction horizons.


def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1 : N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T : i, :]

    return dataX.reshape(dataX.shape + (1,)), dataY


# # please change the data_path to your local path
# data_path = "gam_rhn/95-FI2010/data/BenchmarkDatasets/NoAuction"


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

# # extract label from the FI-2010 dataset
# train_label = get_label(dec_train)
# test_label = get_label(dec_test)

# # prepare training data. We feed past 100 observations into our algorithms and choose the prediction horizon.
# trainX_CNN, trainY_CNN = data_classification(train_lob, train_label, T=100)
# trainY_CNN = trainY_CNN[:, 3] - 1
# trainY_CNN = np_utils.to_categorical(trainY_CNN, 3)

# # trainX_CNN.shape (254651, 100, 40, 1)

# # prepare test data.
# testX_CNN, testY_CNN = data_classification(test_lob, test_label, T=100)
# testY_CNN = testY_CNN[:, 3] - 1
# testY_CNN = np_utils.to_categorical(testY_CNN, 3)

# testX_CNN.shape (139488, 100, 40, 1)


# trainX_CNN.shape (254651, 100, 40, 1)
# each 40 contains the 10 top bid and ask price and volumes for a timestep


## PRODIGY AI HOCKUS POCKUS START
from pathlib import Path
import h5py

home = str(Path.home())

file_name = "arch=DLOB&name=two_model&WL=200&pt=1&sl=1&min_ret=9.523809523809525e-06&vbs=0.1&head=100000&skip=0&vol_max=9.543809523809525e-06&vol_min=9.533809523809525e-06&filter=none&cm_vol_mod=0&sw=on&fd=off&input=obook&ntb=True&tslbc=True.h5"
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
X_train = h5f["X_train"][:]
y_train = h5f["y_train"][:]
X_val = h5f["X_val"][:]
y_val = h5f["y_val"][:]
X_test = h5f["X_test"][:]
y_test = h5f["y_test"][:]
h5f.close()


## PRODIGY AI HOCKUS POCKUS END


### Model Architecture
def create_deeplob(T, NF, number_of_lstm):
    input_lmd = Input(shape=(T, NF, 1))

    # build the convolutional block
    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_lmd)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = Conv2D(32, (1, 10))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = Conv2D(32, (4, 1), padding="same")(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = Conv2D(64, (1, 1), padding="same")(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = Conv2D(64, (3, 1), padding="same")(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = Conv2D(64, (1, 1), padding="same")(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = Conv2D(64, (5, 1), padding="same")(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding="same")(conv_first1)
    convsecond_3 = Conv2D(64, (1, 1), padding="same")(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate(
        [convsecond_1, convsecond_2, convsecond_3], axis=3
    )

    # use the MC dropout here
    conv_reshape = Reshape(
        (int(convsecond_output.shape[1]), int(convsecond_output.shape[3]))
    )(convsecond_output)

    # build the last LSTM layer
    # conv_lstm = CuDNNLSTM(number_of_lstm)(conv_reshape)  ### FOR GPU
    conv_lstm = LSTM(number_of_lstm)(conv_reshape)  ### FOR CPU

    # build the output layer
    out = Dense(3, activation="softmax")(conv_lstm)
    model = Model(inputs=input_lmd, outputs=out)
    adam = keras.optimizers.Adam(lr=10, beta_1=0.9, beta_2=0.999, epsilon=1)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


deeplob = create_deeplob(100, 40, 64)
deeplob.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=64,
    verbose=2,
    validation_data=(X_val, y_val),
)
