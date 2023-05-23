import sys
import os
cwd = os.getcwd()
sys.path.append("..")

import importlib
import plotnine as pn
import matplotlib.pyplot as plt
#try this is x11 fails

# import matplotlib
# # matplotlib.use("Qt5Agg")

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

run_name = 'fixed_horizon_with_two'

h5f = h5py.File(
    '/home/garthtrickett_gmail_com/ProdigyAI/spartan/data/' + run_name + '.h5',
    'r')
X_train = h5f['X_train'][:]
X_val = h5f['X_val'][:]
X_test = h5f['X_test'][:]
Y_train = h5f['Y_train'][:]
Y_val = h5f['Y_val'][:]
Y_test = h5f['Y_test'][:]
h5f.close()

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1,
             val_loss,
             "r.-",
             label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 2])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)


np.random.seed(42)
tf.random.set_seed(42)

# LSTM
model = keras.models.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=[10, 1]),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.LSTM(256, return_sequences=True),
    keras.layers.LSTM(128),
    keras.layers.Dense(3),
    keras.layers.Activation(activation='softmax')
])

# # GRU
# model = keras.models.Sequential([
#     keras.layers.GRU(20, return_sequences=True, input_shape=[100, 1]),
#     keras.layers.GRU(20, return_sequences=True),
#     keras.layers.GRU(20),
#     keras.layers.Dense(3),
#     keras.layers.Activation(activation='softmax')
# ])

# # Simpliefied wavenet
# model = keras.models.Sequential()
# model.add(keras.layers.InputLayer(input_shape=[100, 1]))
# for rate in (1, 2, 4, 8) * 2:
#     model.add(
#         keras.layers.Conv1D(filters=20,
#                             kernel_size=2,
#                             padding="causal",
#                             activation="relu",
#                             dilation_rate=rate))
# model.add(keras.layers.Conv1D(filters=10, kernel_size=1))

# Same for each model LSTM, GRU, Wavenet
checkpointer = ModelCheckpoint(filepath="test.hdf5",
                               verbose=0,
                               save_best_only=True)
es = EarlyStopping(monitor='val_loss', patience=4)

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

NAME = "fixed_horizon_test"
tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}".format(NAME))

history = model.fit(
    X_train,
    Y_train,
    epochs=5,
    batch_size=16,
    verbose=True,
    validation_data=(X_val, Y_val),
    callbacks=[checkpointer, es, tensorboard],
)

plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# Metrics
pred = model.predict(X_val)

print(
    classification_report([np.argmax(y) for y in Y_val],
                          [np.argmax(y) for y in pred]))

print(
    confusion_matrix([np.argmax(y) for y in Y_val],
                     [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_val], bins=5, alpha=0.5, label='Test data')
plt.hist([np.argmax(y) for y in pred],
         bins=5,
         alpha=0.5,
         label='Predicted data')
plt.legend()
plt.show()

pred = model.predict(X_test)

print(
    classification_report([np.argmax(y) for y in Y_test],
                          [np.argmax(y) for y in pred]))

print(
    confusion_matrix([np.argmax(y) for y in Y_test],
                     [np.argmax(y) for y in pred]))

plt.plot()
plt.hist([np.argmax(y) for y in Y_test], bins=5, alpha=0.5, label='Test data')
plt.hist([np.argmax(y) for y in pred],
         bins=5,
         alpha=0.5,
         label='Predicted data')
plt.legend()
plt.show()

model.save("models/{}".format(NAME))

# 100 thousands bars
# try different window sizes
# 10 epochs of patience (magnitudes higher epoch total)
# add recurrent dropout, add recurrent batch normalization
# add time duration feature (throw in everything in the dataset) make sure u have good regularization
# accuracy needs to be high enough that there is profit after fees

# change the csv parts to parquet
# add in config variables that are saved in the names of the data files.
