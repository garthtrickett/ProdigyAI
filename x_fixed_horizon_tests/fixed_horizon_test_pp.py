import sys
import os

cwd = os.getcwd()
sys.path.append("..")

import importlib
import matplotlib.pyplot as plt

# try this is x11 fails

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
from library.labelling import (
    fixed_horizon_labelling,
    volatility_horizon_labelling,
    get_meta_barrier,
)
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

# Sorting out whether we are using the ipython kernel or not
try:
    get_ipython()
    check_if_ipython = True

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)

#
# FROM HERE: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb

data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()

# data = data.head(100000)

FEATURES = ["close"]
WINDOW_LONG = 10
WINDOW_SHORT = 50
HORIZON = 2
N_BARS = len(data) - HORIZON
T = 0.00045

# more cpu's/ram doesnt seem to make a difference in the speed of this (Needs parallelization)
import time

start = time.time()

print("Peforming Horizontal Labelling")
X, labels = fixed_horizon_labelling(data, WINDOW_LONG, N_BARS, HORIZON, T)

# plt.figure()
# plt.hist(labels, bins=5, alpha=0.7)
# plt.show()

print("Peforming Horizontal Splitting")
X_train, X_val, X_test, Y_train, Y_val, Y_test = horizon_splitting(X, labels)

X_train = np.array([x.close for x in X_train]).reshape((len(X_train), WINDOW_LONG, 1))

X_val = np.array([x.close for x in X_val]).reshape((len(X_val), WINDOW_LONG, 1))

X_test = np.array([x.close for x in X_test]).reshape((len(X_test), WINDOW_LONG, 1))

run_name = "fixed_horizon_with_two"
# # Writing files for later
h5f = h5py.File("data/" + run_name + ".h5", "w")
h5f.create_dataset("X_train", data=X_train)
h5f.create_dataset("X_val", data=X_val)
h5f.create_dataset("X_test", data=X_test)
h5f.create_dataset("Y_train", data=Y_train)
h5f.create_dataset("Y_val", data=Y_val)
h5f.create_dataset("Y_test", data=Y_test)
h5f.close()

end = time.time()
print(end - start)
