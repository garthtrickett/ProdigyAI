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

# Saving to send off to Harry
flat_x = X.flatten()  #(28901, 50)
flat_y = y.flatten()
np.savetxt("X.csv", flat_x, delimiter=",")
np.savetxt("y.csv", flat_y, delimiter=",")

unique, counts = np.unique(y, return_counts=True)
sampled_idx_epoch = h5f["sampled_idx_epoch"][:]
h5f.close()

data = pq.read_pandas("data/preprocessed/" + preprocessed_name +
                      "_data.parquet").to_pandas()

data = data.dropna()
print("data load finished")
# FAST AI ?SHit

# convert y labels from -1,0,1 to 0,1,2
# y = convert_y_labels_to_zero_index(y)

unique, counts = np.unique(y, return_counts=True)
run_name = preprocessed_name + "model_type=LSTM"

X = X[0:1000]
y = y[0:1000]

np.random.seed(42)
tf.random.set_seed(42)

# Splitting features and labels into train and test
X_train, X_valid, Y_train, Y_valid = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=1,
                                                      shuffle=False)

# This scales each X_train[i][0] against X_train[j][0] and  X_train[all other][0]
# as in X_train[i][3] against X_train[j][3] and  X_train[all other][3]
# scaler = MinMaxScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_valid = scaler.transform(X_valid)

# ### ROCKET
# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])

eps = 1e-6
# This normalization works over all the samples
# not just against others in the same 3rd dimension location
# X_train = (X_train - X_train.mean(axis=(1, 2), keepdims=True)) / (
#     X_train.std(axis=(1, 2), keepdims=True) + eps)
# X_valid = (X_valid - X_valid.mean(axis=(1, 2), keepdims=True)) / (
#     X_valid.std(axis=(1, 2), keepdims=True) + eps)

unique, counts = np.unique(Y_train, return_counts=True)

bs = 30
n_kernels = 10000
eps = 1e-6

_, features, seq_len = X_train.shape
model = ROCKET(features, seq_len, n_kernels=n_kernels, kss=[7, 9,
                                                            11]).to(device)
# shape of last dimension of X_train_tfm will be n_kernels * 2 || .float() is ~= torch.float32
X_train_tfm = model(torch.tensor(X_train, device=device).float()).unsqueeze(1)
X_valid_tfm = model(torch.tensor(X_valid, device=device).float()).unsqueeze(1)

f_mean = X_train_tfm.mean(dim=0, keepdims=True)
f_std = X_train_tfm.std(dim=0, keepdims=True) + eps
X_train_tfm_norm = (X_train_tfm - f_mean) / f_std
X_valid_tfm_norm = (X_valid_tfm - f_mean) / f_std

# create databunch
data = (ItemLists('.', TSList(X_train_tfm_norm),
                  TSList(X_valid_tfm_norm)).label_from_lists(
                      Y_train,
                      Y_valid).databunch(bs=min(bs, len(X_train)),
                                         val_bs=min(bs * 2, len(X_valid))))

# data.show_batch()


# a Logistic Regression with 20k input features and 2 classes in this case.
def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.)
        nn.init.constant_(layer.bias.data, 0.)


model = nn.Sequential(nn.Linear(n_kernels * 2, data.c))
model.apply(init)
learn = Learner(data, model, metrics=accuracy)

# unique, counts = np.unique(Y_train, return_counts=True)
# majority_class_count = counts.max()
# weights = majority_class_count / counts
# class_weights = torch.FloatTensor(weights).to(device)
# learn.crit = nn.CrossEntropyLoss(weight=class_weights)
learn.save('stage-0')

learn.lr_find()
learn.recorder.plot(suggestion=True)

# lr_to_use = find_appropriate_lr(learn)

learn.load('stage-0')
learn.fit_one_cycle(30, max_lr=1e-04, wd=1e2)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
