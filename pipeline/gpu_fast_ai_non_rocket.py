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
# unique, counts = np.unique(y, return_counts=True)
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

# unique, counts = np.unique(y, return_counts=True)
run_name = preprocessed_name + "model_type=LSTM"

np.random.seed(42)
tf.random.set_seed(42)

# Splitting features and labels into train and test
X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                      y,
                                                      test_size=0.2,
                                                      random_state=1,
                                                      shuffle=False)

unique, counts = np.unique(y_train, return_counts=True)

### FAST AI PREPROCESSING AND BATCHING
# sm = SMOTE()
# sm = ADASYN()  # adds in randomness on top of smote
# y_train = np.asarray(y_train)
# unique, counts = np.unique(y_train, return_counts=True)
# try:
#     X_train, y_train = sm.fit_sample(X_train, y_train)
# except Exception as e:
#     print("Either already balanced or couldn't balance")

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_valid = X_valid.reshape(X_valid.shape[0], 1, X_valid.shape[1])

# Things to try
# - more epochs on gpu with each arch
# - standardize/normalize
# - with and without oversampling

# Minibatch Creation Settings Settings
bs = 64  #
seed = 1234  #
scale_type = 'normalize'  #'standardize', 'normalize'
scale_by_channel = False  #
scale_by_sample = False  #
scale_range = (-1, 1)  #  for normalization only: usually left to (-1, 1)

# Make the minibatch
db = (ItemLists('.', TimeSeriesList(X_train),
                TimeSeriesList(X_valid)).label_from_lists(
                    y_train, y_valid).databunch(
                        bs=min(bs, len(X_train)),
                        val_bs=min(len(X_valid), bs * 2),
                        num_workers=cpus,
                        device=device).scale(scale_type=scale_type,
                                             scale_by_channel=scale_by_channel,
                                             scale_by_sample=scale_by_sample,
                                             scale_range=scale_range))

db.show_batch()

# Build Learner

# Select one arch from these state-of-the-art time series/ 1D models:
# ResCNN, FCN, InceptionTime, ResNet
arch = InceptionTime  #
arch_kwargs = dict()  #
opt_func = Ranger  # a state-of-the-art optimizer
loss_func = LabelSmoothingCrossEntropy()  #

model = arch(db.features, db.c, **arch_kwargs).to(device)
learn = Learner(db, model, opt_func=opt_func, loss_func=loss_func)
learn.save('stage_0')
print(learn.model)
print(learn.summary())

## Train Model

# Find max learning rate
learn.load('stage_0')
learn.lr_find()
learn.recorder.plot(suggestion=True)

# Model Settings
epochs = 30
max_lr = 5e-02
warmup = False
pct_start = .7
metrics = [accuracy]
wd = 1e-2

# Train Model
learn.metrics = metrics
learn.load('stage_0')
learn.fit_one_cycle(epochs,
                    max_lr=max_lr,
                    pct_start=pct_start,
                    moms=(.95, .85) if warmup else (.95, .95),
                    div_factor=25.0 if warmup else 1.,
                    wd=wd)
learn.save('stage_1')

# Make plots
learn.recorder.plot_lr()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()

# Print info about model
print_model_outcome_info(learn, arch)

# Plot confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()