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
unique, counts = np.unique(y, return_counts=True)
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

unique, counts = np.unique(y, return_counts=True)
run_name = preprocessed_name + "model_type=LSTM"

np.random.seed(42)
tf.random.set_seed(42)

# FEATURE IMPORTANCE
# changing from -1 ,0 ,1 labels to 0,1,2
y_boost = np.zeros(y.shape)
y_boost[:] = np.nan
for i in range(len(y)):
    if y[i] == -1:
        y_boost[i] = 0
    elif y[i] == 0:
        y_boost[i] = 1
    elif y[i] == 1:
        y_boost[i] = 2

unique, counts = np.unique(y_boost, return_counts=True)

# Splitting features and labels into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    y_boost,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    shuffle=False)

# Convert training/val data into lightgbm datasets
train_data = lgb.Dataset(X_train, label=Y_train)
validation_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)

# lightgbm settings
param = {
    "max_depth": 2,
    "eta": 1,
    "silent": 1,
    "objective": "multiclass",
    "num_class": 3,
    "tree_method": "gpu_hist",
}

num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

warnings.simplefilter(action="ignore", category=FutureWarning)
columns = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]

# the output of LGBMClassifier().booster_.feature_importance(importance_type='gain') is roughly equivalent
# to gini importances (Mean decrease impurity) which used by RandomForestClassifier provided by Scikit-Learn
# (Mean decrease impurity) (CANNOT be used with non tree based classifiers)
plotImp(bst, columns, num=20)

# SHAP VALUES (This can also be used with neural networks)
# https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20LightGBM.html
explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X)


# Permutation importance (Mean decrease Accuracy) (CAN be used with non tree based classifiers)
def score(X_test, Y_test):
    Y_pred = bst.predict(X_test)
    Y_pred = Y_pred.argmax(axis=-1)
    return accuracy_score(Y_test, Y_pred)


base_score, score_decreases = get_score_importances(score, X_test, Y_test)
feature_importances = np.mean(score_decreases, axis=0)
