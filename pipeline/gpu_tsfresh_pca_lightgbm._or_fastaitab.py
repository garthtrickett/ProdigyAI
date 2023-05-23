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

print("pytorch:", torch.__version__)
print("fastai :", fastai.__version__)

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
flat_x = X.flatten()  # (28901, 50)
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

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
import matplotlib.pyplot as plt
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
    EfficientFCParameters,
)

settings = MinimalFCParameters

df_shift, _ = make_forecasting_frame(data.close,
                                     kind="price",
                                     max_timeshift=20,
                                     rolling_direction=1)

# Drop the last y as df_shift doesn't have it
y = y[:-1]
extracted_features = extract_features(
    df_shift,
    column_id="id",
    column_sort="time",
    column_value="value",
    impute_function=impute,
    show_warnings=False,
    n_jobs=6,
    default_fc_parameters=MinimalFCParameters(),
)

X = select_features(extracted_features, y)

### Do the PCA
pca_train = PCAForPandas(n_components=4)
X = pca_train.fit_transform(X)

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

### FAST AI TABULAR LEARNER
dep_var = 'y'

X_test
train = X_train
train['y'] = Y_train

X_test
test = X_test
test['y'] = Y_test
#cont_names = data.select_dtypes([np.number]).columns
cont_names = ['pca_0', 'pca_1', 'pca_2', 'pca_3']

# Transformations
procs = [
    FillMissing,
    #Normalize,
    Categorify
]

# Test Tabular List
test = TabularList.from_df(test, cont_names=cont_names, procs=procs)

# Train Data Bunch
data = (TabularList.from_df(train,
                            path='.',
                            cont_names=cont_names,
                            procs=procs).split_by_idx(list(range(
                                0, 200))).label_from_df(cols=dep_var).add_test(
                                    test, label=0).databunch())

data.show_batch(rows=10)

# Create deep learning model
learn = tabular_learner(data,
                        layers=[1000, 200, 15],
                        metrics=accuracy,
                        emb_drop=0.1,
                        callback_fns=ShowGraph)

# select the appropriate learning rate
learn.lr_find()

# we typically find the point where the slope is steepest
learn.recorder.plot(suggestion=True)

# Fit the model based on selected learning rate
learn.fit_one_cycle(15, max_lr=slice(5e-02))

# Analyse our model
learn.model
learn.recorder.plot_losses()

# # LIGHTGBM and feature importance

# # Convert training/val data into lightgbm datasets
# train_data = lgb.Dataset(X_train, label=Y_train)
# validation_data = lgb.Dataset(X_test, label=Y_test, reference=train_data)

# # lightgbm settings
# param = {
#     "max_depth": 2,
#     "eta": 1,
#     "silent": 1,
#     "objective": "multiclass",
#     "num_class": 3,
#     "tree_method": "gpu_hist",
# }

# num_round = 100
# bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])

# warnings.simplefilter(action="ignore", category=FutureWarning)
# columns = X.columns

# # the output of LGBMClassifier().booster_.feature_importance(importance_type='gain') is roughly equivalent
# # to gini importances (Mean decrease impurity) which used by RandomForestClassifier provided by Scikit-Learn
# # (Mean decrease impurity) (CANNOT be used with non tree based classifiers)
# plotImp(bst, columns, num=20)

# # SHAP VALUES (This can also be used with neural networks)
# # https://slundberg.github.io/shap/notebooks/Census%20income%20classification%20with%20LightGBM.html
# explainer = shap.TreeExplainer(bst)
# shap_values = explainer.shap_values(X)
# shap.summary_plot(shap_values, X)

# # Permutation importance (Mean decrease Accuracy) (CAN be used with non tree based classifiers)
# def score(X_test, Y_test):
#     Y_pred = bst.predict(X_test)
#     Y_pred = Y_pred.argmax(axis=-1)
#     return accuracy_score(Y_test, Y_pred)

# base_score, score_decreases = get_score_importances(score, X_test.values, Y_test)

# feature_importances = np.mean(score_decreases, axis=0)
# d = dict(zip(X.columns, feature_importances))
# print(d)
