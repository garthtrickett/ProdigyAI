import sys
import os
sys.path.append("..")
import importlib
import plotnine as pn
import matplotlib.pyplot as plt
# matplotlib.use("Qt5Agg") try this is x11 fails
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

data = pq.read_pandas(
    'data/btcusdt_agg_trades_10_tick_bars.parquet').to_pandas()
# new_columns = ['Close','Timestamp','High','Low','Open']
del data['volume']
data.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close']


class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''

    def __init__(self, N, K, sliding_window=True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K  #Number of samples per row (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0] % M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1, 1)

            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(
                    -1, 1)

        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]  #Number of features per sample
        return B[:, :ci], B[:, ci:]  #Sample matrix, Target matrix


#Columns of price data to use
columns = ['Close']

time_stamps = data['Timestamp']
data = data.loc[:, columns]
original_data = data.loc[:, columns]

file_name = 'bitcoin2015to2017_close.h5'

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# normalization
for c in columns:
    data[c] = scaler.fit_transform(data[c].values.reshape(-1, 1))

#Features are input sample dimensions(channels)
A = np.array(data)[:, None, :]
original_A = np.array(original_data)[:, None, :]
time_stamps = np.array(time_stamps)[:, None, None]

#Make samples of temporal sequences of pricing data (channel)
NPS, NFS = 256, 16  #Number of past and future samples
ps = PastSampler(NPS, NFS, sliding_window=False)
B, Y = ps.transform(A)
input_times, output_times = ps.transform(time_stamps)
original_B, = ps.transform(original_A)

import pandas as pd
import numpy as numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU
from tensorflow.python.keras import utils
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf

datas = B
labels = Y

output_file_name = 'bitcoin2015to2017_close_CNN_2_relu'

step_size = datas.shape[1]
batch_size = 8
nb_features = datas.shape[2]

epochs = 10

#split training validation
training_size = int(0.8 * datas.shape[0])
training_datas = datas[:training_size, :]
training_labels = labels[:training_size, :]
validation_datas = datas[training_size:, :]
validation_labels = labels[training_size:, :]
#build model

# 2 layers
model = Sequential()

model.add(
    Conv1D(activation='relu',
           input_shape=(step_size, nb_features),
           strides=3,
           filters=8,
           kernel_size=20))
model.add(Dropout(0.5))
model.add(Conv1D(strides=4, filters=nb_features, kernel_size=16))
'''
# 3 Layers
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=8))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=8))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D( strides=2, filters=nb_features, kernel_size=8))
# 4 layers
model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
#model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Conv1D( strides=2, filters=nb_features, kernel_size=2))
'''

training_labels.shape

model.compile(loss='mse', optimizer='adam')
model.fit(training_datas,
          training_labels,
          verbose=1,
          batch_size=batch_size,
          validation_data=(validation_datas, validation_labels),
          epochs=epochs)
