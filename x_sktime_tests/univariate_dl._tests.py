# -*- coding: utf-8 -*-
print("script started")

import os
import sys
sys.path.append("..")
from multiprocessing import cpu_count

import h5py
import numpy as np
import pyarrow.parquet as pq

# Scikit-Learn ≥0.20 is required


from sklearn.model_selection import train_test_split
from sktime.pipeline import Pipeline


from library.core import *

import pandas as pd
import numpy as np



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

# Getting the data in a format that sktime-dl can use
X_range_index = pd.RangeIndex(start=0, stop=len(X), step=1)
num_threads = cpus * 2
X = mp_pandas_obj(
    func=make_sk_time_features,
    pd_obj=("molecule", X_range_index),
    num_threads=num_threads,
    X=X,
)
X = X.sort_values(by=['id'])
X = X.set_index('id')

# Change from -1,0,1 to 1,2,3
y_boost = np.zeros(y.shape)
y_boost[:] = np.nan
for i in range(len(y)):
    if y[i] == -1:
        y_boost[i] = 1
    elif y[i] == 0:
        y_boost[i] = 2
    elif y[i] == 1:
        y_boost[i] = 3

# make the y's have same type as the gunpoint dataset
y = y_boost
y = y.astype(int)
y = y.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    shuffle=False)

from sktime.pipeline import Pipeline
from sktime_dl.classifiers.deeplearning import CNNClassifier

network = CNNClassifier()
steps = [('clf', network)]
clf = Pipeline(steps)

hist = clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

print("End test_pipeline()")

