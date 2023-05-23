# Sorting out whether we are using the ipython kernel or not
import os
import sys
import numpy as np
sys.path.append("..")
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.highlevel.tasks import TSCTask
from sktime.highlevel.strategies import TSCStrategy

from sktime.transformers.compose import RowwiseTransformer
from sktime.transformers.compose import ColumnTransformer
from sktime.transformers.compose import Tabulariser
from sktime.transformers.segment import RandomIntervalSegmenter

from sktime.pipeline import Pipeline
from sktime.pipeline import FeatureUnion
from sktime.classifiers.distance_based import ProximityForest

from sktime.classifiers.compose import TimeSeriesForestClassifier
from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier

from sktime.datasets import load_gunpoint
from sktime.utils.time_series import time_series_slope

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AR

from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import numpy as np

try:
    get_ipython()
    check_if_ipython = True
except Exception:
    check_if_ipython = False
    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)


def make_sk_time_features(X, molecule):
    X = X[molecule.start:molecule.stop]
    print(molecule.start)
    print(molecule.stop)
    df = pd.DataFrame(columns=['dim_0', 'id'])
    for i in range(0, (molecule.stop - molecule.start)):
        df = df.append({
            'dim_0': pd.Series(X[i]),
            'id': molecule.start + i
        },
                       ignore_index=True)
    return df


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

# X_train, y_train = load_from_tsfile_to_dataframe(
#     "data/sktime/GunPoint/GunPoint_TRAIN.ts")
# X_test, y_test = load_from_tsfile_to_dataframe(
#     "data/sktime/GunPoint/GunPoint_TEST.ts")

# X_train.dim_0[0]
# # binary target variable
# np.unique(y_train)

# KNN with dynamic time warping
knn = KNeighborsTimeSeriesClassifier(metric='dtw')
knn.fit(X_train, y_train)
knn.score(X_test, y_test)

#Fully modular time-series forest classifier (TSF)
#We can specify the time-series tree classifier as a fully modular pipeline
# using series-to-primitive feature extraction transformers and a final decision tree classifier.
steps = [
    ('segment', RandomIntervalSegmenter(n_intervals='sqrt')),
    ('transform',
     FeatureUnion([
         ('mean',
          RowwiseTransformer(FunctionTransformer(func=np.mean,
                                                 validate=False))),
         ('std',
          RowwiseTransformer(FunctionTransformer(func=np.std,
                                                 validate=False))),
         ('slope',
          RowwiseTransformer(
              FunctionTransformer(func=time_series_slope, validate=False)))
     ])), ('clf', DecisionTreeClassifier())
]
base_estimator = Pipeline(steps, random_state=1)
# We can direclty fit and evaluate the single tree, which itself is simply a pipeline.
base_estimator.fit(X_train, y_train)
base_estimator.score(X_test, y_test)
# For time series forest, we can simply use the single tree
# as the base estimator in the forest ensemble.
tsf = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                 n_estimators=100,
                                 criterion='entropy',
                                 bootstrap=True,
                                 oob_score=True,
                                 random_state=1)

tsf.fit(X_train, y_train)
if tsf.oob_score:
    print(tsf.oob_score_)

tsf.score(X_test, y_test)

# RISE (BROKEN)
# Another popular variant of time series forest is the so-called
# Random Interval Spectral Ensemble (RISE),
# which makes use of several series-to-series feature extraction transformers, including:

# Fitted auto-regressive coefficients,
# Estimated autocorrelation coefficients,
# Power spectrum coefficients.


def ar_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    model = AR(endog=x)
    return model.fit(maxlag=nlags).params.ravel()


def acf_coefs(x, maxlag=100):
    x = np.asarray(x).ravel()
    nlags = np.minimum(len(x) - 1, maxlag)
    return acf(x, nlags=nlags).ravel()


def powerspectrum(x, **kwargs):
    x = np.asarray(x).ravel()
    fft = np.fft.fft(x)
    ps = fft.real * fft.real + fft.imag * fft.imag
    return ps[:ps.shape[0] // 2].ravel()


steps = [
    ('segment', RandomIntervalSegmenter(n_intervals=1, min_length=5)),
    ('transform',
     FeatureUnion([
         ('ar',
          RowwiseTransformer(FunctionTransformer(func=ar_coefs,
                                                 validate=False))),
         ('acf',
          RowwiseTransformer(
              FunctionTransformer(func=acf_coefs, validate=False))),
         ('ps',
          RowwiseTransformer(
              FunctionTransformer(func=powerspectrum, validate=False)))
     ])), ('tabularise', Tabulariser()), ('clf', DecisionTreeClassifier())
]
base_estimator = Pipeline(steps)

rise = TimeSeriesForestClassifier(base_estimator=base_estimator,
                                  n_estimators=50,
                                  bootstrap=True,
                                  oob_score=True)

rise.fit(X_train, y_train)
if rise.oob_score:
    print(rise.oob_score_)

rise.score(X_test, y_test)

##
# Proximity Forest (PF)
# A variant of Elastic Ensemble (EE) is Proximity Forest (PF),
# which promises to be more scalable and faster than EE by utilitising trees.

# A proximity forest consists of 3 main components:

# A proximity stump (PS) is simply a 1-nearest-neighbour classifier which uses n exemplar
# instances picked from the train set (n is usually the number of classes in the problem,
#  with one exemplar picked per class). A PS has a distance measure and accompanying
# parameters to find the proximity of each instance in the test set to the exemplars.
# The closest exemplar's class label is used as the prediction.

# A proximity tree (PT) is a classic decision tree, but uses a PS at each node
# to define the split of train instances among exemplar instances.
# A sub-PT is constructed for each exemplar instance and trained on the closest instances.
# This continues until reaching leaf status (pure by default).

# A proximity forest (PF) is an ensemble of PT, using majority voting to predict class labels.

# The pipeline of a proximity forest is as follows:

pf = ProximityForest(n_trees=1)
pf.fit(X_train, y_train)
pf.score(X_test, y_test)

# The high level create a unified interface between different but related time series methods,
# while still closely following the sklearn estimator design whenever possible. On the high level,
# two new classes are introduced:

# A task, which encapsulates the information about the learning task,
# for example the name of the target variable, and any additional
# necessary instructions on how to run fit and predict.

# A strategy which wraps the low level estimators and
# takes a task and the whole dataframe as input in fit.

train = X_train
train['class_val'] = y_train

test = X_test
test['class_val'] = y_test

task = TSCTask(target='class_val', metadata=train)

clf = TimeSeriesForestClassifier(n_estimators=50)
strategy = TSCStrategy(clf)

# Fit using task and training data
# Predict and evaluate fitted strategy on test data

strategy.fit(task, train)

y_pred = strategy.predict(test)
y_test = test[task.target]
accuracy_score(y_test, y_pred)