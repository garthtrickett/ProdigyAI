# Sorting out whether we are using the ipython kernel or not
import os
import sys
import time
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

from sktime.transformers.shapelets import ContractedShapeletTransform

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

import pandas as pd

X_train, y_train = load_from_tsfile_to_dataframe(
    "data/sktime/GunPoint/GunPoint_TRAIN.ts")
X_test, y_test = load_from_tsfile_to_dataframe(
    "data/sktime/GunPoint/GunPoint_TEST.ts")

# The following workbook demonstrates a full workflow of using the shapelet transform
#  in sktime with a scikit-learn classifier with the GunPoint problem.

# How long (in minutes) to extract shapelets for.
# This is a simple lower-bound initially; once time is up, no further shapelets will be assessed
time_limit_in_mins = 0.1

# The initial number of shapelet candidates to assess per training series. If all series are visited
# and time remains on the contract then another pass of the data will occur
initial_num_shapelets_per_case = 10

# Whether or not to print on-going information about shapelet extraction. Useful for demo/debugging
verbose = 2

st = ContractedShapeletTransform(
    time_limit_in_mins=time_limit_in_mins,
    num_candidates_to_sample_per_case=initial_num_shapelets_per_case,
    verbose=verbose)
st.fit(X_train, y_train)

# Plotting shapelets
%matplotlib inline
import matplotlib.pyplot as plt

# for each extracted shapelet (in descending order of quality/information gain)
for s in st.shapelets[0:5]:

    # summary info about the shapelet
    print(s)
    # plot the series that the shapelet was extracted from
    plt.plot(
        X_train.iloc[s.series_id,0],
        'gray'
    )
    # overlay the shapelet onto the full series
    plt.plot(
        list(range(s.start_pos,(s.start_pos+s.length))),
        X_train.iloc[s.series_id,0][s.start_pos:s.start_pos+s.length],
        'r',
        linewidth=3.0
    )
    plt.show()


# plotting them on the same graph

# for each extracted shapelet (in descending order of quality/information gain)
for i in range(0,min(len(st.shapelets),5)):
    s = st.shapelets[i]
    # summary info about the shapelet 
    print("#"+str(i)+": "+str(s))
    
    # overlay shapelets
    plt.plot(
        list(range(s.start_pos,(s.start_pos+s.length))),
        X_train.iloc[s.series_id,0][s.start_pos:s.start_pos+s.length]
    )

plt.show()



# Full pipeline of shapelet_transform + random forest

# example pipleine with 1 minute time limit
pipeline = Pipeline([
    ('st', ContractedShapeletTransform(time_limit_in_mins=0.1, 
                                       num_candidates_to_sample_per_case=10, 
                                       verbose=False)),
    ('rf', RandomForestClassifier(n_estimators=100)),
])

start = time.time()
pipeline.fit(X_train, y_train)
end_build = time.time()
preds = pipeline.predict(X_test)
end_test = time.time()

print("Results:")
print("Correct:")
correct = sum(preds == y_test)
print("\t"+str(correct)+"/"+str(len(y_test)))
print("\t"+str(correct/len(y_test)))
print("\nTiming:")
print("\tTo build:   "+str(end_build-start)+" secs")
print("\tTo predict: "+str(end_test-end_build)+" secs")