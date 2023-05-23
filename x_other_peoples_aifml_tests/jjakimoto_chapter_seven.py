import sys
import os

sys.path.append("..")
cwd = os.getcwd()


%load_ext autoreload
%autoreload 2

# import standard libs
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
from IPython.core.debugger import set_trace as bp
from pathlib import Path

# import python scientific stack
import pandas as pd
pd.set_option('display.max_rows', 100)
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import numba as nb
import math
import pyfolio as pf

from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from itertools import cycle
from scipy import interp

# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# %matplotlib inline
import seaborn as sns

import pandas as pd
import numpy as np
from finance_ml.stats import get_vol
from finance_ml.labeling import get_t1, cusum_filter, get_events

sns_params = {
    'font.size':9.5,
    'font.weight':'medium',
    'figure.figsize':(10,7),
}

plt.style.use('seaborn-talk')
plt.style.use('bmh')
#plt.style.use('dark_background')
sns.set_context(sns_params)
savefig_kwds=dict(dpi=300, bbox_inches='tight', frameon=True, format='png')
nanex_colors = ("#f92b20", "#fe701b", "#facd1f", "#d6fd1c", "#65fe1b",
                "#1bfe42", "#1cfdb4", "#1fb9fa", "#1e71fb", "#261cfd")
nanex_cmap = mpl.colors.ListedColormap(nanex_colors,name='nanex_cmap')
plt.register_cmap('nanex_cmap', cmap=nanex_cmap)

# import util libs
from tqdm import tqdm, tqdm_notebook
import missingno as msno

from src.utils.utils import *
import src.features.bars as brs
import src.features.snippets as snp

from mlfinlab.corefns.core_functions import CoreFunctions
from mlfinlab.corefns.financial_functions import FinancialFunctions
from mlfinlab.fracdiff.fracdiff import frac_diff_ffd, compute_differencing_amt 

import copyreg, types
copyreg.pickle(types.MethodType,snp._pickle_method,snp._unpickle_method)
RANDOM_STATE = 777

try:
    get_ipython()
    check_if_ipython = True

except Exception as e:
    check_if_ipython = False

    split_cwd = cwd.split("/")
    last_string = split_cwd.pop(-1)
    cwd = cwd.replace(last_string, "")
    os.chdir(cwd)



print(cwd)

data = (pd.read_csv('data/dollar_bars.csv',
                    parse_dates=['date_time'])
        .set_index('date_time'))
cprint(data)


close = data['close']


# Get daily volatility
vol = get_vol(close, days=1)
# Get cusum event timestamps where price is deviating enough given the volatility
sampled_idx = cusum_filter(close, vol)
# get the timestamps for the vertical barrier
t1 = get_t1(close, sampled_idx, days=1)
# No side/size double model so side = none
side =  None
# Return what would happen for each event from sampled_idx (take profit or not etc)
events = get_events(close, timestamps=sampled_idx, trgt=vol,
                       sltp=[1, 1], t1=t1, side=side)
events.head()


# Create the X and y's
index = events.index
features_df = data.drop(columns=['cum_vol', 'cum_dollar', 'cum_ticks']).dropna().loc[index]
features = features_df
features_df.dropna()

events['type'] = pd.Categorical(events['type'])

label = pd.get_dummies(events['type'].loc[features_df.index], prefix = 'category')
label = label.drop(columns=['category_t1'])


# K fold validation with shuffle=False (because shuffling causes unrealistic performance)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
scores = []
for _ in range(10):
    clf = RandomForestClassifier()
    kfold = KFold(n_splits=10, shuffle=False)
    scores.append(cross_val_score(clf, features, label, cv=kfold))
print(np.mean(scores), np.var(scores))


from sklearn.model_selection._split import _BaseKFold
import time


def get_train_times(t1, test_times):
    trn = t1.copy(deep=True)
    for i, j in test_times.iteritems():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index
        df1 = trn[(i <= trn) & (trn <= j)].index
        df2 = trn[(trn.index <= i) & (j <= trn)].index
        trn = trn.drop(df0.union(df1.union(df2)))
    return trn


class PurgedKFold(_BaseKFold):
    def __init__(self, n_splits=3, t1=None, pct_embargo=0., purging=True):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label through dates must be a pd.Series')
        super(PurgedKFold, self).__init__(n_splits=n_splits, shuffle=False,
                                          random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purging = purging

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and t1 must have the same index')
        indices = np.arange(X.shape[0])
        # Embargo width
        embg_size = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]
        for st, end in test_ranges:
            # Test data
            test_indices = indices[st:end]
            # Training data prior to test data
            t0 = self.t1.index[st]
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # Add training data after test data
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + embg_size:]))
            # Purging
            if self.purging:
                train_t1 = t1.iloc[train_indices]
                test_t1 = t1.iloc[test_indices]
                train_t1 = get_train_times(train_t1, test_t1)
                train_indices = self.t1.index.searchsorted(train_t1.index)
            yield train_indices, test_indices


from sklearn.metrics import log_loss, accuracy_score
import numpy as np

from finance_ml.sampling import get_sample_tw, get_num_co_events

def cv_score(clf, X, y, sample_weight=None, scoring='neg_log_loss',
             t1=None, n_splits=3, cv_gen=None, pct_embargo=0., purging=False):
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Wrong scoring method')
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=n_splits, t1=t1,
                             pct_embargo=pct_embargo,
                             purging=purging)
    scores = []
    for train, test in cv_gen.split(X=X):
        train_params = dict()
        test_params = dict()
        # Sample weight is an optional parameter
        if sample_weight is not None:
            train_params['sample_weight'] = sample_weight.iloc[train].values
            test_params['sample_weight'] = sample_weight.iloc[test].values
        clf_ = clf.fit(X=X.iloc[train, :], y=y.iloc[train], **train_params)
        # Scoring
        if scoring == 'neg_log_loss':
            prob = clf_.predict_proba(X.iloc[test, :])
            score_ = -log_loss(y.iloc[test], prob, labels=clf.classes_, **test_params)
        else:
            pred = clf_.predict(X.iloc[test, :])
            score_ = accuracy_score(y.iloc[test], pred, **test_params)
        scores.append(score_)
    return np.array(scores)

# Sample weights
n_co_events = get_num_co_events(close.index, t1, events.index)
sample_weight = get_sample_tw(t1, n_co_events, events.index)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
t1_ = t1.loc[features.index]

scores = []
for _ in range(100):
    scores_ = cv_score(clf, features, label,
                       pct_embargo=0., t1=t1_, purging=False)
    scores.append(np.mean(scores_))
print(np.mean(scores), np.var(scores))