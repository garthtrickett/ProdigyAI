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

# Their data
# data = (pd.read_csv('data/dollar_bars.csv',
#                     parse_dates=['date_time'])
#         .set_index('date_time'))
# head = 5000
# if head > 0:
#     data = data.head(head)

# data.index[0]

# read parquet file of our dollar bars
import pyarrow as pa
import pyarrow.parquet as pq
data = pq.read_pandas("data/btcusdt_agg_trades_10_tick_bars.parquet").to_pandas()
print("data load finished")
head = 5000
if head > 0:
    data = data.head(head)
data = data.set_index("date_time")
data.index = pd.to_datetime(data.index)
data.index[0]

# data = data.set_index("date_time")
# data.index = pd.to_datetime(data.index)
# data.reset_index(inplace=True)




# compute bands
window = 50
data['avg'], data['upper'], data['lower'] = FinancialFunctions.bbands(data['close'],
                                                                      window, no_of_stdev=1.5)
cprint(data)

#compute frac diff
diff_amt = compute_differencing_amt(data['close'])
print('Differening amount: {:.3f}'.format(diff_amt))
fracs = frac_diff_ffd(np.log(data['close']), differencing_amt=diff_amt, threshold=1e-5)
frac_df = pd.Series(data=fracs, index=data.index)
frac_df.tail()
data['fracdiff'] = [frac_df[x] for x in data.index]

    
data.dropna(inplace=True)
cprint(data)

# Compute sides
data['side'] = np.nan


long_signals = (data['close'] <= data['lower'])#& (data['close'].shift(1) >= data['lower'].shift(1))
short_signals = (data['close'] >= data['upper'])#& (data['close'].shift(1) <= data['upper'].shift(1))

data.loc[long_signals, 'side'] = 1
data.loc[short_signals, 'side'] = -1

print(data.side.value_counts())
# Remove Look ahead biase by lagging the signal
data['side'] = data['side'].shift(1)

# Save the raw data
raw_data = data.copy()

# Drop the NaN values from our data set
data.dropna(axis=0, how='any', inplace=True)

# close prices
close = data['close']

# Before the triple barrier generate a side using NN (rather than bollinger bands)
# drop all the rows from data where our data[side]=nan

# at first we just don't filter and instead
# return every single timestamp our version of (sampled_idx) close.index

# return t1 which is just a dataframe with close.index as the index and the first column
# is a timestamp of close.index[i] + horizon into the future (end of the barrier which we set)
# 2019-05-01 01:05:37.417 (index of price at time t)   2019-05-01 02:08:40.368 (index of price at time t + horizon)

# return our events dataframe which contains
# time index of price at time t, side, time_barrier_hit, trgt, type
# trgt is the amount of profit desired given the profit threshold(static) and the volatility around that time

# return dataframe with labels
# timestamp index, ret(actual return), trgt(desired return), bin(should you act or not), side (primary model decision)
# gap between ret and trgt could determine bet size


# use honchars volatility measure as a threshold for the vertical barrier/horizontal for side 
# use the pt/sl amounts for 0,1 should you actually bet
# read the cusum/meta labelling chapter again to figure out which way prado actually intended.

# jjakimoto's triple barrier method
# Get daily volatility
from finance_ml.stats import get_vol
from finance_ml.labeling import get_t1, cusum_filter, get_events
vol = get_vol(close, seconds=3600)
# Get cusum event timestamps where price is deviating enough given the volatility
sampled_idx = cusum_filter(close, vol)
# t1 here is the vertical barrier (end of the horizon)
t1 = get_t1(close, sampled_idx, seconds=3600)
# No side/size double model so side = none
side =  None
# Return what would happen for each event from sampled_idx (take profit or not etc)
events = get_events(close, timestamps=sampled_idx, trgt=vol,
                       sltp=[1, 1], t1=t1, side=data['side'])
# events['t1'] is when any barrier is hit (timestamp index, timestamp column)
events.head()

len(events)

# dataframe with
# time t, side, time_barrier_hit, trgt, type

triple_barrier_events = events

labels = CoreFunctions.get_bins(triple_barrier_events, data['close'])
labels.side.value_counts()

# Compute sides
raw_data['side'] = np.nan


long_signals = (raw_data['close'] <= raw_data['lower']) 
short_signals = (raw_data['close'] >= raw_data['upper']) 

raw_data.loc[long_signals, 'side'] = 1
raw_data.loc[short_signals, 'side'] = -1

print(raw_data.side.value_counts())

# Remove Look ahead bias by lagging the signal
raw_data['side'] = raw_data['side'].shift(1)

#extract data at the specified events

# Get features at event dates
X = raw_data.loc[labels.index, :]

# Drop unwanted columns
X.drop([#'fracdiff', 
        'avg', 'upper', 'lower', 'open', 'high', 'low', 'close'
        ], axis=1, inplace=True)

y = labels['bin']
X.head()

#PRIMARY MODEL PERFORMANCE
primary_forecast = pd.DataFrame(labels['bin'])
primary_forecast['pred'] = 1
primary_forecast.columns = ['actual', 'pred']

# Performance Metrics
actual = primary_forecast['actual']
pred = primary_forecast['pred']
print(classification_report(y_true=actual, y_pred=pred))

print("Confusion Matrix")
print(confusion_matrix(actual, pred))

print('')
print("Accuracy")
print(accuracy_score(actual, pred))

#META MODEL FIT

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

n_estimator = 1000
depth = 2

rf = RandomForestClassifier(max_depth=depth, n_estimators=n_estimator,
                            criterion='entropy', class_weight='balanced_subsample',
                            random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

#TRAINING PERFORMANCE
# Performance Metrics
y_pred_rf = rf.predict_proba(X_train)[:, 1]
y_pred = rf.predict(X_train)
fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
print(classification_report(y_train, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_train, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_train, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# Feature Importance
title = 'Feature Importance:'
figsize = (15, 5)

feat_imp = pd.DataFrame({'Importance':rf.feature_importances_})    
feat_imp['feature'] = X.columns
feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
feat_imp = feat_imp

feat_imp.sort_values(by='Importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)
feat_imp.plot.barh(title=title, figsize=figsize)
plt.xlabel('Feature Importance Score')
plt.show()

# TEST PERFORMANCE

# Performance Metrics
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print('')
print("Accuracy")
print(accuracy_score(y_test, y_pred))

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



# Derive the performance from a 10-fold
# purged CV of an RF on (x,y) with 1% embargo
# Black Arb CEO's implementation
def crossValPlot2(skf,classifier,X,y):
    """Code adapted from:
    """
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    idx = pd.IndexSlice
    f,ax = plt.subplots(figsize=(10,7))
    i = 0
    for train, test in skf.split(X, y):
        probas_ = (classifier.fit(X.iloc[idx[train]], y.iloc[idx[train]])
                   .predict_proba(X.iloc[idx[test]]))
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[idx[test]], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    ax.legend(bbox_to_anchor=(1,1))


# Sample Weights
cpus = 4
close = raw_data.close
t1 = triple_barrier_events['t1'].loc[X.index]
idx = triple_barrier_events.loc[X.index].index

numCoEvents = snp.mpPandasObj(snp.mpNumCoEvents,('molecule',idx),
                              cpus,closeIdx=close.index,t1=t1)
numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
numCoEvents = numCoEvents.reindex(close.index).fillna(0)


sample_weights = pd.DataFrame(index=X.index)
sample_weights['w']=snp.mpPandasObj(snp.mpSampleW,('molecule',idx),cpus,
                         t1=t1, numCoEvents=numCoEvents, close=close)
sample_weights['w']*=sample_weights.shape[0]/sample_weights['w'].sum()
cprint(sample_weights)


skf = snp.PurgedKFold(n_splits=10,t1=t1,pctEmbargo=0.01)
classifier = RandomForestClassifier(n_estimators=n_estimator, max_depth=depth,
                                    criterion='entropy',
                                    class_weight='balanced_subsample',
                                    random_state=RANDOM_STATE)
# cross validation scheme, ML classifier, Features, Labels
crossValPlot2(skf,classifier,X,y)

# Prados implementation which extends the sklearn fold class for purging, embargo and sample weights
from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None)
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[
            (i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),
                                                   self.n_splits)
        ]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index)
            if maxT1Idx<X.shape[0]: # right train ( with embargo)
                train_indices=np.concatenate((train_indices, indices[maxT1Idx+mbrg:]))
            yield train_indices,test_indices

# Snippet 7.4 Using the PurgedKFold Class

def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',
            t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    idx = pd.IndexSlice
    if cvGen is None:
        cvGen=snp.PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged
    score=[]
    for train,test in cvGen.split(X=X):
        fit=clf.fit(X=X.iloc[idx[train],:],y=y.iloc[idx[train]],
                    sample_weight=sample_weight.iloc[idx[train]].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[idx[test],:])
            score_=-log_loss(y.iloc[idx[test]], prob,
                                    sample_weight=sample_weight.iloc[idx[test]].values,
                                    labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[idx[test],:])
            score_=accuracy_score(y.iloc[idx[test]],pred,
                                  sample_weight=sample_weight.iloc[idx[test]].values)
        score.append(score_)
    return np.array(score)


scores = cvScore(classifier,X,y,sample_weights['w'],t1=t1,pctEmbargo=0.01,cv=10)
scores = pd.Series(scores).sort_values()
scores