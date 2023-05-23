# Mean reverting strategy based on Bollinger bands Strategy
# This notebook answers question 3.5 form the text book Advances in Financial Machine Learning.

# 3.5 Develop a mean-reverting strategy based on Bollinger bands. For each observation, the model suggests a side, but not a size of the bet.

# (a) Derive meta-labels for ptSl = [0, 2] and t1 where numDays = 1. Use as trgt the daily standard deviation as computed by Snippet 3.1.
# (b) Train a random forest to decide whether to trade or not. Use as features: volatility, seial correlation, and teh crossinmg moving averages.
# (c) What is teh accuracy of prediction from the primary model? (i.e. if the secondary model does not filter the bets) What are the precision, recall and FI-scores?
# (d) What is teh accuracy of prediction from the primary model? What are the precision, recall and FI-scores?

# Import the Hudson and Thames MlFinLab package
import mlfinlab as ml
from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from mlfinlab.features import fracdiff
import snippets as snp

import multiprocessing as mp
from multiprocessing import cpu_count
cpus = cpu_count() - 1

import blackarbsceo_bars as brs

import pymc3 as pm
import arviz


# FROM HERE: https://github.com/hudson-and-thames/research/blob/master/Chapter3/2019-03-06_JJ_Trend-Follow-Question.ipynb

import numpy as np
import pandas as pd
import pyfolio as pf
import timeit

from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline
import plotnine as pn

# # Read in their data
# data = pd.read_csv('data/sample_dollar_bars.csv')
# data.index = pd.to_datetime(data['date_time'])
# data = data.drop('date_time', axis=1)


# # # Read our in Data
data = pd.read_csv('data/dollar_bars.csv', index_col=0, parse_dates=True)

data.dropna(axis=0, how='any', inplace=True)


# Fracc DIFFing
# get the close prices
data_series = data['close'].to_frame()

# form the cumulative sum of the log prices
# log_prices = np.log(data_series).cumsum()

# Log the prices
log_prices = np.log(data_series)

d = 0.4
fd_series = fracdiff.frac_diff_ffd(log_prices, diff_amt=d, thresh=1e-5)

# Filter Events: CUSUM Filter
# Predict what will happen when a CUSUM event is triggered.
# Use the signal from the MAvg Strategy to determine the side of the bet.

# # Compute daily volatility and cusum filter from mean_bollinger_band_strategy.py
# daily_vol = ml.util.get_daily_vol(close=data['close'], lookback=50)
# cusum_events = ml.filters.cusum_filter(data['close'], threshold=daily_vol['2019-05-01':'2019-05-31'].mean() * 0.1)
# compute volatility

vol = fd_series.std()


cusum_events = filters.cusum_filter(fd_series.dropna(), threshold=vol[0]*0.00001)


# Compute vertical barrier
vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=data['close'], num_days=1)

dol_bars_feature = data['close'].loc[cusum_events]
frac_diff_feature = fd_series.loc[cusum_events]

features_mat = (pd.DataFrame()
                .assign(dollar_bars=dol_bars_feature,
                        frac_diff=frac_diff_feature,
                        )
                .drop_duplicates().dropna())

features_mat.head()

# compute daily volatility
daily_vol = utils.get_daily_vol(features_mat.dollar_bars)

# compute vertical barriers
# t1 = snp.addVerticalBarrier(tEvents, ftMtx.dbars, numDays=5)
vertical_barriers = labeling.add_vertical_barrier(t_events=cusum_events, close=features_mat.dollar_bars, num_days=5)


# Triple barrier
pt_sl = [2, 2]
min_ret = 0.0005
triple_barrier_events = labeling.get_events(close=features_mat.dollar_bars,
                                  t_events=cusum_events,
                                  pt_sl=pt_sl,
                                  target=daily_vol,
                                  min_ret=min_ret,
                                  num_threads=2,
                                  vertical_barrier_times=vertical_barriers)

events = triple_barrier_events
close = data_series['close']


# 4.1. a compute a t1 series using dollar bars derived from dataset
dbars = data
close = dbars.close.copy()
dailyVol = snp.getDailyVol(close)

# Symmetric CUSUM Filter [2.5.2.1]
tEvents = snp.getTEvents(close,h=dailyVol.mean())


t1 = snp.addVerticalBarrier(tEvents, close)

# select profit taking stoploss factor
ptsl = [1,1]
# target is dailyVol computed earlier
target=dailyVol
# select minRet
minRet = 0.005
# Getting the time of first touch
events = snp.getEvents(close,tEvents,ptsl,target,minRet,cpus,t1=t1)



# 4.1 b (b) Apply the function mpNumCoEvents to compute
# the number of overlapping outcomes at each point in time
numCoEvents = snp.mpPandasObj(snp.mpNumCoEvents,('molecule',events.index),
                              cpus,closeIdx=close.index,t1=events['t1'])
numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
numCoEvents = numCoEvents.reindex(close.index).fillna(0)
out=pd.DataFrame()


# Estimating the average uniqueness of a label [4.2]
out['tW'] = snp.mpPandasObj(snp.mpSampleTW,('molecule',events.index),
                            cpus,t1=events['t1'],numCoEvents=numCoEvents)


# Determination of sample weight by absolute return attribution [4.10]
out['w']=snp.mpPandasObj(snp.mpSampleW,('molecule',events.index),cpus, t1=events['t1'],numCoEvents=numCoEvents,close=close)
out['w']*=out.shape[0]/out['w'].sum()

fig, ax = plt.subplots(figsize=(9,6))
out.reset_index(drop=True).plot(subplots=True, alpha=0.5, ax=ax);

# 4.1 c (c) Plot the time series of number of concurrent labels on primary axis
# and time series of exponentially weighted moving standard deviation of returns on secondary axis

coEvents_std = (
    pd.DataFrame()
    .assign(
        numCoEvents = numCoEvents.reset_index(drop=True),
        std = brs.returns(data.close).ewm(50).std().reset_index(drop=True))
)
print(coEvents_std)


fig, ax = plt.subplots(figsize=(9,6))

coEvents_std.numCoEvents.plot(legend=True, ax=ax)
coEvents_std['std'].plot(secondary_y=True, legend=True, ax=ax)



# 4.1 (d) Produce a scatterplot of the number of concurrent labels (x-axis) and the exponentially weighted moving std dev of returns (y-axis).
import warnings
warnings.filterwarnings("ignore")

(pn.ggplot(coEvents_std, pn.aes('numCoEvents', 'std'))
 +pn.geom_point()
 +pn.stat_smooth())


# [4.2] Using the function mpSampleTW compute the avg uniqueness of each label.
# What is the first-order serial correlation, AR(1) of this time series? Is it
# statistically significant? Why?
def plot_traces(traces, retain=0):
    '''
    Convenience function:
    Plot traces with overlaid means and values
    '''

    ax = pm.traceplot(traces[-retain:],
                      lines=tuple([(k, {}, v['mean'])
                                   for k, v in pm.summary(traces[-retain:]).iterrows()]))

    for i, mn in enumerate(pm.summary(traces[-retain:])['mean']):
        ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0), xycoords='data'
                    ,xytext=(5,10), textcoords='offset points', rotation=90
                    ,va='bottom', fontsize='large', color='#AA0022')


lag = 1
lag_col = f'tW_lag_{lag}'
out[lag_col] = out['tW'].shift(lag)
print(out.dropna())

with pm.Model() as mdl:
    pm.GLM.from_formula(f'tW ~ {lag_col}', out.dropna())
    trace = pm.sample(3000, cores=1, nuts_kwargs={'target_accept':0.95})

plt.figure(figsize=(9, 6))
plot_traces(trace, retain=1_000)
plt.tight_layout();


df_smry = pm.summary(trace[1000:])
df_smry

# [4.3] Fit a random forest to a financial dataset where $I^{-1}\sum_{i=1}^{I}\bar u \ll 1$


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Xy = (pd.DataFrame()
      .assign(close=close,
              close_lag=close.shift(1))
     ).dropna()

y = Xy.loc[:,'close'].values
X = Xy.loc[:,'close_lag'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    shuffle=False)
RANDOM_STATE = 777
n_estimator = 50
rf = RandomForestRegressor(max_depth=1, n_estimators=n_estimator,
                           criterion='mse', oob_score=True,
                           random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# rf.oob_score_  

from sklearn.model_selection import cross_validate

n_estimator = 50
rf = RandomForestRegressor(max_depth=1, n_estimators=n_estimator,
                           criterion='mse', oob_score=True,
                           random_state=RANDOM_STATE)

scores = cross_validate(rf, X, y, cv=5, return_estimator=True)

oob_scores = [est.oob_score_ for est in scores['estimator']]
oob_scores, np.mean(oob_scores)


# Why is out-of-bag accuracy so much higher than cross-validation accuracy? 
# Which one is more correct / less biased? What is the source of this bias?

# Out of bag accuracy is higher than cross-validation b/c the incorrect 
# assumption of IID draws leads to oversampling of redudant samples.

# For random forests this means that the trees too similar. 
# The random sampling also means that in-bag and out-of-bag samples 
# will be similar inflating the oob_score_. In this example the cross-validation 
# is less-biased.

# Modify the code in Section 4.7 to apply an exponential time-decay factor

def getExTimeDecay(tW,clfLastW=1.,exponent=1):
    # apply exponential decay to observed uniqueness (tW)
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0: slope=((1.-clfLastW)/clfW.iloc[-1])**exponent
    else: slope=(1./((clfLastW+1)*clfW.iloc[-1]))**exponent
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    print(round(const,4), round(slope,4))
    return clfW




f,ax=plt.subplots(2,figsize=(10,7))
fs = [1,.75,.5,0,-.25,-.5]
ls = ['-','-.','--',':','--','-.']
for lstW, l in zip(fs,ls):
    decayFactor = getExTimeDecay(out['tW'].dropna(),
                                 clfLastW=lstW,
                                 exponent=0.75) # experiment by changing exponent
    ((out['w'].dropna()*decayFactor).reset_index(drop=True).plot(ax=ax[0],alpha=0.5))
    s = (pd.Series(1,index=out['w'].dropna().index)*decayFactor)
    s.plot(ax=ax[1], ls=l, label=str(lstW))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))


def getTimeDecay(tW,clfLastW=1.):
        # apply piecewise-linear decay to observed uniqueness (tW)
        # newest observation gets weight=1, oldest observation gets weight=clfLastW clfW=tW.sort_index().cumsum()
        clfW=tW.sort_index().cumsum()
        if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1] 
        else:slope=1./((clfLastW+1)*clfW.iloc[-1])
        const=1.-slope*clfW.iloc[-1]
        clfW=const+slope*clfW
        clfW[clfW<0]=0
        print(const,slope)
        return clfW

for lstW, l in zip(fs,ls):
    decayFactor = getTimeDecay(out['tW'].dropna(),
                                 clfLastW=lstW) # experiment by changing exponent
    ((out['w'].dropna()*decayFactor).reset_index(drop=True).plot(ax=ax[0],alpha=0.5))
    s = (pd.Series(1,index=out['w'].dropna().index)*decayFactor)
    s.plot(ax=ax[1], ls=l, label=str(lstW))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))