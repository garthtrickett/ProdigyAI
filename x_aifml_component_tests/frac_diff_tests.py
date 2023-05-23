# import
import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt

from scipy.stats import jarque_bera
from scipy.stats import linregress
from scipy.optimize import curve_fit

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

from mlfinlab.filters import filters
from mlfinlab.labeling import labeling
from mlfinlab.util import utils
from mlfinlab.features import fracdiff

%matplotlib inline


adf = lambda s: adfuller(s, autolag='AIC')
p_val = lambda s: adfuller(s, autolag='AIC')[1]

# # Read in their data
# data = pd.read_csv('data/sample_dollar_bars.csv')
# data.index = pd.to_datetime(data['date_time'])
# data = data.drop('date_time', axis=1)

# # # Read our in Data
data = pd.read_csv('data/dollar_bars.csv', index_col=0, parse_dates=True)

# 5.5 Take the dollar bar series on E-mini S&P 500 futures.

# get the close prices
data_series = data['close'].to_frame()



# form the cumulative sum of the log prices
log_prices = np.log(data_series).cumsum()

d = 0.4
fd_series = fracdiff.frac_diff_ffd(log_prices, diff_amt=d, thresh=1e-5)

def plot_min_ffd(close_prices):
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    for d in np.linspace(0, 1, 11):
        df1 = np.log(close_prices[['close']]).resample('1D').last()  # downcast to daily obs
        df1.dropna(inplace=True)
        df2 = fracdiff.frac_diff_ffd(df1, diff_amt=d, thresh=0.01).dropna()
        corr = np.corrcoef(df1.loc[df2.index, 'close'], df2['close'])[0, 1]
        df2 = adfuller(df2['close'], maxlag=1, regression='c', autolag=None)
        out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr]  # with critical value
    out[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(out['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')
    return


plot_min_ffd(log_prices) # around 0.4 needed

# 5.5.c Compute the correlation of fracdiff series to the original (untransformed) series

comb_df = log_prices.copy().rename(columns={'close':'cumsum'})
comb_df = comb_df.join(fd_series.rename(columns={'close':'fdseries'})).dropna()
comb_df = comb_df.join(data_series.rename(columns={'close':'original'})).dropna()
comb_df.head()
comb_df.corr()


# 5.5.d Apply an Engle-Granger cointegration test on the original and the fracdiff series.

comb_df['original'].plot()
ax = comb_df['fdseries'].plot(secondary_y=True, color='r')
ax.set_ylabel('FD Series')
plt.show()

cl_prices = comb_df['original'].ravel()
fd_prices = comb_df['fdseries'].ravel()

res = coint(cl_prices, fd_prices, autolag='AIC')

print('P-value: {:.6f}'.format(res[1]))

#


# The p-value is below the critical value of 0.05, so
# we can reject the NULL that there is no cointegration.

# 5.5.e Apply a Jarque-Bera normality test on the fracdiff series

jb_test = jarque_bera(fd_prices)
print('P-value: {:.6f}'.format(jb_test[1]))


# The p-value is below 0.05 critical value, so we can reject the NULL 
# hypothesis that data has skewness and kurtosis matching normal distribution. 
# This shows that the underlying distribution of fd_prices is not Gaussian.

# 5.6 Take the the fracDiff series from 5.5
# 5.6.a Apply a CUSUM filter (Chapter 2), where h is twice the standard deviation of the series

# compute volatility
vol = fd_series.std()
print('Volatility: {:0.4f}'.format(vol[0]))

# apply cusum filter
cusum_events = filters.cusum_filter(fd_series.dropna(), threshold=vol[0]*0.000001)



# 5.6.b Use the filtered timestamps to sample a features
#  matrix. Use as one of the features the fracDiff value.

def relative_strength_index(df, n):
    """ Calculate Relative Strength Index(RSI) for given data.
    https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py

    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'high'] - df.loc[i, 'high']
        DoMove = df.loc[i, 'low'] - df.loc[i + 1, 'low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(round(PosDI * 100. / (PosDI + NegDI)), name='RSI_' + str(n))
    # df = df.join(RSI)
    return RSI


# Compute RSI
def get_rsi(data, window=14):
    df = data.copy(deep=True).reset_index()
    rsi = relative_strength_index(df, window)
    rsi_df = pd.Series(data=rsi.values, index=data.index)
    return rsi_df

rsi_df = get_rsi(data, window=14)

rsi_df.head(20)
data.head()

dol_bars_feature = data['close'].loc[cusum_events]
frac_diff_feature = fd_series.loc[cusum_events]
rsi_feature = rsi_df[cusum_events]

features_mat = (pd.DataFrame()
                .assign(dollar_bars=dol_bars_feature,
                        frac_diff=frac_diff_feature,
                        rsi=rsi_feature)
                .drop_duplicates().dropna())

features_mat.head()

# 5.6.c Form labels using the triple-barrier method, with symmetric horizontal
# barriers of twice the daily standard deviation, and a vertical barrier of 5 days

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

# create labels
labels = labeling.get_bins(triple_barrier_events, features_mat.dollar_bars)
clean_labels = labeling.drop_labels(labels)
clean_labels.bin.value_counts()

clean_labels.head()

