import numpy as np
import pandas as pd


# fast frac diff to an array
def fast_fracdiff(x, d):
    import pylab as pl

    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T])


# for prado fracdiff
def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# Prado fracdiff
def fracDiff(series, d, thres=0.01):
    """
    Increasing width window, with treatment of NaNs
    Note 1: For thres=1, nothing is skipped
    Note 2: d can be any positive fractional, not necessarily
        bounded between [0,1]
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, series.shape[0])
    # bp()
    # 2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]
    # 3) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc]
            test_val = series.loc[loc, name]  # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample("1m").mean()
            if not np.isfinite(test_val).any():
                continue  # exclude NAs
            try:
                df_.loc[loc] = np.dot(w[-(iloc + 1) :, :].T, seriesF.loc[:loc])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


# for prado frac fdd
def getWeights_FFD(d, thres):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


# for fast fdd frac diff
def get_weight_ffd(d, thres, lim):
    w, k = [1.0], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# Fast fdd fracdiff to an array
def frac_diff_ffd_array(x, d, thres=1e-5):
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width : i + 1])[0])
    return np.array(output)


# Fast fdd fracdiff to a panda
def frac_diff_ffd_panda(x, d, thres=1e-5):
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    x_no_timestamp_index = x.reset_index()
    for i in range(width, len(x)):
        output.append(
            (x_no_timestamp_index.iloc[i, 0], np.dot(w.T, x[i - width : i + 1])[0][0])
        )
    output = list(filter((0).__ne__, output))
    output = pd.DataFrame(output, columns=["timestamp", "close"])

    return output


def fracDiff_FFD(series, d, thres=1e-5):
    # Constant width window (new solution)
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method="ffill").dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            test_val = series.loc[loc1, name]  # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample("1m").mean()
            if not np.isfinite(test_val).any():
                continue  # exclude NAs
            # print(f'd: {d}, iloc1:{iloc1} shapes: w:{w.T.shape}, series: {seriesF.loc[loc0:loc1].notnull().shape}')
            try:
                df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
            except:
                continue
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df


def get_output_df(width, series, weights, name, molecule):
    series_f = series[[name]].fillna(method="ffill").dropna()
    temp_df_ = pd.Series(index=series.index)
    series = series.loc[molecule]
    series_f = series_f.loc[molecule]
    for iloc1 in range(width, series_f.shape[0]):
        loc0 = series_f.index[iloc1 - width]
        loc1 = series.index[iloc1]

        # At this point all entries are non-NAs, hence no need for the following check
        # if np.isfinite(series.loc[loc1, name]):
        temp_df_[loc1] = np.dot(weights.T, series_f.loc[loc0:loc1])[0, 0]

    return temp_df_
