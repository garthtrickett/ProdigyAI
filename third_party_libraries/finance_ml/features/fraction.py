import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, tqdm_notebook
from third_party_libraries.mlfinlab.features.fracdiff import *


def get_opt_d(
    series,
    ds=None,
    maxlag=1,
    thres=1e-5,
    max_size=10000,
    p_thres=1e-2,
    autolag=None,
    verbose=1,
    **kwargs
):
    """Find minimum value of degree of stationary differntial
    
    Params
    ------
    series: pd.Series
    ds: array-like, default np.linspace(0, 1, 100)
        Search space of degree.
    lag: int, default 1
        The lag scale when making differential like series.diff(lag)
    thres: float, default 1e-5
        Threshold to determine fixed length window
    p_threds: float, default 1e-2
    auto_lag: str, optional
    verbose: int, default 1
        If 1 or 2, show the progress bar. 2 for notebook
    kwargs: paramters for ADF
    
    Returns
    -------
    int, optimal degree
    """
    if ds is None:
        ds = np.linspace(0, 1, 100)
    # Sort to ascending order
    ds = np.array(ds)
    sort_idx = np.argsort(ds)
    ds = ds[sort_idx]
    if verbose == 2:
        iter_ds = tqdm_notebook(ds)
    elif verbose == 1:
        iter_ds = tqdm(ds)
    else:
        iter_ds = ds
    opt_d = ds[-1]
    # Compute pval for each d
    for d in iter_ds:
        diff = frac_diff_ffd(series, d, thresh=thres)
        pval = adfuller(
            diff[diff.columns[0]].dropna().values, maxlag=maxlag, autolag=autolag
        )[1]
        if pval < p_thres:
            opt_d = d
            break
    return diff, opt_d
