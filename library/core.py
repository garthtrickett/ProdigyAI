import sys
import fnmatch
import tempfile
from multiprocessing import cpu_count

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
import sklearn
from third_party_libraries.python_binance.binance.client import Client
from third_party_libraries.python_binance.binance.enums import (
    KLINE_INTERVAL_1MINUTE,
    WEBSOCKET_DEPTH_20,
)
from third_party_libraries.python_binance.binance.websockets import BinanceSocketManager
from imblearn.pipeline import Pipeline
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sqlalchemy import create_engine
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm, tqdm_notebook
from third_party_libraries.timeseriesAI.fastai_timeseries import *
from third_party_libraries.finance_ml.model_selection import PurgedKFold
cpus = cpu_count() - 1


def filter_events(data, close_np_array, close_index_np_array,
                  volatility_threshold, filter):
    """
    Wrapper for one of the event filters. For example the cusum filter
    :param data: (pandas.DataFrame) The ohlcv data from binance API
    :param close_index_np_array: (np.array) close column from data converted into np.array
    :param close_index_np_array: (np.array) close column datetime index in ms
    :param volatility_threshold: (float) Mean of volatility over the entire period
    :param filter:  (function) The specific filter function


    """
    if filter == "cm":
        t_events = cusum_filter(close_np_array, close_index_np_array,
                                volatility_threshold)
        t_events_df = pd.DataFrame(t_events, columns=["int_index"])
        t_events_df.set_index(["int_index"], inplace=True)
        t_events_as_date_time_index = pd.to_datetime(t_events_df.index)
        return t_events_as_date_time_index
    else:
        return data.dropna().index


@njit
def combine_x_and_sample_weights(X, P, flat_X_and_P, WINDOW_LENGTH):
    """
    Numba function that that adds a sampleweight after every WINDOW_LENGTH X into a 1d array
    :param X: (np.array) Prices
    :param P: (np.array) Sample Weights
    :param flat_X_and_P: (np.array) Empty array
    :param WINDOW_LENGTH: (int) Setting up how long the feature window should be. x(T) to X(T-WINDOWLONG)
    :return: X and P combined and flattened into 1d array
    """
    j = 0
    for i in range(len(flat_X_and_P)):
        if (i - j) / (WINDOW_LENGTH * (j + 1)) != 1:
            flat_X_and_P[i] = X[i - j]
        else:
            flat_X_and_P[i] = P[j]
            j = j + 1
    return flat_X_and_P


def make_x_and_sample_weights(P, X):
    """
    Combining the sample weights and feature window so that they can be put into the SMOTE oversampling for stage 1
    :param P: (np.array) Array of sample weights
    :param X: (np.array) Array of feature windows
    :return: X_and_P (np.array) Array of shape len(X), X[i] + P
    """
    P = P.reshape(len(P))
    flat_X_and_P_length = len(X) * (X.shape[1] + 1)
    X_shape_0 = X.shape[0]
    X_shape_1 = X.shape[1]
    flat_X_and_P = np.zeros(flat_X_and_P_length)
    flat_X_and_P[:] = np.nan
    flat_X = X.reshape(X.shape[0] * X.shape[1])

    X_and_P = combine_x_and_sample_weights(flat_X, P, flat_X_and_P, X.shape[1])
    X_and_P = X_and_P.reshape(len(P), X.shape[1] + 1)
    return X_and_P


@njit
def add_P_or_sample_weights_to_X(X, P, flat_X_and_P, WINDOW_LENGTH):
    """
    Combine the prediction of stage 1 or the sample weights to the feature window.
    :param X: (np.array) Feature window.
    :param P: (np.array) Sample weights or the side predictions from stage one.
    :param flat_X_and_P: (np.array) Empty 1d array to be filled by concatenation of X and P.
    :param WINDOW_LENGTH:  (float) How long the feature window is going to be.
    :return: (np.array) Combined X and P into flat 1d array
    """
    j = 0
    for i in range(len(flat_X_and_P)):
        if (i - j) / (WINDOW_LENGTH * (j + 1)) != 1:
            flat_X_and_P[i] = X[i - j]
        else:
            flat_X_and_P[i] = P[j]
            j = j + 1
    return flat_X_and_P


# Combining the gpu_output of stage 1 with X
def make_X_sample_weights_and_P(sample_weights, X, P):
    """
    Combine X, sample weights and sidepredictions from stage 1 into one array so that we can use Smote to oversample
    :param sample_weights:
    :param X:
    :param P:
    :return:
    """

    sample_weights = sample_weights.reshape(len(sample_weights))
    flat_P = P.reshape(len(P))
    flat_X_sample_weights_length = len(X) * (X.shape[1] + 1)
    flat_X_sample_weights_and_P_length = len(X) * (X.shape[1] + 2)
    flat_X_sample_weights = np.zeros(flat_X_sample_weights_length)
    flat_X_sample_weights[:] = np.nan
    flat_X_sample_weights_and_P = np.zeros(flat_X_sample_weights_and_P_length)
    flat_X_sample_weights_and_P[:] = np.nan
    flat_X = X.reshape(X.shape[0] * X.shape[1])

    # Add the sample weight to each feature window row
    X_and_sample_weights = add_P_or_sample_weights_to_X(
        flat_X, sample_weights, flat_X_sample_weights, X.shape[1])
    # X_and_sample_weights = X_and_sample_weights.reshape(
    #     len(sample_weights), X.shape[1] + 1
    # )

    # Add the prediction of stage 1 to sample weights and the feature window
    X_sample_weights_and_P = add_P_or_sample_weights_to_X(
        X_and_sample_weights, flat_P, flat_X_sample_weights_and_P,
        X.shape[1] + 1)

    # Change from flat to 3d array
    X_sample_weights_and_P = X_sample_weights_and_P.reshape(
        len(sample_weights), X.shape[1] + 2)
    return X_sample_weights_and_P


@njit
def cusum_filter(close_np_array, close_index_np_array, volatility_threshold):
    """
    Numba version of the cusum_filter from Prados book
    :param close_np_array: (np.arary) Prices.
    :param close_index_np_array: (np.array) Datetime index in epoch for prices.
    :param volatility_threshold: (float) Mean of volatility over the period.
    :return: (np.array) Array of indexes that hit the cusum filter thresholds
    """
    sPos, sNeg = 0, 0
    t_events = np.zeros(len(close_np_array), dtype=np.int64)
    diff = np.diff(close_np_array)
    for i in range(len(diff)):
        sPos = max(0, sPos + diff[i])
        sNeg = min(0, sNeg + diff[i])
        if sNeg < -volatility_threshold:
            sNeg = 0
            t_events[i] = close_index_np_array[i + 1]
        elif sPos > volatility_threshold:
            sPos = 0
            t_events[i] = close_index_np_array[i + 1]

    return t_events[t_events != 0]


@njit
def volatility_levels_numba(close, WINDOW_LENGTH):
    """
    Get volatilty levels for each close prices window (t-WINDOWLONG, t)
    :param close: (np.array)
    :param WINDOW_LENGTH: (float)
    :return: (np.array)
    """
    res = np.zeros(len(close))
    res[:] = np.nan
    for i in range(len(close)):
        if i >= WINDOW_LENGTH:
            window = close[i - WINDOW_LENGTH:i]
            window_abs_returns = np.abs(np.diff(window) / window[1:])
            window_volatility_level = np.std(window_abs_returns) + np.mean(
                window_abs_returns)
            res[i] = window_volatility_level
    res = res.reshape(len(res), 1)
    return res


@njit(parallel=True)
def make_window_numba(prices_for_window_index_array, close_index_array,
                      close_array, WINDOW_LENGTH):
    """
    Make feature array from ohlcv close prices
    :param prices_for_window_index_array: (np.array) Datetime index for prices output by triple barrier functions.
    :param close_index_array:  (np.array)  Datetime index of all the prices.
    :param close_array: (np.array) Array of all close prices.
    :param WINDOW_LENGTH: (float) How long to make each feature window.
    :return: (np.array) Feature window array.
    """
    X = np.zeros((len(prices_for_window_index_array), WINDOW_LENGTH))
    X[:] = np.nan
    for i in prange(len(prices_for_window_index_array)):
        for j in prange(len(close_index_array)):
            if prices_for_window_index_array[i] == close_index_array[j]:
                X[i] = close_array[j + 1 - WINDOW_LENGTH:j + 1]
    return X


@njit(parallel=True)
def make_window_multivariate_numba_tabl(len_prices_for_window_index_array,
                                        input_features, WINDOW_LENGTH):
    X = np.zeros((len_prices_for_window_index_array, len(input_features),
                  WINDOW_LENGTH))
    X[:] = np.nan
    for i in prange(len_prices_for_window_index_array):
        for k in range(len(input_features)):
            X[i][k] = input_features[k][i:i + WINDOW_LENGTH]
    return X


@njit(parallel=True)
def make_window_multivariate_numba_tabl_version_two(
    len_prices_for_window_index_array, input_features, WINDOW_LENGTH):
    X = np.zeros((len_prices_for_window_index_array, len(input_features),
                  WINDOW_LENGTH))
    X[:] = np.nan
    for i in prange(len_prices_for_window_index_array):
        for k in range(len(input_features)):
            for l in prange(WINDOW_LENGTH):
                X[i][k][l] = input_features[k][i + l]
    return X


# X = np.empty((self.batch_size, *self.dim))

#         for i, ID in enumerate(list_IDs_temp):
#             for k in range(self.dim[0]):
#                 for l in range(len(self.input_features_normalized)):
#                     X[i][k][l] = self.input_features_normalized[l][ID + k]


@njit(parallel=True)
def make_window_multivariate_numba_dlob(len_prices_for_window_index_array,
                                        input_features, WINDOW_LENGTH):
    X = np.zeros((len_prices_for_window_index_array, WINDOW_LENGTH,
                  len(input_features)))
    X[:] = np.nan
    for i in prange(len_prices_for_window_index_array):
        for k in prange(WINDOW_LENGTH):
            for l in prange(len(input_features)):
                # (254651, 100, 40, 1)
                X[i][k][l] = input_features[l][i + k]
    # return X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    return X


def make_window_multivariate_numba(
    len_prices_for_window_index_array,
    input_features,
    WINDOW_LENGTH,
    model_arch,
):
    if model_arch == "TABL":
        X = make_window_multivariate_numba_tabl(
            len_prices_for_window_index_array,
            input_features,
            WINDOW_LENGTH,
        )
    elif model_arch == "DLOB":
        X = make_window_multivariate_numba_dlob(
            len_prices_for_window_index_array,
            input_features,
            WINDOW_LENGTH,
        )

    return X


def pandas_series_to_numba_ready_np_arrays(close_copy):
    """
    Take a pandas.DataFrame and turn it into a np.array
    :param close_copy: (pandas.DataFrame)
    :return: (np.array)
    """
    close_copy.index = close_copy.index.astype(np.int64)
    close_copy = close_copy.astype(np.float64)
    close_np_array = close_copy.to_numpy()
    close_index_np_array = close_copy.index.to_numpy()
    return close_np_array, close_index_np_array


@njit(parallel=True)
def apply_pt_sl_on_t1_numba(
    close_index_array,
    close_prices_array,
    events_index_array,
    events_t1_array,
    events_side_array,
    events_target_array,
    pt_sl,
):
    """
    Numba version of prados function apply_pt_sl_on_t1. Currently not used doesn't work when there is too many rows.
    :param close_index_array:
    :param close_prices_array:
    :param events_index_array:
    :param events_t1_array:
    :param events_side_array:
    :param events_target_array:
    :param pt_sl:
    :return:
    """
    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    out_index_array = np.zeros((len(events_index_array)), dtype=(np.int64))
    out_index_array[:] = np.nan

    out_sign_array = np.zeros((len(events_index_array)), dtype=(np.int64))
    out_sign_array[:] = np.nan

    out_t1_array = np.zeros((len(events_index_array)), dtype=(np.int64))
    out_t1_array[:] = np.nan

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking_targets = events_target_array * profit_taking_multiple
        profit_taking_index = events_index_array
    else:
        profit_taking_index = events_index_array

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss_targets = events_target_array * -stop_loss_multiple
        stop_loss_index = events_index_array
    else:
        stop_loss_index = events_index_array

    for i in prange(len(events_t1_array)):

        path_prices = np.zeros((len(close_prices_array)), dtype=(np.float64))
        path_prices_returns = np.zeros((len(close_prices_array)),
                                       dtype=(np.float64))
        path_prices_index = np.zeros((len(close_index_array)),
                                     dtype=(np.int64))
        path_prices[:] = np.nan
        path_prices_index[:] = np.nan
        path_prices_returns[:] = np.nan
        for j in prange(len(close_prices_array)
                        ):  # df0 = close[loc:vertical_barrier]  # path prices
            if (close_index_array[j] >= events_index_array[i]
                    and close_index_array[j] <= events_t1_array[i]):
                path_prices[i] = close_prices_array[j]
                path_prices_index[i] = close_index_array[j]

        path_prices = path_prices[path_prices != np.nan]
        path_prices_index = path_prices_index[path_prices_index != np.nan]
        path_prices_returns_index = np.zeros((len(close_index_array)),
                                             dtype=(np.int64))
        path_prices_returns_index[:] = np.nan
        for k in prange(len(path_prices_returns)):
            path_prices_returns[k] = (path_prices[k] / close_prices_array[i]
                                      ) - 1 * events_side_array[i]

            # path_prices_returns_index[k] =

        take_profit_hit = 0
        stop_loss_hit = 0
        l = 0
        # for l in range(len(path_prices_returns)):
        while (len(path_prices_returns) > l + 1 and take_profit_hit != 1
               and stop_loss_hit != 1):
            if path_prices_returns[l] > profit_taking_targets[i]:
                take_profit_hit = 1
                out_sign_array[i] = 1
                out_t1_array[i] = path_prices_index[l]
                out_index_array[i] = events_index_array[i]
            elif path_prices_returns[l] < stop_loss_targets[i]:
                stop_loss_hit = 1
                out_sign_array[i] = -1
                out_t1_array[i] = path_prices_index[l]
                out_index_array[i] = events_index_array[i]

                events_index_array[i]

            l = l + 1

    return out_sign_array, out_t1_array, out_index_array


class XInThreeDims(TransformerMixin, BaseEstimator):
    """A template for a custom transformer. This instance reshapes from 2d to 3d"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # transform X via code or additional methods
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X


class MyPipeline(Pipeline):
    """
    # Extends the sklearn pipeline class to allow the sample_weight to be added
    """
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def get_counts_from_one_hot_encoding(y):
    """
    Return counts of each labels from one hot encoded array
    :param y: (np.array) one hot encoded array of labels
    :return: (dict) Containing each labels
    """
    minus_one_count = 0
    zero_count = 0
    one_count = 0

    for row in y:
        if row[0] == 1:
            minus_one_count = minus_one_count + 1
        if row[1] == 1:
            zero_count = zero_count + 1
        if row[2] == 1:
            one_count = one_count + 1
    counts = {
        "minus_one_count": minus_one_count,
        "zero_count": zero_count,
        "one_count": one_count,
    }

    return counts


def clf_hyper_fit(feat,
                  label,
                  t1,
                  pipe_clf,
                  search_params,
                  scoring=None,
                  n_splits=3,
                  bagging=[0, None, 1.0],
                  rnd_search_iter=0,
                  n_jobs=-1,
                  pct_embargo=0.0,
                  **fit_params):
    """
    Purged and Embargoed nested cross validation. Implementation inspired from Prado's book
    :param feat:
    :param label:
    :param t1:
    :param pipe_clf:
    :param search_params:
    :param scoring:
    :param n_splits:
    :param bagging:
    :param rnd_search_iter:
    :param n_jobs:
    :param pct_embargo:
    :param fit_params:
    :return:
    """
    # Set defaut value for scoring
    print("scoring type " + scoring)
    unique, counts = np.unique(label, return_counts=True)
    if scoring is None:
        if set(unique) == {0, 1}:
            scoring = "f1"
        else:
            scoring = "neg_log_loss"

    print("scoring type " + scoring)
    # HP serach on traing data
    inner_cv = PurgedKFold(n_splits=n_splits,
                           t1=t1,
                           pct_embargo=pct_embargo,
                           num_threads=cpus * 2)
    if rnd_search_iter == 0:
        # If this code is run its a GridSearch
        search = GridSearchCV(
            estimator=pipe_clf,
            param_grid=search_params,
            scoring=scoring,
            cv=inner_cv,
            # n_jobs=n_jobs,
            iid=False,
        )
    else:
        # If this code is run its a randomized search
        search = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=search_params,
            scoring=scoring,
            cv=inner_cv,
            # n_jobs=n_jobs,
            iid=False,
        )

    best_pipe = search.fit(feat, label, **fit_params).best_estimator_
    # Fit validated model on the entirely of dawta
    if bagging[0] > 0:
        bag_est = BaggingClassifier(
            base_estimator=MyPipeline(best_pipe.steps),
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]),
            # n_jobs=n_jobs,
        )
        bag_est = best_pipe.fit(
            feat,
            label,
            sample_weight=fit_params[bag_est.base_estimator.steps[-1][0] +
                                     "__sample_weight"],
        )
        best_pipe = Pipeline([("bag", bag_est)])
    return best_pipe


def make_x_from_set(train_set, X):
    """
    Make features from train_set/test_set CV indicies
    :param train_set:
    :param X:
    :return:
    """
    X_from_set = np.zeros((X.shape[0], X.shape[1]), dtype=np.float64)
    X_from_set[:] = np.nan
    for i in range(len(X)):
        if i in train_set:
            X_from_set[i] = X[i]
    return X_from_set


@njit
def binarize_y_side(y):
    """
    One hot encode stage 1 side predictions
    :param y:
    :return:
    """
    one_hot_encoded_y = np.zeros((len(y), 3), dtype=np.int64)
    for i in range(len(y)):
        if y[i] == -1:
            one_hot_encoded_y[i] = [1, 0, 0]
        elif y[i] == 0:
            one_hot_encoded_y[i] = [0, 1, 0]
        elif y[i] == 1:
            one_hot_encoded_y[i] = [0, 0, 1]
    return one_hot_encoded_y


@njit
def binarize_y_size(y):
    """
    One hot encode stage 2 size predictions
    :param y:
    :return:
    """
    one_hot_encoded_y = np.zeros((len(y), 2), dtype=np.int64)
    for i in range(len(y)):
        if y[i] == 0:
            one_hot_encoded_y[i] = [0, 1]
        elif y[i] == 1:
            one_hot_encoded_y[i] = [0, 1]
    return one_hot_encoded_y


def frac_diff_ffd(x, d, thres=1e-5):
    w = get_weight_ffd(d, thres, len(x))
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    return np.array(output)


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


def get_opt_d(series,
              ds=None,
              maxlag=1,
              thres=1e-5,
              max_size=10000,
              p_thres=1e-2,
              autolag=None,
              verbose=1,
              **kwargs):
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
        import time

        start = time.time()
        diff = frac_diff_ffd(series, d, thres=thres)
        end = time.time()
        # print("frac_time" + str(end - start))
        start = time.time()
        pval = adfuller(diff, maxlag=maxlag, autolag=autolag)[1]
        end = time.time()
        # print("fuller_time" + str(end - start))
        if pval < p_thres:
            opt_d = d
            break
    return diff, opt_d


class LoglossObjective(object):
    def calc_ders_range(self, approxes, targets, weights):
        # approxes, targets, weights are indexed containers of floats
        # (containers which have only __len__ and __getitem__ defined).
        # weights parameter can be None.
        #
        # To understand what these parameters mean, assume that there is
        # a subset of your dataset that is currently being processed.
        # approxes contains current predictions for this subset,
        # targets contains target values you provided with the dataset.
        #
        # This function should return a list of pairs (der1, der2), where
        # der1 is the first derivative of the loss function with respect
        # to the predicted value, and der2 is the second derivative.
        #
        # In our case, logloss is defined by the following formula:
        # target * log(sigmoid(approx)) + (1 - target) * (1 - sigmoid(approx))
        # where sigmoid(x) = 1 / (1 + e^(-x)).

        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            e = np.exp(approxes[index])
            p = e / (1 + e)
            der1 = (1 - p) if targets[index] > 0.0 else -p
            der2 = -p * (1 - p)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result


def plotImp(model, columns, num=20):
    """Plot feature importance (Mean decrease impurity) """
    feature_imp = pd.DataFrame(
        sorted(zip(model.feature_importance(importance_type="gain"), columns)),
        columns=["Value", "Feature"],
    )
    plt.figure(figsize=(40, 20))
    sns.set(font_scale=5)
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.show()
    # plt.savefig("lgbm_importances-01.png")


def makeOverSamplesADASYN(X, y):
    # ADASYN balances classes by adding synthetic data
    from imblearn.over_sampling import ADASYN

    sm = ADASYN()
    X, y = sm.fit_sample(X, y)
    return (X, y)


def connect_to_database(DB_HOST, DB_PASSWORD, DB_USER, DB_NAME):
    # Connect to the database with sql alchemy and psycopg2

    engine_string = ("postgres://" + DB_USER + ":" + DB_PASSWORD + "@" +
                     DB_HOST + "/" + DB_NAME)

    engine = create_engine(engine_string)

    psycopg2_connect_string = ("dbname=" + DB_NAME + " user=" + DB_USER +
                               " host=" + DB_HOST + " password=" + DB_PASSWORD)

    conn = psycopg2.connect(psycopg2_connect_string)
    cur = conn.cursor()
    return cur, engine, conn


def read_sql_tmpfile(query, engine):
    """
    Used in the postgres database connection
    :param query:
    :param engine:
    :return:
    """
    with tempfile.TemporaryFile() as tmpfile:
        copy_sql = "COPY ({query}) TO STDOUT WITH CSV {head}".format(
            query=query, head="HEADER")
        conn = engine.raw_connection()
        cur = conn.cursor()
        cur.copy_expert(copy_sql, tmpfile)
        tmpfile.seek(0)
        df = pd.read_csv(tmpfile)

        return df


def get_symbol_list(exchange_info, selected_symbols=None, market=None):
    """
    Get list of symbols
    :param exchange_info: (object) From python binance api
    :param selected_symbols: (list) Optionally select symbols
    :return:
    """
    symbols_from_binance = exchange_info["symbols"]

    # Making a new list of symbols from the binance exchange information
    symbol_list = []
    for symbol in symbols_from_binance:
        if market == "futures":
            symbol_list.append(symbol["symbol"])
        elif symbol["isMarginTradingAllowed"] is True:
            symbol_list.append(symbol["symbol"])

    if selected_symbols is not None:
        for symbol in selected_symbols:
            if symbol in symbol_list:
                print(symbol)
            else:
                selected_symbols.remove(symbol)
    else:
        selected_symbols = symbol_list

    return selected_symbols


def get_wild_card_variables(wildcard):
    """Get all local variables that contain wildcard"""
    variables = sys._getframe(1).f_locals
    rets = []
    for var in variables.keys():
        if fnmatch.fnmatch(var, wildcard):
            rets.append(var)

    return rets


def load_client():
    """
    Load the Binance api client object
    :return:
    """
    # Got the api key and api secret to authenticate to the binance api
    from support_files.locations import (
        binance_apiKey as binKey,
        binance_apiSecret as binSecret,
        SaveDir,
    )

    # connecting to binance API
    client = Client(binKey, binSecret)

    return client


class ROCKET(nn.Module):
    def __init__(self, c_in, seq_len, n_kernels=10000, kss=[7, 9, 11]):
        """
        ROCKET is a GPU Pytorch implementation of the ROCKET methods generate_kernels 
        and apply_kernels that can be used  with univariate and multivariate time series.
        Input: is a 3d torch tensor of type torch.float32. When used with univariate TS, 
        make sure you transform the 2d to 3d by adding unsqueeze(1).
        c_in: number of channels or features. For univariate c_in is 1.
        seq_len: sequence length
        """
        super().__init__()
        kss = [ks for ks in kss if ks < seq_len]
        convs = nn.ModuleList()
        for i in range(n_kernels):
            ks = np.random.choice(kss)
            dilation = 2**np.random.uniform(0,
                                            np.log2((seq_len - 1) // (ks - 1)))
            padding = int(
                (ks - 1) * dilation // 2) if np.random.randint(2) == 1 else 0
            weight = torch.randn(1, c_in, ks)
            weight -= weight.mean()
            bias = 2 * (torch.rand(1) - 0.5)
            layer = nn.Conv1d(c_in,
                              1,
                              ks,
                              padding=2 * padding,
                              dilation=int(dilation),
                              bias=True)
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer.bias = torch.nn.Parameter(bias, requires_grad=False)
            convs.append(layer)
        self.convs = convs
        self.n_kernels = n_kernels
        self.kss = kss

    def forward(self, x):
        for i in range(self.n_kernels):
            out = self.convs[i](x)
            _max = out.max(dim=-1).values
            _ppv = torch.gt(out, 0).sum(dim=-1).float() / out.shape[-1]
            cat = torch.cat((_max, _ppv), dim=-1)
            output = cat if i == 0 else torch.cat((output, cat), dim=-1)
        return output


# def find_appropriate_lr(
#         model: Learner,
#         lr_diff: int = 15,
#         loss_threshold: float = 0.05,
#         adjust_value: float = 1,
#         plot: bool = False,
# ) -> float:
#     # Run the Learning Rate Finder
#     model.lr_find()
#     # Get loss values and their corresponding gradients, and get lr values
#     losses = np.array(model.recorder.losses)
#     assert lr_diff < len(losses)
#     loss_grad = np.gradient(losses)
#     lrs = model.recorder.lrs
#     # Search for index in gradients where loss is lowest before the loss spike
#     # Initialize right and left idx using the lr_diff as a spacing unit
#     # Set the local min lr as -1 to signify if threshold is too low
#     r_idx = -1
#     l_idx = r_idx - lr_diff
#     while (l_idx >= -len(losses)) and (abs(loss_grad[r_idx] - loss_grad[l_idx])
#                                        > loss_threshold):
#         local_min_lr = lrs[l_idx]
#         r_idx -= 1
#         l_idx -= 1

#     lr_to_use = local_min_lr * adjust_value
#     if plot:
#         # plots the gradients of the losses in respect to the learning rate change
#         plt.plot(loss_grad)
#         plt.plot(
#             len(losses) + l_idx,
#             loss_grad[l_idx],
#             markersize=10,
#             marker="o",
#             color="red",
#         )
#         plt.ylabel("Loss")
#         plt.xlabel("Index of LRs")
#         plt.show()

#         plt.plot(np.log10(lrs), losses)
#         plt.ylabel("Loss")
#         plt.xlabel("Log 10 Transform of Learning Rate")
#         loss_coord = np.interp(np.log10(lr_to_use), np.log10(lrs), losses)
#         plt.plot(np.log10(lr_to_use),
#                  loss_coord,
#                  markersize=10,
#                  marker="o",
#                  color="red")
#         plt.show()
#     return lr_to_use


def print_model_outcome_info(learn, arch):
    archs_names, acc_, acces_, acc5_, n_params_, = [], [], [], [], []
    archs_names.append(arch.__name__)
    early_stop = math.ceil(
        np.argmin(learn.recorder.losses) / len(learn.data.train_dl))
    acc_.append("{:.5}".format(learn.recorder.metrics[-1][0].item()))
    acces_.append("{:.5}".format(learn.recorder.metrics[early_stop -
                                                        1][0].item()))
    acc5_.append("{:.5}".format(np.mean(np.max(learn.recorder.metrics))))
    n_params_.append(count_params(learn))
    clear_output()
    df = (pd.DataFrame(
        np.stack((archs_names, acc_, acces_, acc5_, n_params_)).T,
        columns=[
            "arch",
            "accuracy",
            "accuracy train loss",
            "max_accuracy",
            "n_params",
        ],
    ).sort_values("accuracy train loss",
                  ascending=False).reset_index(drop=True))
    display(df)


def minmax_scale_as_1d_array(X, min_stat=None, max_stat=None):
    dim_one = X.shape[0]
    dim_two = X.shape[1]
    X = X.flatten()
    if min_stat == None and max_stat == None:
        min_stat = np.amin(X)
        max_stat = np.amax(X)

    X = (X - min_stat) / (max_stat - min_stat)
    X = X.reshape((dim_one, dim_two))
    return X, min_stat, max_stat


def convert_y_labels_to_zero_index(y):
    y_new = np.zeros(y.shape)
    y_new[:] = np.nan
    for i in range(len(y)):
        if y[i] == -1:
            y_new[i] = 0
        elif y[i] == 0:
            y_new[i] = 1
        elif y[i] == 1:
            y_new[i] = 2
    return y_new


def plot_window_and_touch_and_label(window_index, window_length, data,
                                    triple_barrier_events, labels):
    window_start_time = triple_barrier_events.iloc[window_index -
                                                   window_length].name
    index_time = triple_barrier_events.iloc[window_index].name
    touch_time = triple_barrier_events.iloc[window_index].t1

    index_time_price = data.loc[
        triple_barrier_events.iloc[window_index].name].close
    touch_time_price = data.loc[
        triple_barrier_events.iloc[window_index].t1].close

    touch_time_integer_index = data.index.get_loc(touch_time)

    triple_barrier_only_data = data.close[triple_barrier_events.index]

    plt.plot(
        (triple_barrier_events)[window_index - window_length + 1:window_index +
                                1].index.insert(window_length, touch_time),
        np.append(
            triple_barrier_only_data[window_index - window_length +
                                     1:window_index + 1].values,
            touch_time_price,
        ),
        "o",
        color="black",
    )

    print(labels.iloc[window_index])
    print("window start time: " + str(window_start_time))
    print("index time: " + str(index_time))
    print("index_time_price: " + str(index_time_price))
    print("touch time " + str(touch_time))
    print("touch_time_price: " + str(touch_time_price))


def make_one_hot_encoded_percentage_change_features(data, y, num_percentiles):
    X_percentage_changes = data.dropna().close.pct_change().dropna()
    # Should be even
    X_percentage_changes_bins, q, c = make_bins_from_percentage_changes(
        X_percentage_changes, num_percentiles)
    X_percentage_changes_bins = pd.get_dummies(
        X_percentage_changes_bins).values.tolist()

    X_percentage_changes_bins = np.asarray(X_percentage_changes_bins)

    X = X_percentage_changes_bins
    y = y[1:]
    return X, y


def make_bins_from_percentage_changes(X_percentage_changes, num_percentiles):
    q = []
    for i in range(num_percentiles):
        q.append(i / num_percentiles)
        if i + 1 == num_percentiles:
            q.append(1)
    c = list(range(len(q) - 1))

    X_percentage_changes_bins = pd.qcut(X_percentage_changes, q=q, labels=c)
    return X_percentage_changes_bins, q, c


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it 
    compatible with pandas DataFrames.
    """
    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = [
            "pca_{}".format(i) for i in range(len(pandas_df.columns))
        ]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the 
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError(
                    "The columns of the new X is not compatible with the columns from the previous X data"
                )
        else:
            self._X_columns = list(X.columns)

        return X


def make_sk_time_features(X, molecule):
    X = X[molecule.start:molecule.stop]
    print(molecule.start)
    print(molecule.stop)
    df = pd.DataFrame(columns=["dim_0", "id"])
    for i in range(0, (molecule.stop - molecule.start)):
        df = df.append({
            "dim_0": pd.Series(X[i]),
            "id": molecule.start + i
        },
                       ignore_index=True)
    return df


@njit
def process_new_bid_or_ask(existing_bids_or_asks, bids_or_asks_prices,
                           bids_or_asks_volumes, book_type):

    outcome_list = np.zeros(len(bids_or_asks_prices))
    outcome_list[:] = -1
    appended = False
    outcome_list_count = 0

    for j in range(len(bids_or_asks_prices)):  # For all existing bids
        isin = 0
        for q in range(len(existing_bids_or_asks)):
            if bids_or_asks_prices[j] == existing_bids_or_asks[q][0]:
                isin = 1
                zero_index = q

        if isin == 1:
            existing_bids_or_asks[zero_index][1] = bids_or_asks_volumes[j]
        elif isin == 0:
            for n in range(len(bids_or_asks_prices)):
                if bids_or_asks_prices[n] == bids_or_asks_prices[j]:
                    k = n

            outcome_list[outcome_list_count] = k
            outcome_list_count = outcome_list_count + 1
            # import pdb
            # pdb.set_trace()

    for l in range(len(
            bids_or_asks_prices)):  # case3: New price doesn't exist so add it
        isin = 0
        for p in range(len(outcome_list)):
            if outcome_list[p] == l:
                isin = 1

        if isin == 1:
            existing_bids_or_asks_flat = np.concatenate(
                (existing_bids_or_asks.ravel(),
                 np.asarray([bids_or_asks_prices[l]])))
            existing_bids_or_asks_flat = np.concatenate(
                (existing_bids_or_asks_flat,
                 np.asarray([bids_or_asks_volumes[l]])))
            existing_bids_or_asks = None

            existing_bids_or_asks = existing_bids_or_asks_flat.reshape(
                (int(len(existing_bids_or_asks_flat) / 2), 2))

    existing_bids_or_asks_new = np.zeros((existing_bids_or_asks.shape[0], 2))
    existing_bids_or_asks_new[:] = np.nan
    if existing_bids_or_asks.ndim == 1:
        existing_bids_or_asks = existing_bids_or_asks.reshape(
            (int(len(existing_bids_or_asks) / 2), 2))

    for m in range(len(existing_bids_or_asks)):
        if float(existing_bids_or_asks[m][1]) > 0.0:
            appended = True
            existing_bids_or_asks_new[m] = existing_bids_or_asks[m]

    return existing_bids_or_asks_new


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


import csv
from io import StringIO

from sqlalchemy import create_engine


def psql_insert_copy(table, conn, keys, data_iter):
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ", ".join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = "{}.{}".format(table.schema, table.name)
        else:
            table_name = table.name

        sql = "COPY {} ({}) FROM STDIN WITH CSV".format(table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)


@njit
def get_nans_index_from_2d_array_numba(array):
    nan_index_array = np.zeros(len(array))
    nan_count = 0
    for i in prange(len(array)):
        if np.isnan(np.min(array[i])) == True:
            nan_count = nan_count + 1
    nan_index_array = np.zeros(nan_count)
    nan_indexer = 0
    for i in prange(len(array)):
        if np.isnan(np.min(array[i])) == True:
            nan_index_array[nan_indexer] = i
            nan_indexer = nan_indexer + 1
    if nan_count > 0:
        return nan_index_array
    else:
        return None


@njit
def get_multiple_nans_index_from_2d_array_numba(array):
    nan_array = np.zeros(len(array))
    nan_array[:] = np.nan
    for i in prange(len(array)):
        if np.isnan(np.min(array[i])) == False:
            nan_array[i] = 0
    return nan_array


@njit
def process_new_bid_or_ask_manager(existing_bids, prices_bids, volumes_bids,
                                   book_type):
    prices_bids = np.asarray(prices_bids)
    volumes_bids = np.asarray(volumes_bids)
    old_existing_bids = existing_bids

    existing_bids = process_new_bid_or_ask(existing_bids, prices_bids,
                                           volumes_bids, book_type)
    nan_index = None
    nan_index = get_nans_index_from_2d_array_numba(existing_bids)

    return existing_bids, nan_index


@njit
def trim_existing_bids_or_asks(existing_bids, depth_level):
    if len(existing_bids) < depth_level:
        while len(existing_bids) < depth_level:
            existing_bids = np.append(existing_bids,
                                      np.asarray([[np.nan, np.nan]]),
                                      axis=0)
    return existing_bids


def make_filtered_row_list(rows):
    rows_filtered_for_change_in_mid_price = np.zeros((len(rows), len(rows[0])))
    rows_filtered_for_change_in_mid_price[:] = np.nan
    row_count = 0
    for row in rows:
        if row_count > 0:
            if old_row != row[5]:
                rows_filtered_for_change_in_mid_price[row_count] = row
        old_row = row[5]
        row_count = row_count + 1

    return rows_filtered_for_change_in_mid_price


@njit
def filter_bids_or_asks(bids, not_nan_index_np_array):
    filtered_bids = np.zeros(
        (len(not_nan_index_np_array), bids.shape[1], bids.shape[2]))
    filtered_bids[:] = np.nan
    bid_add_count = 0
    for i in range(len(bids)):
        if bid_add_count >= len(not_nan_index_np_array):
            break
        if i == not_nan_index_np_array[bid_add_count]:
            filtered_bids[bid_add_count] = bids[i]
            bid_add_count = bid_add_count + 1
    return filtered_bids

    not_nan_index_list = []
    for i in range(len(nan_array)):
        if nan_array[i] == 0:
            nan_count = nan_count + 1
            not_nan_index_list.append(i)

    not_nan_index_np_array = np.asarray(not_nan_index_list)

    return not_nan_index_np_array, nan_count


def make_rows_filtered_for_change_in_mid_price_np_array(
    nan_count, rows, not_nan_index_np_array):
    rows_filtered_for_change_in_mid_price_np_array = np.zeros(
        (nan_count, len(rows[0])), dtype=np.float64)

    not_nan_index_np_array_count = 0
    rows_filtered_for_change_in_mid_price_np_array[:] = np.nan
    for i in range(len(rows)):
        if not_nan_index_np_array_count >= len(not_nan_index_np_array):
            break
        if i == not_nan_index_np_array[not_nan_index_np_array_count]:
            rows_filtered_for_change_in_mid_price_np_array[
                not_nan_index_np_array_count] = rows[i]
            not_nan_index_np_array_count = not_nan_index_np_array_count + 1
    return rows_filtered_for_change_in_mid_price_np_array


def make_nan_index_array_from_nan_array(nan_array):
    nan_count = 0
    not_nan_index_list = []
    for i in range(len(nan_array)):
        if nan_array[i] == 0:
            nan_count = nan_count + 1
            not_nan_index_list.append(i)

    not_nan_index_np_array = np.asarray(not_nan_index_list)

    return not_nan_index_np_array, nan_count


# @njit
def combine_trades_and_orderbook(rows_trade, rows_orderbook, bids, asks,
                                 level):
    combined_trades_and_orderbook_array = np.zeros(
        (len(rows_trade), 5 + level * 4))
    combined_trades_and_orderbook_array[:] = np.nan
    j_index = 0
    for i in range(len(rows_trade)):
        row_found = 0
        if rows_trade[i][1] >= rows_orderbook[0][1]:

            for j in range(j_index, len(rows_orderbook)):
                if rows_orderbook[j][1] > rows_trade[i][1]:
                    # print(rows_trade[i][2])
                    # print(rows_orderbook[j][1])
                    # print(rows_orderbook[j-1][1])
                    index_offset = 1
                    while row_found == 0:
                        if (rows_orderbook[j - index_offset][1] <=
                                rows_trade[i][1] and row_found != 1):
                            row_found = 1
                            # print("")
                            # print(rows_orderbook[j - index_offset][1])
                            # print(rows_trade[i][1])
                            # print("")
                            j_index = j
                            combined_trades_and_orderbook_array[i][
                                0] = rows_trade[i][1]
                            combined_trades_and_orderbook_array[i][
                                1] = rows_orderbook[j - index_offset][1]
                            combined_trades_and_orderbook_array[i][
                                2] = rows_orderbook[j - index_offset][5]
                            combined_trades_and_orderbook_array[i][
                                3] = rows_trade[i][3]
                            combined_trades_and_orderbook_array[i][
                                4] = rows_trade[i][4]
                            combined_trades_and_orderbook_array[i][
                                5:level + 5] = bids[j - index_offset][0:10][:,
                                                                            0]
                            combined_trades_and_orderbook_array[i][level + 5:(
                                level * 2) + 5] = bids[j -
                                                       index_offset][0:10][:,
                                                                           1]
                            combined_trades_and_orderbook_array[i][
                                (level * 2) + 5:(level * 3) +
                                5] = asks[j - index_offset][0:10][:, 0]
                            combined_trades_and_orderbook_array[i][
                                (level * 3) + 5:(level * 4) +
                                5] = asks[j - index_offset][0:10][:, 1]
                            row_found = 1
                            index_offset = index_offset + 1
                            break
                        if row_found == 1:
                            break
                    if row_found == 1:
                        break
    return combined_trades_and_orderbook_array


def add_nanosecond_to_duplicate_df_indexes(df):

    ### code to duplicate indexes
    # get duplicated values as float and replace 0 with NaN
    values = df.index.duplicated(keep="first").astype(float)
    values[values == 0] = np.NaN

    missings = np.isnan(values)
    cumsum = np.cumsum(~missings)
    diff = np.diff(np.concatenate(([0.0], cumsum[missings])))
    values[missings] = -diff
    result = df.index + np.cumsum(values).astype(np.timedelta64)
    df.index = result

    return df


def get_means_by_position_in_feature_tensor(X):
    means = np.zeros((X.shape[1], X.shape[2]))
    for i in range(len(X)):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                means[j][k] = means[j][k] + X[i][j][k]
    means = means / (len(X))
    return means


def get_std_by_position_in_feature_tensor(X, means):
    std = np.zeros((X.shape[1], X.shape[2]))
    for i in range(len(X)):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                std[j][k] = std[j][k] + ((X[i][j][k] - means[j][k])**2)
    std = np.sqrt((std / len(X)))
    return std


def apply_z_score_by_position_in_feature_tensor(input_np_array, means, std):
    for i in range(len(input_np_array)):
        for j in range(input_np_array.shape[1]):
            for k in range(input_np_array.shape[2]):
                input_np_array[i][j][k] = (input_np_array[i][j][k] -
                                           means[j][k]) / std[j][k]
    return input_np_array


@njit
def get_max_by_position_in_feature_tensor(X):
    maxes = np.zeros((X.shape[1], X.shape[2]))
    mins = np.zeros((X.shape[1], X.shape[2]))
    for i in range(len(X)):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                if i == 0:
                    maxes[j][k] = X[i][j][k]
                    mins[j][k] = X[i][j][k]
                if maxes[j][k] < X[i][j][k]:
                    maxes[j][k] = X[i][j][k]
                elif mins[j][k] > X[i][j][k]:
                    mins[j][k] = X[i][j][k]
    return maxes, mins


@njit
def apply_min_max_by_position_in_feature_tensor(input_np_array, maxes, mins):
    for i in range(len(input_np_array)):
        for j in range(input_np_array.shape[1]):
            for k in range(input_np_array.shape[2]):
                input_np_array[i][j][k] = (input_np_array[i][j][k] - mins[j][k]
                                           ) / (maxes[j][k] - mins[j][k])
    return input_np_array


def make_input_features_from_orderbook_data(volumes_for_all_labels):
    volumes_np_array = np.asarray(volumes_for_all_labels)
    input_features = np.zeros(
        (volumes_np_array.shape[1], len(volumes_for_all_labels)))
    for i in range(volumes_np_array.shape[1]):
        input_features[i] = volumes_np_array[:, [i]].reshape(
            len(volumes_for_all_labels))

    return input_features


@njit
def get_fit_scalars(scaling_type, input_features):
    if scaling_type == 0:  # Min Max
        maxes_np_array, mins_np_array = get_maxes_and_mins_from_input_features(
            input_features)
        return maxes_np_array, mins_np_array
    elif scaling_type == 1:  # Z_score
        mean_np_array, std_np_array = get_means_and_stds_from_input_features(
            input_features)

        return mean_np_array, std_np_array


@njit
def scale_input_features(
    scaling_type,
    maxes_or_means_np_array,
    mins_or_stds_np_array,
    input_features,
    minimum=None,
    maximum=None,
):
    if scaling_type == 0:  # Min_max
        input_features_normalized_scaffold = min_max_norm_input_features(
            maxes_or_means_np_array,
            mins_or_stds_np_array,
            input_features,
            minimum,
            maximum,
        )
    elif scaling_type == 1:  # Z_score
        input_features_normalized_scaffold = z_score_norm_input_features(
            maxes_or_means_np_array, mins_or_stds_np_array, input_features)
    return input_features_normalized_scaffold


@njit
def get_means_and_stds_from_input_features(input_features):
    means_np_array = np.zeros(input_features.shape[0])
    stds_np_array = np.zeros(input_features.shape[0])
    for i in range(len(input_features)):
        means_np_array[i] = np.mean(input_features[i])
        stds_np_array[i] = np.std(input_features[i])

    return means_np_array, stds_np_array


@njit
def get_maxes_and_mins_from_input_features(input_features):
    maxes_np_array = np.zeros(input_features.shape[0])
    mins_np_array = np.zeros(input_features.shape[0])
    for i in range(len(input_features)):
        maxes_np_array[i] = np.max(input_features[i])
        mins_np_array[i] = np.min(input_features[i])

    return maxes_np_array, mins_np_array


@njit(parallel=True)
def min_max_norm_input_features(maxes_np_array, mins_np_array, input_features,
                                minimum, maximum):
    input_features_normalized_scaffold = np.zeros(
        (len(input_features), input_features[0].shape[0]))
    input_features_normalized_scaffold[:] = np.nan

    if minimum == -1 and maximum == 1:
        multiplier_one = 2
        multiplier_two = 1
    elif minimum == 0 and maximum == 1:
        multiplier_one = 1
        multiplier_two = 0

    for i in prange(len(input_features)):
        for j in prange(len(input_features[0])):
            normed_sample = (
                multiplier_one * (input_features[i][j] - mins_np_array[i]) /
                (maxes_np_array[i] - mins_np_array[i])) - multiplier_two
            input_features_normalized_scaffold[i][j] = normed_sample

    return input_features_normalized_scaffold


@njit(parallel=True)
def z_score_norm_input_features(means_np_array, stds_np_array, input_features):
    input_features_normalized_scaffold = np.zeros(
        (len(input_features), input_features[0].shape[0]))
    input_features_normalized_scaffold[:] = np.nan

    for i in prange(len(input_features)):
        for j in prange(len(input_features[0])):
            normed_sample = (input_features[i][j] -
                             means_np_array[i]) / stds_np_array[i]
            input_features_normalized_scaffold[i][j] = normed_sample

    return input_features_normalized_scaffold


@njit
def average_bids_or_asks(bids, asks, index_np_array, orderbook_depth,
                         previous_index, start_index, end_index):
    bids = bids[start_index:end_index, :orderbook_depth]
    asks = asks[start_index:end_index, :orderbook_depth]

    averaged_bids = np.zeros(bids.shape)
    averaged_bids[:] = np.nan
    averaged_asks = np.zeros(asks.shape)
    averaged_asks[:] = np.nan

    additional_duplicates = -1
    for i in range(start_index, end_index):
        if index_np_array[i] == previous_index:
            additional_duplicates = additional_duplicates + 1
        else:
            break
    printcounter = 0
    print_count_multiplier = 0
    for i in range(start_index, end_index):
        if printcounter == 100000:
            print_count_multiplier = print_count_multiplier + 1
            print(printcounter * print_count_multiplier)
            printcounter = 0
        printcounter = printcounter + 1
        # additional_duplicates == 0 implies 1 pair of duplicates
        if index_np_array[i] == index_np_array[
                i + 1] and additional_duplicates == -1:
            additional_duplicates = 0
            for j in range(start_index, end_index):
                if j > i + 1 and index_np_array[i] == index_np_array[j]:
                    additional_duplicates = additional_duplicates + 1

            averaged_bids_sample = np.zeros((bids.shape[1], bids.shape[2]))

            averaged_asks_sample = np.zeros((asks.shape[1], asks.shape[2]))

            total_duplicate_volume_bids, total_duplicate_volume_asks = get_total_duplicate_volumes(
                additional_duplicates, bids, asks, i)

            averaged_bids_sample[:, 1] = total_duplicate_volume_bids.reshape(
                averaged_bids_sample[:, 1].shape) / (additional_duplicates + 2)

            averaged_asks_sample[:, 1] = total_duplicate_volume_asks.reshape(
                averaged_asks_sample[:, 1].shape) / (additional_duplicates + 2)

            averaged_bids_sample, averaged_asks_sample = add_up_duplicate_price_times_volume(
                additional_duplicates,
                bids,
                asks,
                averaged_bids_sample,
                averaged_asks_sample,
                i,
            )

            averaged_bids_sample, averaged_asks_sample = divide_weighted_price_by_total_volume(
                bids,
                averaged_bids_sample,
                total_duplicate_volume_bids,
                averaged_asks_sample,
                total_duplicate_volume_asks,
            )

            # if index_np_array[i] == 1586124693529.0:
            #     import pdb

            #     pdb.set_trace()

            averaged_bids[i] = averaged_bids_sample
            averaged_asks[i] = averaged_asks_sample

        elif additional_duplicates > -1:
            additional_duplicates = additional_duplicates - 1
        elif additional_duplicates == -1:
            averaged_bids[i] = bids[i]
            averaged_asks[i] = asks[i]
    return averaged_bids, averaged_asks


@njit(parallel=True)
def get_total_duplicate_volumes(additional_duplicates, bids, asks, i):
    total_duplicate_volume_bids = np.zeros((bids.shape[1], 1))
    total_duplicate_volume_asks = np.zeros((asks.shape[1], 1))

    for k in prange(additional_duplicates + 2):
        for m in prange(bids.shape[1]):
            total_duplicate_volume_bids[m] = (total_duplicate_volume_bids[m] +
                                              bids[i + k][m][1])
            total_duplicate_volume_asks[m] = (total_duplicate_volume_asks[m] +
                                              asks[i + k][m][1])
    return total_duplicate_volume_bids, total_duplicate_volume_asks


@njit(parallel=True)
def add_up_duplicate_price_times_volume(additional_duplicates, bids, asks,
                                        averaged_bids_sample,
                                        averaged_asks_sample, i):
    for k in prange(additional_duplicates + 2):
        for m in prange(bids.shape[1]):
            averaged_bids_sample[m][0] = averaged_bids_sample[m][0] + (
                bids[i + k][m][0] * bids[i + k][m][1])
            averaged_asks_sample[m][0] = averaged_asks_sample[m][0] + (
                asks[i + k][m][0] * asks[i + k][m][1])
    return averaged_bids_sample, averaged_asks_sample


@njit(parallel=True)
def divide_weighted_price_by_total_volume(
    bids,
    averaged_bids_sample,
    total_duplicate_volume_bids,
    averaged_asks_sample,
    total_duplicate_volume_asks,
):
    for m in prange(bids.shape[1]):
        averaged_bids_sample[m][0] = (averaged_bids_sample[m][0] /
                                      total_duplicate_volume_bids[m][0])
        averaged_asks_sample[m][0] = (averaged_asks_sample[m][0] /
                                      total_duplicate_volume_asks[m][0])
    return averaged_bids_sample, averaged_asks_sample


@njit(parallel=True)
def duplicate_fast_search(index_duplicated):
    for i in prange(len(index_duplicated)):
        if index_duplicated[i] == True:
            print(i)
        if i == len(index_duplicated) - 1:
            print(69)


def get_exchange_info():
    r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo")
    return loads(r.content.decode())


@njit
def get_feature_stats(train_lob):
    maxes = np.zeros(train_lob.shape[1])
    mins = np.zeros(train_lob.shape[1])
    means = stds = np.zeros(train_lob.shape[1])
    stds = np.zeros(train_lob.shape[1])
    for i in range(train_lob.shape[1]):
        train_lob_one_feature = train_lob[:, i]
        maxes[i] = np.max(train_lob_one_feature)
        mins[i] = np.min(train_lob_one_feature)
        means[i] = np.mean(train_lob_one_feature)
        stds[i] = np.std(train_lob_one_feature)

    return mins, maxes, means, stds


@njit
def get_integer_indexes_for_last_n_day_scaling(input_features,
                                               psuedo_day_length_in_seconds,
                                               close_index_array):
    feature_span_in_ms = close_index_array[-1] - close_index_array[0]
    number_of_splits = (
        (feature_span_in_ms / 1000) // psuedo_day_length_in_seconds) + 1

    start_and_finish_indexes = np.zeros((int(number_of_splits) - 1, 2))
    start_and_finish_indexes[:] = np.nan

    start_index = None
    split_number = 0
    for i in range(len(close_index_array)):
        if start_index is None:
            start_index = close_index_array[i]
            start_and_finish_indexes[split_number, 0] = i

        else:
            if close_index_array[
                    i] - start_index > 1000 * psuedo_day_length_in_seconds or close_index_array[
                        i] == close_index_array[-1]:
                end_index = close_index_array[i - 1]
                start_and_finish_indexes[split_number, 1] = i
                split_number = split_number + 1
                start_index = None
                end_index = None
    return start_and_finish_indexes.astype(np.int64)


@njit
def normalize_based_on_past_n_days_stats(start_and_finish_indexes,
                                         input_features, scaling_type, minimum,
                                         maximum):
    for i in range(len(start_and_finish_indexes)):
        print(i)
        if i == 0:
            fit_scaling_start_index = 0
            fit_scaling_end_index = 0
            input_features_start_index = 0
            input_features_end_index = 0
        if i > 0:
            fit_scaling_start_index = np.max(np.array([i - 5, 0]))
            fit_scaling_end_index = i - 1
            input_features_start_index = i
            input_features_end_index = i

        maxes_or_means_np_array, mins_or_stds_np_array = get_fit_scalars(
            scaling_type, input_features[:, start_and_finish_indexes[
                fit_scaling_start_index,
                0]:start_and_finish_indexes[fit_scaling_end_index, 1] + 1])

        input_features_normalized = scale_input_features(
            scaling_type,
            maxes_or_means_np_array,
            mins_or_stds_np_array,
            input_features[:, start_and_finish_indexes[
                input_features_start_index,
                0]:start_and_finish_indexes[input_features_end_index, 1] + 1],
            minimum,
            maximum,
        )
        if i == 0:
            previous_input_features_normalized = input_features_normalized.copy(
            )
        elif i == 1:
            appended_input_features_normalized = np.concatenate(
                (previous_input_features_normalized,
                 input_features_normalized),
                axis=1)
        elif i > 1:
            appended_input_features_normalized = np.concatenate(
                (appended_input_features_normalized,
                 input_features_normalized),
                axis=1)

    return appended_input_features_normalized


@njit
def split_train_validation_and_test_prices_for_window_and_close_index_array(
    prices_for_window_index_array, close_index_array, window_length, y):
    # Do all the splitting here for train/val/test
    end_index = round(len(prices_for_window_index_array) * 0.8)
    prices_for_window_index_array_train_and_val = prices_for_window_index_array[:
                                                                                end_index]

    for i in range(len(close_index_array)):
        if prices_for_window_index_array[:end_index][0] == close_index_array[
                i]:
            close_index_array_new_start_index_train_and_val = i
        elif prices_for_window_index_array[:end_index][
                -1] == close_index_array[i]:
            close_index_array_new_end_index_train_and_val = i

    input_features_end_index_train_and_val = close_index_array_new_end_index_train_and_val + 1

    close_index_array_train_and_val = close_index_array[
        close_index_array_new_start_index_train_and_val -
        window_length:close_index_array_new_end_index_train_and_val + 1]

    y_train_and_val = y[:round(len(y) * 0.8)]

    end_index = round(len(prices_for_window_index_array_train_and_val) * 0.8)

    prices_for_window_index_array_train = prices_for_window_index_array_train_and_val[:round(
        len(prices_for_window_index_array_train_and_val) * 0.8)]

    for i in range(len(close_index_array)):
        if prices_for_window_index_array_train_and_val[:end_index][
                0] == close_index_array[i]:
            close_index_array_new_start_index_train = i
        elif prices_for_window_index_array_train_and_val[:end_index][
                -1] == close_index_array[i]:
            close_index_array_new_end_index_train = i

    input_features_start_index_train = close_index_array_new_start_index_train - window_length

    input_features_end_index_train = close_index_array_new_end_index_train + 1

    close_index_array_train = close_index_array[
        close_index_array_new_start_index_train -
        window_length:close_index_array_new_end_index_train + 1]

    y_train = y_train_and_val[:round(len(y_train_and_val) * 0.8)]

    start_index = round(len(prices_for_window_index_array_train_and_val) * 0.8)

    prices_for_window_index_array_val = prices_for_window_index_array_train_and_val[
        end_index:]

    for i in range(len(close_index_array)):
        if prices_for_window_index_array_train_and_val[start_index:][
                0] == close_index_array[i]:
            close_index_array_new_start_index_val = i
        elif prices_for_window_index_array_train_and_val[start_index:][
                -1] == close_index_array[i]:
            close_index_array_new_end_index_val = i

    close_index_array_val = close_index_array[
        close_index_array_new_start_index_val -
        window_length:close_index_array_new_end_index_val + 1]

    input_features_start_index_val = close_index_array_new_start_index_val - window_length

    input_features_end_index_val = close_index_array_new_end_index_val + 1

    y_val = y_train_and_val[round(len(y_train_and_val) * 0.8):]

    start_index = round(len(prices_for_window_index_array) * 0.8)

    prices_for_window_index_array_test = prices_for_window_index_array[
        start_index:]

    for i in range(len(close_index_array)):
        if prices_for_window_index_array[start_index:][0] == close_index_array[
                i]:
            close_index_array_new_start_index_test = i
        elif prices_for_window_index_array[start_index:][
                -1] == close_index_array[i]:
            close_index_array_new_end_index_test = i

    close_index_array_test = close_index_array[
        close_index_array_new_start_index_test - window_length:]

    input_features_start_index_test = close_index_array_new_start_index_test - window_length

    y_test = y[round(len(y) * 0.8):]

    return prices_for_window_index_array_train_and_val, close_index_array_train_and_val, prices_for_window_index_array_train, close_index_array_train, prices_for_window_index_array_val, close_index_array_val, prices_for_window_index_array_test, close_index_array_test, input_features_end_index_train_and_val, input_features_start_index_train, input_features_end_index_train, input_features_start_index_val, input_features_end_index_val, input_features_start_index_test, y_train, y_val, y_test


@njit
def get_labels_by_deeplob_method(df_np_array, horizon, minimum_return):
    first_label_integer_index = 0
    labels = np.zeros(len(df_np_array))
    labels[:] = np.nan
    for i in prange(len(df_np_array)):
        if i >= horizon and i < len(df_np_array) - horizon:
            if first_label_integer_index == 0:
                first_label_integer_index = i
            means_past = np.mean(df_np_array[i - horizon:i + 1])
            means_future = np.mean(df_np_array[i + 1:i + horizon])
            movement = (means_future - means_past) / means_past
            # print(movement)
            if movement > minimum_return:
                labels[i] = 1
            elif movement < -minimum_return:
                labels[i] = -1
            else:
                labels[i] = 0
        elif first_label_integer_index > 0:
            last_label_integer_index = i
            return labels[~np.isnan(
                labels)], first_label_integer_index, last_label_integer_index


@njit
def trim_close_index_array(close_index_array, prices_for_window_index_array):
    for i in range(len(close_index_array)):
        if close_index_array[i] == prices_for_window_index_array[-1]:
            end_index = i
    return close_index_array[:end_index + 1], end_index