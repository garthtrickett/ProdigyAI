from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

from numba import njit


from .utils import get_train_times


class PurgedKFold(_BaseKFold):
    """Cross Validation with purging and embargo
    
    Params
    ------
    n_splits: int
        The number of splits for cross validation
    t1: pd.Series
        Index and value correspond to the begining and end of information
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for purging
    """

    def __init__(
        self, n_splits=3, t1=None, pct_embargo=0.0, purging=True, num_threads=1
    ):
        super(PurgedKFold, self).__init__(
            n_splits=n_splits, shuffle=False, random_state=None
        )
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be pd.Series")
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purging = purging
        self.num_threads = num_threads

    def split(self, X, y=None, groups=None):
        """Get train and test times stamps
        
        Params
        ------
        X: pd.DataFrame
        y: pd.Series, optional
        
        Returns
        -------
        train_indices, test_indices: np.array
        """

        X_without_index = pd.DataFrame(X.reshape(len(X), X.shape[1]))

        X_without_index.index = self.t1.index
        X = X_without_index
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")
        indices = np.arange(X.shape[0])
        # Embargo width
        embg_size = int(X.shape[0] * self.pct_embargo)
        # Pandas is close set when using [t0:t1]
        test_ranges = [
            (i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)
        ]
        for st, end in test_ranges:
            test_indices = indices[st:end]
            t0 = self.t1.index[st]
            # Avoid look ahead leakage here
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # Edge point of test set in the most recent side
            max_t1_idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            if max_t1_idx < X.shape[0]:
                # Adding indices after test set
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embg_size :])
                )
            # Purging

            if self.purging:
                train_t1 = self.t1.iloc[train_indices]
                test_t1 = self.t1.iloc[test_indices]

                train_t1_copy = train_t1.copy(deep=True)
                train_t1_copy.index = train_t1_copy.index.astype(np.int64)
                train_t1_copy = train_t1_copy.astype(np.int64)
                train_t1_df = train_t1_copy.reset_index(name="index")
                train_t1_np_array = train_t1_df.to_numpy()

                test_t1_copy = test_t1.copy(deep=True)
                test_t1_copy.index = test_t1_copy.index.astype(np.int64)
                test_t1_copy = test_t1_copy.astype(np.int64)
                test_t1_df = test_t1_copy.reset_index(name="index")
                test_t1_np_array = test_t1_df.to_numpy()

                @njit
                def arr_unique(arr):
                    aux = np.sort(arr)
                    flag = np.concatenate(
                        (np.ones(1, dtype=np.bool_), aux[1:] != aux[:-1])
                    )
                    return aux[flag]

                @njit
                def arr_union(ar1, ar2):
                    if ar1.shape[0] == 0:
                        return ar2
                    elif ar2.shape[0] == 0:
                        return ar1
                    else:
                        return arr_unique(np.concatenate((ar1, ar2)))

                @njit
                def get_train_times_numba(train_t1_np_array, test_t1_np_array):
                    for i in range(len(test_t1_np_array)):

                        list_zero = []
                        list_one = []
                        list_two = []

                        for j in range(len(test_t1_np_array)):
                            if (test_t1_np_array[i][0] <= train_t1_np_array[j][0]) & (
                                train_t1_np_array[j][0] <= test_t1_np_array[i][1]
                            ):
                                list_zero.append(train_t1_np_array[j][0])

                            elif (test_t1_np_array[i][0] <= train_t1_np_array[j][1]) & (
                                train_t1_np_array[j][1] <= test_t1_np_array[i][1]
                            ):
                                list_one.append(train_t1_np_array[j][0])

                            elif (train_t1_np_array[j][0] <= test_t1_np_array[i][0]) & (
                                test_t1_np_array[i][1] <= train_t1_np_array[j][1]
                            ):
                                list_two.append(train_t1_np_array[j][0])

                            # df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test
                            # df1=trn[(i<=trn)&(trn<=j)].index # train ends within test
                            # df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test

                            # df0 = trn[(init <= trn.index) & (trn.index <= end)].index
                            # df1 = trn[(init <= trn) & (trn <= end)].index
                            # df2 = trn[(trn.index <= init) & (end <= trn)].index

                        array_zero = np.asarray(list_zero)
                        array_one = np.asarray(list_one)
                        array_two = np.asarray(list_two)

                        indexes_to_drop = arr_union(
                            arr_union(array_zero, array_one), array_two
                        )

                        if len(indexes_to_drop) != 0:
                            new_train_t1_np_array = np.zeros(
                                (len(train_t1_np_array), 2), dtype=np.int64
                            )

                            for i in range(len(train_t1_np_array)):

                                for index in indexes_to_drop:
                                    if train_t1_np_array[i][0] != index:

                                        new_train_t1_np_array[i][0] = train_t1_np_array[
                                            i
                                        ][0]
                                        new_train_t1_np_array[i][1] = train_t1_np_array[
                                            i
                                        ][1]

                            train_t1_np_array = new_train_t1_np_array

                    return train_t1_np_array

                train_t1_np_array = train_t1_np_array.astype(int)
                test_t1_np_array = test_t1_np_array.astype(int)

                print("before" + str(len(train_t1_np_array)))
                train_t1_np_array = get_train_times_numba(
                    train_t1_np_array, test_t1_np_array
                )
                print("after" + str(len(train_t1_np_array)))

                train_t1 = pd.DataFrame(
                    train_t1_np_array, columns=["index", "datetime"]
                )
                train_t1.set_index(["index"], inplace=True)
                train_t1.index = pd.to_datetime(train_t1.index)
                train_t1.datetime = pd.to_datetime(train_t1.datetime)

                # train_t1 = get_train_times(
                #     train_t1, test_t1, num_threads=self.num_threads
                # )

                train_indices = self.t1.index.searchsorted(train_t1.index)

            yield train_indices, test_indices


class CPKFold(object):
    """Cross Validation with purging and embargo
    
    Params
    ------
    n_splits: tuple
        Combinatorial of (n_splits[0], n_splits[1]). n_splits[1] is the number of test.
    t1: pd.Series
        Index and value correspond to the begining and end of information
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for purging
    """

    def __init__(self, n_splits, t1=None, pct_embargo=0.0, purging=True, num_threads=1):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be pd.Series")
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.purging = purging
        self.num_threads = num_threads

    def split(self, X, y=None, groups=None):
        """Get train and test times stamps
        
        Params
        ------
        X: pd.DataFrame
        y: pd.Series, optional
        
        Returns
        -------
        train_indices, test_indices: np.array
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")
        indices = np.arange(X.shape[0])
        # Embargo width
        embg_size = int(X.shape[0] * self.pct_embargo)
        # Generate Combinatorial Pairs for training
        split_indices = np.array_split(indices, self.n_splits[0])
        self._split_locs = np.arange(self.n_splits[0])
        self._test_loc = {i: X.index[idx] for i, idx in enumerate(split_indices)}
        self._test_combs = np.array(
            list(combinations(self._split_locs, self.n_splits[1]))
        )
        train_combs = []
        for comb_idx in self._test_combs:
            train_comb = list(set(self._split_locs).difference(set(comb_idx)))
            train_combs.append(train_comb)

        train_indices_embg = []
        train_indices = []
        for comb_idx in train_combs:
            train_index_embg = []
            train_index = []
            for i in comb_idx:
                if i < self.n_splits[0] - 1:
                    train_index_ = np.hstack(
                        (split_indices[i], split_indices[i + 1][:embg_size])
                    )
                    train_index_embg.append(train_index_)
                    train_index.append(split_indices[i])
                else:
                    train_index_embg.append(split_indices[i])
                    train_index.append(split_indices[i])
            train_indices_embg.append(np.array(list(set(np.hstack(train_index_embg)))))
            train_indices.append(np.array(list(set(np.hstack(train_index)))))

        for train_index, train_index_embg in zip(train_indices, train_indices_embg):
            test_index = np.array(list(set(indices).difference(set(train_index))))
            # Purging
            if self.purging:
                train_t1 = self.t1.iloc[train_index]
                test_t1 = self.t1.iloc[test_index]
                train_t1 = get_train_times(
                    train_t1, test_t1, num_threads=self.num_threads
                )
                train_index = self.t1.index.searchsorted(train_t1.index)

            yield train_index, test_index

    def get_test_combs(self):
        return self._test_combs, self._test_loc


def generate_signals(
    clf,
    X,
    y,
    sample_weight=None,
    n_splits=(4, 2),
    t1=None,
    pct_embargo=0.0,
    purging=True,
    num_threads=1,
    **kwargs
):
    """Cross Validation with default purging and embargo
    
    Params
    ------
    X: pd.DataFrame
    y: pd.Series, optional
    sample_weight: pd.Series, optional
        If specified, apply this to bot testing and training
    n_splits: tuple
        Combinatorial of (n_splits[0], n_splits[1]). n_splits[1] is the number of test.
    t1: pd.Series
        Index and value correspond to the begining and end of information
    pct_embargo: float, default 0
        The percentage of applying embargo
    purging: bool, default True
        If true, apply purging method
    num_threads: int, default 1
        The number of threads for purging
    kwargs: Parameters for scoring function
        
    Returns
    -------
    result: dict(list)
        Each element is signal generated from classifier
    test_times: timestamps
    """
    cv_gen = CPKFold(
        n_splits=n_splits,
        t1=t1,
        pct_embargo=pct_embargo,
        purging=purging,
        num_threads=num_threads,
    )
    signals = []
    for train, test in cv_gen.split(X=X):
        train_params = dict()
        test_params = dict()
        # Sample weight is an optional parameter
        if sample_weight is not None:
            train_params["sample_weight"] = sample_weight.iloc[train].values
            test_params["sample_weight"] = sample_weight.iloc[test].values
        test_params.update(kwargs)
        clf_fit = clf.fit(
            X=X.iloc[train, :].values, y=y.iloc[train].values, **train_params
        )
        # Scoring
        signal = clf_fit.predict_proba(X.iloc[test, :].values)
        signal = pd.DataFrame(signal, index=X.iloc[test].index)
        signals.append(signal)

    combs = cv_gen.get_test_combs()
    result = defaultdict(list)
    test_times = combs[1]
    for signal, comb in zip(signals, combs[0]):
        for i in comb:
            result[i].append(signal.loc[test_times[i]])
    return result, test_times
