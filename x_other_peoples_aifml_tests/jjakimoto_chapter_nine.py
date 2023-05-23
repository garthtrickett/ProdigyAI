import numpy as np
import pandas as pd

from finance_ml.model_selection import PurgedKFold


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline


# Extends the sklearn pipeline class to allow the sample_weight to be added
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


def clf_hyper_fit(
    feat,
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
    **fit_params
):
    # Set defaut value for scoring
    if scoring is None:
        if set(label.values) == {0, 1}:
            scoring = "f1"
        else:
            scoring = "neg_log_loss"
    # HP serach on traing data
    inner_cv = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    if rnd_search_iter == 0:
        # If this code is run its a GridSearch
        search = GridSearchCV(
            estimator=pipe_clf,
            param_grid=search_params,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
            iid=False,
        )
    else:
        # If this code is run its a randomized search
        search = RandomizedSearchCV(
            estimator=pipe_clf,
            param_distributions=search_params,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
        )
        bag_est = best_pipe.fit(
            feat,
            label,
            sample_weight=fit_params[
                bag_est.base_estimator.steps[-1][0] + "__sample_weight"
            ],
        )
        best_pipe = Pipeline([("bag", bag_est)])
    return best_pipe


from scipy.stats import rv_continuous


class LogUniformGen(rv_continuous):
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a=1, b=np.exp(1)):
    return LogUniformGen(a=a, b=b, name="log_uniform")


a = 1e-3
b = 1e3
size = 10000
vals = log_uniform(a=a, b=b).rvs(size=size)


from finance_ml.datasets import get_cls_data


X, label = get_cls_data(n_features=10, n_informative=5, n_redundant=0, n_samples=10000)
print(X.head())
print(label.head())


from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 9.1 Use GridSearchCV on 10-fold CV to find the C,
# gamma optimal hyperparameters on a SVC with RBF kernel,
# where param_grid={'C':[1E2,1E-1,1,10,100],'gamma':[1E-2,1E-1,1,10,100]} and
# the scoring function is neg_log_loss.

name = "svc"
params_grid = {
    name + "__C": [1e-2, 1e-1, 1, 10, 100],
    name + "__gamma": [1e-2, 1e-1, 1, 10, 100],
}
kernel = "rbf"
clf = SVC(kernel=kernel, probability=True)
pipe_clf = Pipeline([(name, clf)])
fit_params = dict()

clf = clf_hyper_fit(
    X,
    label["bin"],
    t1=label["t1"],
    pipe_clf=pipe_clf,
    scoring="neg_log_loss",
    search_params=params_grid,
    n_splits=3,
    bagging=[0, None, 1.0],
    rnd_search_iter=0,
    n_jobs=-1,
    pct_embargo=0.0,
    **fit_params
)


# 9.2 Use RandomizedSearchCV on 10-fold CV to find the C,
# gamma optimal hyper-parameters on an SVC with RBF kernel,
# where param_distributions={'C':logUniform(a=1E-2,b=
# 1E2),'gamma':logUniform(a=1E-2,b=1E2)},n_iter=25 and
# neg_log_loss is the scoring function

name = "svc"
params_dist = {
    name + "__C": log_uniform(a=1e-2, b=1e2),
    name + "__gamma": log_uniform(a=1e-2, b=1e2),
}
kernel = "rbf"
clf = SVC(kernel=kernel, probability=True)
pipe_clf = Pipeline([(name, clf)])
fit_params = dict()

clf = clf_hyper_fit(
    X,
    label["bin"],
    t1=label["t1"],
    pipe_clf=pipe_clf,
    scoring="neg_log_loss",
    search_params=params_grid,
    n_splits=3,
    bagging=[0, None, 1.0],
    rnd_search_iter=25,
    n_jobs=-1,
    pct_embargo=0.0,
    **fit_params
)

