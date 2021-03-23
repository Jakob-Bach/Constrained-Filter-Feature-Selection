"""Utility for prediction pipelines

Functions for preparing, conducting and evaluating predictions.
"""


from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.base
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree
import xgboost


METRICS = {'r2': sklearn.metrics.r2_score}

MODELS = {
    'linear-regression': {'func': sklearn.linear_model.LinearRegression, 'args': dict()},
    'regression-tree': {'func': sklearn.tree.DecisionTreeRegressor, 'args': {'random_state': 25}},
    'xgb-linear': {'func': xgboost.XGBRegressor,
                   'args': {'booster': 'gblinear', 'n_estimators': 20, 'objective': 'reg:squarederror',
                            'verbosity': 0, 'n_jobs': 1, 'random_state': 25}},
    'xgb-tree': {'func': xgboost.XGBRegressor,
                 'args': {'booster': 'gbtree', 'n_estimators': 20, 'objective': 'reg:squarederror',
                          'verbosity': 0, 'n_jobs': 1, 'random_state': 25}}
}


def create_split_idx(X: pd.DataFrame, n_splits: Optional[int] = 1) ->\
        Sequence[Tuple[Sequence[int], Sequence[int]]]:
    if n_splits < 0:
        raise ValueError('Need to split at least once.')
    if n_splits == 0:
        return [(np.array(range(len(X))), np.array([], dtype='int32'))]
    if n_splits == 1:
        splits = sklearn.model_selection.train_test_split(range(len(X)), train_size=0.8,
                                                          shuffle=True, random_state=25)
        return [(np.sort(splits[0]), np.sort(splits[1]))]  # sort() also creates an array
    splitter = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=25)
    return list(splitter.split(X))


def evaluate_prediction(
        model: sklearn.base.BaseEstimator, X_train: pd.DataFrame, y_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None, y_test: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    if len(X_train.columns) == 0:  # no features selected
        return {**{'train_' + metric_name: float('nan') for metric_name in METRICS.keys()},
                **{'test_' + metric_name: float('nan') for metric_name in METRICS.keys()}}
    model.fit(X_train, y_train)
    results = dict()
    for metric_name, metric_func in METRICS.items():
        pred_train = model.predict(X_train)
        results['train_' + metric_name] = metric_func(y_true=y_train, y_pred=pred_train)
        if (X_test is not None) and (y_test is not None):
            pred_test = model.predict(X_test)
            results['test_' + metric_name] = metric_func(y_true=y_test, y_pred=pred_test)
    return results
