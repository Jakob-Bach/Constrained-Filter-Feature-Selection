"""Pipeline utility functions

Functions to be used in the prediction pipelines.
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


# Adapted from an answer by NISHA DAGA at https://stackoverflow.com/a/44674459
def drop_correlated_features(X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None,
                             threshold: float = 0.95) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if threshold is None:
        return X_train, X_test
    if (threshold < 0) or (threshold > 1):
        raise ValueError(f'Correlation threshold of {threshold} is not in expected range [0,1].')
    corr_cols = []
    corr_df = X_train.corr().abs()
    for i in range(len(corr_df.columns)):
        if corr_df.columns[i] not in corr_cols:
            for j in range(i):
                if (corr_df.iloc[i, j] >= threshold) and (corr_df.columns[j] not in corr_cols):
                    corr_cols.append(corr_df.columns[i])
    X_train = X_train.drop(columns=corr_cols)
    if X_test is not None:
        X_test = X_test.drop(columns=corr_cols)
    return X_train, X_test


def drop_low_quality_features(qualities: Sequence[float], X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None,
                              threshold: float = 0.1) -> Tuple[Sequence[float], pd.DataFrame, Optional[pd.DataFrame]]:
    if len(qualities) != len(X_train.columns):
        raise ValueError('Number of qualities needs to match number of columns.')
    if threshold is None:
        return qualities, X_train, X_test
    keep_col = [x >= threshold for x in qualities]
    qualities = [x for x in qualities if x >= threshold]
    X_train = X_train.loc[:, keep_col]
    if X_test is not None:
        X_test = X_test.loc[:, keep_col]
    return qualities, X_train, X_test


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
