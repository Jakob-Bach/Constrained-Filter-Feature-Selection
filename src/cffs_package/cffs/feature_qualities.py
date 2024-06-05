"""Utility for computing feature qualities

Functions to compute feature qualities for datasets.
"""

import math
from typing import Sequence

import pandas as pd
import sklearn.feature_selection


# Absolute value of correlation between feature and target.
def abs_corr(X: pd.DataFrame, y: pd.Series) -> Sequence[float]:
    # Z3 uses rational-number representation instead of float, so rounding leads to speed-up:
    result = [round(abs(X[feature].corr(y)), 2) for feature in list(X)]
    # Correlation is undefined if standard deviation is zero; replace with zero:
    return [0 if math.isnan(x) else x for x in result]


# Mutual information between feature and target. Use the built-in estimation method from sklearn.
def mut_info(X: pd.DataFrame, y: pd.Series) -> Sequence[float]:
    result = sklearn.feature_selection.mutual_info_regression(
        X=X, y=y, discrete_features=False, n_neighbors=3, random_state=25)
    return [round(x, 2) for x in result]
