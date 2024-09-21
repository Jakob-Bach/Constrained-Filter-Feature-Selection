"""Feature qualities

Functions to compute (univariate) feature qualities for numeric datasets with numeric prediction
targets.

Literature
----------
Bach et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"
"""

import math
from typing import Sequence

import pandas as pd
import sklearn.feature_selection


def abs_corr(X: pd.DataFrame, y: pd.Series) -> Sequence[float]:
    """Absolute correlation

    Computes the absolute value of the Pearson correlation between each feature and the prediction
    target as a measure of univariate feature quality. Taking the absolute values ensures that we
    only measure the strength of the relationship, not their direction.

    Literature
    ----------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Parameters
    ----------
    X : pd.DataFrame
        Dataset (each row is a data object, each column a feature). All values must be numeric.
    y : pd.Series
        Prediction target. Must be numeric and have the same number of entries as `X` has rows.

    Returns
    -------
    Sequence[float]
        The feature qualities (as many as `X` has columns). Missing values (due to a feature or the
        target being constant) are replaced with zero. To speed up the solver for constrained
        feature selection (Z3 uses a rational-number representation instead of float), we round the
        feature qualities.
    """

    result = [round(abs(X[feature].corr(y)), 2) for feature in list(X)]
    return [0 if math.isnan(x) else x for x in result]


def mut_info(X: pd.DataFrame, y: pd.Series) -> Sequence[float]:
    """Mutual information

    Computes the mutual information between each feature and the prediction target as a measure of
    univariate feature quality.

    Literature
    ----------
    https://en.wikipedia.org/wiki/Mutual_information

    Parameters
    ----------
    X : pd.DataFrame
        Dataset (each row is a data object, each column a feature). All values must be numeric.
    y : pd.Series
        Prediction target. Must be numeric and have the same number of entries as `X` has rows.

    Returns
    -------
    Sequence[float]
        The feature qualities (as many as `X` has columns). To speed up the solver for constrained
        feature selection (Z3 uses a rational-number representation instead of float), we round the
        feature qualities.
    """

    result = sklearn.feature_selection.mutual_info_regression(
        X=X, y=y, discrete_features=False, n_neighbors=3, random_state=25)
    return [round(x, 2) for x in result]
