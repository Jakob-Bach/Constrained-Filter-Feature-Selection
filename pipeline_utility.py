"""Pipeline utility functions

Functions to be used in the prediction pipelines.
"""


from typing import Optional, Tuple

import pandas as pd


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
