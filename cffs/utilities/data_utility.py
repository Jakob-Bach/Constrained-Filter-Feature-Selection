"""Utility for working with datasets

Functions for reading and writing data. Although the individual functions are quite short,
having a central I/O makes changes of file formats and naming schemes easier. As I/O is not a
performance bottleneck at the moment, we use plain CSV files for serialization.
"""

import pathlib
from typing import Optional, Sequence, Tuple

import pandas as pd


# Feature-part and target-part of a dataset are saved separately.
def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv'), squeeze=True)
    assert isinstance(y, pd.Series)  # a DataFrame would cause errors somewhere in the pipeline
    return X, y


def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str, directory: pathlib.Path) -> None:
    X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    y.to_csv(directory / (dataset_name + '_y.csv'), index=False)


# List dataset names either based on feature-value files or target-values_files.
def list_datasets(directory: pathlib.Path, use_X: bool = True) -> Sequence[str]:
    if use_X:
        return [file.name.split('_X.')[0] for file in list(directory.glob('*_X.*'))]
    return [file.name.split('_y.')[0] for file in list(directory.glob('*_y.*'))]


def load_qualities(dataset_name: str, directory: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(directory / (dataset_name + '_qualities.csv'))


def save_qualities(qualities: pd.DataFrame, dataset_name: str, directory: pathlib.Path) -> None:
    qualities.to_csv(directory / (dataset_name + '_qualities.csv'), index=False)


# There are results files for individual dataset-constraint combinations as well as consolidated
# results files. Both these types only differ in their naming scheme.
def load_results(directory: pathlib.Path, dataset_name: Optional[str] = None,
                 constraint_name: Optional[str] = None) -> pd.DataFrame:
    if (dataset_name is not None) and (constraint_name is not None):
        return pd.read_csv(directory / (dataset_name + '_' + constraint_name + '_results.csv'))
    return pd.read_csv(directory / 'results.csv')


def save_results(results: pd.DataFrame, directory: pathlib.Path, dataset_name: Optional[str] = None,
                 constraint_name: Optional[str] = None) -> None:
    if (dataset_name is not None) and (constraint_name is not None):
        results.to_csv(directory / (dataset_name + '_' + constraint_name + '_results.csv'), index=False)
    else:
        results.to_csv(directory / 'results.csv', index=False)
