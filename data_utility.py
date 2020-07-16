"""Data utility functions

Functions for reading and writing data. Although the individual functions are quite short,
having a central I/O makes changes of file formats and naming schemes easier.
"""

import pathlib
from typing import Optional, Sequence, Tuple

import pandas as pd


def load_dataset(dataset_name: str, directory: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(directory / (dataset_name + '_X.csv'))
    y = pd.read_csv(directory / (dataset_name + '_y.csv'), squeeze=True)
    assert type(y).__name__ == 'Series'
    return X, y


def save_dataset(X: pd.DataFrame, y: pd.Series, dataset_name: str, directory: pathlib.Path) -> None:
    X.to_csv(directory / (dataset_name + '_X.csv'), index=False)
    y.to_csv(directory / (dataset_name + '_y.csv'), index=False)


def list_datasets(directory: pathlib.Path) -> Sequence[str]:
    return [file.name.split('_X.')[0] for file in list(directory.glob('*_X.*'))]


def load_qualities(dataset_name: str, directory: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(directory / (dataset_name + '_qualities.csv'))


def save_qualities(qualities: pd.DataFrame, dataset_name: str, directory: pathlib.Path) -> None:
    qualities.to_csv(directory / (dataset_name + '_qualities.csv'), index=False)


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
