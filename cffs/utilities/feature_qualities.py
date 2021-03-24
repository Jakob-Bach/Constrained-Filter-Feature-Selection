"""Utility for computing feature qualities

Functions to compute feature qualities for datasets.
You can also run the file as a script. However, this is not necessary to execute the
experimental pipelines, as the pipelines call the quality-computation functions themselves.
(Since the pipelines consider train-test splitting when computing feature qualities.)

Usage: python -m cffs.utilities.feature_qualities --help
"""

import argparse
import math
import pathlib
from typing import Sequence

import pandas as pd
import sklearn.feature_selection
import tqdm

import cffs.utilities.data_utility


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


QUALITIES = {'abs_corr': abs_corr, 'mut_info': mut_info}  # defaults for experimental pipeline


# Compute feature qualities for a directory of datasets in X-y representation and save results
# as files (one output file per dataset) in the same directory. Parameter "qualities" needs to
# contain strings that are keys in the dictionary "QUALITIES".
def compute_qualities(data_dir: pathlib.Path, qualities: Sequence[str]) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError('Directory does not exist.')
    X_dataset_names = cffs.utilities.data_utility.list_datasets(directory=data_dir, use_X=True)
    y_dataset_names = cffs.utilities.data_utility.list_datasets(directory=data_dir, use_X=False)
    if len(X_dataset_names) == 0:
        raise FileNotFoundError('No data files found.')
    if len(X_dataset_names) != len(y_dataset_names):
        raise FileNotFoundError('Number of feature-value files and target-value files differ.')
    if X_dataset_names != y_dataset_names:
        raise FileNotFoundError('Dataset names for feature-value files and target-value files differ.')
    print('Computing feature qualities ...')
    for dataset_name in tqdm.tqdm(X_dataset_names):
        X, y = cffs.utilities.data_utility.load_dataset(dataset_name=dataset_name, directory=data_dir)
        quality_table = pd.DataFrame({'Feature': list(X)})
        for quality in qualities:
            quality_table[quality] = QUALITIES[quality](X, y)
        cffs.utilities.data_utility.save_qualities(quality_table, dataset_name=dataset_name, directory=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves datasets (with X, y stored separately) from a directory, ' +
        'computes one or more feature quality measures for each dataset ' +
        'and stores the results in the same directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/openml/', dest='data_dir',
                        help='Directory for input and output data.')
    parser.add_argument('-q', '--qualities', nargs='+', choices=list(QUALITIES.keys()),
                        default=list(QUALITIES.keys()), help='Feature qualities to be computed.')
    compute_qualities(**vars(parser.parse_args()))
    print('Feature qualities computed and saved.')
