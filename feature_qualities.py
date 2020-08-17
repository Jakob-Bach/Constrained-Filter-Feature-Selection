"""Feature quality computation

Functions to compute feature qualities for datasets.
You can also run it as a script.

Usage: python feature_qualities.py --help
"""

import argparse
import math
import pathlib
import sys
from typing import Sequence

import pandas as pd
import tqdm

from data_utility import load_dataset, save_qualities


# Z3 uses rational number representation instead of float, so rounding leads to speed-up
def abs_corr(X: pd.DataFrame, y: pd.Series) -> Sequence[float]:
    result = [round(abs(X[feature].corr(y)), 2) for feature in list(X)]
    return [0 if math.isnan(x) else x for x in result]


QUALITIES = {'abs_corr': abs_corr}


def compute_qualities(data_dir: pathlib.Path, qualities: Sequence[str]) -> None:
    if not data_dir.is_dir():
        print('Directory does not exist.')
        sys.exit(1)
    X_files = list(data_dir.glob('*_X.*'))
    y_files = list(data_dir.glob('*_y.*'))
    if len(X_files) != len(y_files):
        print('Number of data files and target variable files differ.')
        sys.exit(1)
    if len(X_files) == 0:
        print('No data files found.')
        sys.exit(1)
    X_dataset_names = [file.name.split('_X.')[0] for file in X_files]
    y_dataset_names = [file.name.split('_y.')[0] for file in y_files]
    if X_dataset_names != y_dataset_names:
        print('Dataset names in data files and target variable files differ.')
        sys.exit(1)
    for dataset_name in tqdm.tqdm(X_dataset_names):
        X, y = load_dataset(dataset_name=dataset_name, directory=data_dir)
        quality_table = pd.DataFrame({'Feature': list(X)})
        for quality in qualities:
            quality_table[quality] = QUALITIES[quality](X, y)
        save_qualities(quality_table, dataset_name=dataset_name, directory=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves several datasets (with X, y stored separately) from a directory, computes ' +
        'one or more feature quality measures for each dataset and stores the result in the same directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/demo/', dest='data_dir',
                        help='Directory for input and output data.')
    parser.add_argument('-q', '--qualities', nargs='+', choices=list(QUALITIES.keys()),
                        default=list(QUALITIES.keys()), help='Feature qualities to be computed.')
    compute_qualities(**vars(parser.parse_args()))
    print('Feature qualities computed and saved.')
