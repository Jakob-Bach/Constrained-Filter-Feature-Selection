"""Preparation of a demo dataset for the study with synthetic constraints

Script which saves one demo dataset in a format suitable for the synthetic-constraints pipeline.

Usage: python -m synthetic_constraints.prepare_demo_dataset --help
"""

import argparse
import pathlib

import pandas as pd
import sklearn.datasets

from utilities import data_utility


# Store a sklearn demo dataset as prediction-ready (X-y format) CSVs.
def prepare_demo_dataset(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(data_utility.list_datasets(data_dir)) > 0:
        print('Data directory already contains prediction-ready datasets. ' +
              'Files might be overwritten, but not deleted.')
    dataset = sklearn.datasets.load_boston()
    features = dataset['feature_names']
    X = pd.DataFrame(data=dataset['data'], columns=features)
    y = pd.Series(data=dataset['target'], name='target')
    data_utility.save_dataset(X, y, dataset_name='boston', directory=data_dir)


# Parse some command-line arguments, prepare dataset, and save the results.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepares a demo dataset for the experiment pipeline and stores it ' +
        'in the specified directory.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=str, default='.', help='Output directory for data.')
    args = parser.parse_args()
    print('Dataset preparation started.')
    prepare_demo_dataset(data_dir=pathlib.Path(args.directory))
    print('Dataset prepared and saved.')
