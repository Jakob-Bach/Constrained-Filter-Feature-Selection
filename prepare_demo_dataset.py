"""Demo dataset preparation

Script which saves one demo dataset in a format suitable for the experiment pipeline.

Usage: python prepare_demo_dataset.py --help
"""

import argparse
import pathlib

import pandas as pd
from sklearn.datasets import load_boston

from data_utility import save_dataset


def prepare_demo_dataset(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(list(data_dir.glob('*'))) > 0:
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    dataset = load_boston()
    features = dataset['feature_names']
    X = pd.DataFrame(data=dataset['data'], columns=features)
    y = pd.Series(data=dataset['target'], name='target')
    save_dataset(X, y, dataset_name='boston', directory=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepares a demo dataset for the experiment pipeline and stores it ' +
        'in the specified directory.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=str, default='.', help='Output directory for data.')
    prepare_demo_dataset(data_dir=pathlib.Path(parser.parse_args().directory))
    print('Dataset prepared and saved.')
