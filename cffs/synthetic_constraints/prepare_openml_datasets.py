"""Preparation of OpenML datasets for the study with synthetic constraints

Script which downloads datasets from OpenML and saves them in a format suitable for the
synthetic-constraints pipeline.

Usage: python -m cffs.synthetic_constraints.prepare_openml_datasets --help
"""

import argparse
import pathlib
from typing import Optional, Sequence

import numpy as np
import openml
import tqdm

from cffs.utilities import data_utility


# Download and store OpenML datasets as prediction-ready (X-y format) CSVs.
# Either retrieve datasets by "data_ids" or search according to fixed dataset characteristics.
# Note that further datasets that satisfy the characteristics might be added to OpenML in future,
# so passing the data ids here is the safer way for an exact reproduction of the experiments.
def prepare_openml_datasets(data_dir: pathlib.Path, data_ids: Optional[Sequence[int]] = None) -> None:
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(data_utility.list_datasets(data_dir)) > 0:
        print('Data directory already contains prediction-ready datasets. ' +
              'Files might be overwritten, but not deleted.')
    if (data_ids is None) or (len(data_ids) == 0):
        print('Getting an overview of datasets ...')
        dataset_overview = openml.datasets.list_datasets(status='active', output_format='dataframe')
        dataset_overview = dataset_overview[
            (dataset_overview['format'] != 'Sparse_ARFF') &  # Sparse_ARFF would break when calling get_data()
            (dataset_overview['NumberOfClasses'] == 0) &  # regression datasets
            (dataset_overview['NumberOfNumericFeatures'] >= 11) &  # number includes target column
            (dataset_overview['NumberOfNumericFeatures'] <= 15) &
            (dataset_overview['NumberOfInstances'] >= 100) &
            (dataset_overview['NumberOfInstances'] <= 10000) &
            (dataset_overview['NumberOfMissingValues'] == 0)
        ]
        # Pick latest version of each dataset:
        dataset_overview = dataset_overview.sort_values(by='version').groupby('name').last()
        dataset_overview.to_csv(data_dir / '_data_overview.csv', index=True)  # indexed by name, which should be stored
        data_ids = list(dataset_overview['did'])
    else:
        print('Using pre-defined dataset ids.')
    print('Downloading datasets ...')
    for data_id in tqdm.tqdm(data_ids):
        dataset = openml.datasets.get_dataset(dataset_id=data_id, download_data=True)
        dataset_name = dataset.name
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        assert np.issubdtype(y.dtype, np.number)  # regression problem
        # Feature types are a bit tricky; there is a "categorical_indicator" returned by get_data(),
        # but that is not reliable; "data_type" from class "OpenMLFeature" is better:
        numeric_cols = [feature.name for feature in dataset.features.values() if feature.data_type == 'numeric']
        assert len(numeric_cols) == dataset_overview.loc[dataset_name].NumberOfNumericFeatures
        # "dataset.features" also includes target and might include non-existing columns
        X = X[[col for col in numeric_cols if col in list(X)]]  # drop other column types
        data_utility.save_dataset(X, y, dataset_name=dataset_name, directory=data_dir)


# Parse some command-line arguments, prepare datasets, and save the results.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Retrieves several datasets from OpenML, prepares them for the ' +
        'experiment pipeline and stores them in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/openml/', dest='data_dir',
                        help='Output directory for data.')
    parser.add_argument('-i', '--ids', type=int, default=[], nargs='*', dest='data_ids',
                        help='Data ids. If none provided, will search for suitable datasets automatically.')
    args = parser.parse_args()
    print('Dataset preparation started')
    prepare_openml_datasets(**vars(args))
    print('Datasets prepared and saved.')
