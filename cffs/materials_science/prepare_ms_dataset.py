"""Preparation of a dataset for the case study in materials science

Script which pre-processes and saves a materials-science dataset in a format suitable for the
materials-science pipeline.

Usage: python -m cffs.materials_science.prepare_ms_dataset --help
"""

import argparse
import pathlib

from cffs.utilities import data_utility
from cffs.materials_science import ms_data_utility


REACTION_TYPE = 'glissile'  # focus on one type of dislocation reactions


# Load raw CSV, pre-process, and store as prediction-ready (X-y format) CSVs.
def prepare_ms_dataset(input_file: pathlib.Path, data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Data directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(data_utility.list_datasets(data_dir)) > 0:
        print('Data directory already contains prediction-ready datasets. ' +
              'Files might be overwritten, but not deleted.')
    dataset = ms_data_utility.preprocess_voxel_data(path=input_file)
    prediction_scenario = ms_data_utility.prepare_prediction_scenario(
        dataset=dataset, reaction_type=REACTION_TYPE, add_aggregates=True)
    data_utility.save_dataset(X=prediction_scenario['dataset'][prediction_scenario['features']],
                              y=prediction_scenario['dataset'][prediction_scenario['target']],
                              dataset_name=f'{input_file.stem}_predict_{REACTION_TYPE}',
                              directory=data_dir)


# Parse some command-line arguments, prepare dataset, and save the results.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepares a materials science dataset for the MS pipeline and stores the ' +
        'pre-processed version in the specified directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=pathlib.Path, default='data/ms/voxel_data.csv',
                        dest='input_file', help='Input CSV file.')
    parser.add_argument('-d', '--directory', type=pathlib.Path, default='data/ms/',
                        dest='data_dir', help='Output directory for data.')
    args = parser.parse_args()
    print('Dataset preparation started.')
    prepare_ms_dataset(**vars(args))
    print('Dataset prepared and saved.')
