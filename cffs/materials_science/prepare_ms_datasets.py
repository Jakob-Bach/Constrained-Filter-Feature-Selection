"""Materials science dataset preparation

Script which saves multiple materials science dataset in a format suitable for the MS pipeline.

Usage: python prepare_ms_datasets.py --help
"""


import argparse
import pathlib

from cffs.utilities import data_utility
from cffs.materials_science import ms_data_utility


DATA_PATH = pathlib.Path('C:/MyData/Versetzungsdaten/Balduin_config41/sampled_merged_last_voxel_data_size2400_order2_speedUp2_strain_rate.csv')
DATA_NAME = 'sampled_merged_2400_strain'
REACTION_TYPE = 'glissile'


def prepare_ms_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(list(data_dir.glob('*'))) > 0:
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    dataset = ms_data_utility.prepare_voxel_data(DATA_PATH)
    prediction_scenario = ms_data_utility.predict_voxel_data_absolute(
        dataset=dataset, reaction_type=REACTION_TYPE, add_aggregates=True)
    data_utility.save_dataset(X=prediction_scenario['dataset'][prediction_scenario['features']],
                              y=prediction_scenario['dataset'][prediction_scenario['target']],
                              dataset_name=DATA_NAME + '_absolute_' + REACTION_TYPE, directory=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepares multiple material science datasets for the MS pipeline and stores them ' +
        'in the specified directory.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=str, default='data/ms/', help='Output directory for data.')
    prepare_ms_datasets(data_dir=pathlib.Path(parser.parse_args().directory))
    print('Datasets prepared and saved.')
