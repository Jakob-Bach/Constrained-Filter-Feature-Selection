"""Materials science dataset preparation

Script which saves multiple materials science dataset in a format suitable for the MS pipeline.

Usage: python ms_prepare_datasets.py --help
"""


import argparse
import pathlib

import tqdm

import data_utility
import ms_datasets


MS_DATA_PATHS = {
    'delta_sampled_2400': pathlib.Path(
        'C:/MyData/Versetzungsdaten/Voxel_Data/delta_sampled_merged_last_voxel_data_size2400_order2_speedUp2.csv'),
    'delta_sampled_2400_strain': pathlib.Path(
        'C:/MyData/Versetzungsdaten/Balduin_config41/delta_sampled_merged_last_voxel_data_size2400_order2_speedUp2_strain_rate.csv'),
    'sampled_voxel_2400': pathlib.Path(
        'C:/MyData/Versetzungsdaten/Voxel_Data/sampled_voxel_data_size2400_order2_speedUp2.csv'),
    'sampled_voxel_2400_strain': pathlib.Path(
        'C:/MyData/Versetzungsdaten/Balduin_config41/sampled_voxel_data_size2400_order2_speedUp2_strain_rate.csv')
}


def prepare_ms_datasets(data_dir: pathlib.Path) -> None:
    if not data_dir.is_dir():
        print('Directory does not exist. We create it.')
        data_dir.mkdir(parents=True)
    if len(list(data_dir.glob('*'))) > 0:
        print('Data directory is not empty. Files might be overwritten, but not deleted.')
    for dataset_name, data_path in tqdm.tqdm(MS_DATA_PATHS.items()):
        if data_path.stem.startswith('delta'):
            dataset = ms_datasets.prepare_delta_voxel_data(data_path)
        else:
            dataset = ms_datasets.prepare_sampled_voxel_data(data_path)
        prediction_scenario = ms_datasets.predict_voxel_data_absolute(
            dataset=dataset, dataset_name=dataset_name + '_absolute', reaction_type='glissile')
        data_utility.save_dataset(X=prediction_scenario['dataset'][prediction_scenario['features']],
                                  y=prediction_scenario['dataset'][prediction_scenario['target']],
                                  dataset_name=dataset_name + '_absolute_glissile', directory=data_dir)
        prediction_scenario = ms_datasets.predict_voxel_data_relative(
            dataset=dataset, dataset_name=dataset_name + '_relative', reaction_type='glissile')
        data_utility.save_dataset(X=prediction_scenario['dataset'][prediction_scenario['features']],
                                  y=prediction_scenario['dataset'][prediction_scenario['target']],
                                  dataset_name=dataset_name + '_relative_glissile', directory=data_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepares multiple material science datasets for the MS pipeline and stores them ' +
        'in the specified directory.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--directory', type=str, default='data/ms/', help='Output directory for data.')
    prepare_ms_datasets(data_dir=pathlib.Path(parser.parse_args().directory))
    print('Datasets prepared and saved.')
