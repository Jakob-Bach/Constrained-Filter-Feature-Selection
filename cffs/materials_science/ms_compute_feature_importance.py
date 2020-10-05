"""Material science feature importance

Script which computes feature importances for several datasets and prediction
targets from our case study in materials science.
"""


import pandas as pd

from cffs.utilities import feature_qualities
from cffs.materials_science import ms_datasets

SAVE_DIR = None
REACTION_TYPES = ['coll', 'lomer', 'glissile']

sampled_voxel_dataset = ms_datasets.prepare_sampled_voxel_data(delta_steps=20, subset='consecutive')
sampled_merged_dataset = ms_datasets.prepare_sampled_merged_data(delta_steps=20, subset='consecutive')
prediction_problems = []
for reaction_type in REACTION_TYPES:
    prediction_problems.append(ms_datasets.predict_voxel_data_absolute(
        dataset=sampled_voxel_dataset, reaction_type=reaction_type, dataset_name='sampled_voxel_data'))
    prediction_problems.append(ms_datasets.predict_voxel_data_relative(
        dataset=sampled_voxel_dataset, reaction_type=reaction_type, dataset_name='sampled_voxel_data'))
    prediction_problems.append(ms_datasets.predict_voxel_data_absolute(
        dataset=sampled_merged_dataset, reaction_type=reaction_type, dataset_name='sampled_merged_data'))
    prediction_problems.append(ms_datasets.predict_voxel_data_relative(
        dataset=sampled_merged_dataset, reaction_type=reaction_type, dataset_name='sampled_merged_data'))

results = {}
for problem in prediction_problems:
    problem_name = problem['dataset_name'] + '_' + problem['target']
    X = problem['dataset'][problem['features']]
    y = problem['dataset'][problem['target']]
    importance_table = pd.DataFrame({'Feature': problem['features']})
    for quality_name, quality_func in feature_qualities.QUALITIES.items():
        importance_table[quality_name] = quality_func(X, y)
    importance_table.sort_values(by=importance_table.columns[1], ascending=False, inplace=True)
    results[problem_name] = importance_table
    if SAVE_DIR is not None:
        importance_table.to_csv(SAVE_DIR + '/' + problem_name + '.csv', index=False)
