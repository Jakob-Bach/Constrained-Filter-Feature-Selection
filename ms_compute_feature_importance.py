"""Material science feature importance

Script which computes feature importances for several datasets and prediction
targets from our case study in materials science.
"""


import pandas as pd

from ms_datasets import *

SAVE_DIR = None

sampled_voxel_dataset = prepare_sampled_voxel_data(delta_steps=20, subset='consecutive')
delta_voxel_dataset = prepare_delta_voxel_data(subset='consecutive')
prediction_problems = []
for reaction_type in ['coll', 'lomer', 'glissile']:
    prediction_problems.append(predict_sampled_voxel_data_absolute(dataset=sampled_voxel_dataset,
                                                                   reaction_type=reaction_type))
    prediction_problems.append(predict_sampled_voxel_data_relative(dataset=sampled_voxel_dataset,
                                                                   reaction_type=reaction_type))
    prediction_problems.append(predict_delta_voxel_data_absolute(dataset=delta_voxel_dataset,
                                                                 reaction_type=reaction_type))
    prediction_problems.append(predict_delta_voxel_data_relative(dataset=delta_voxel_dataset,
                                                                 reaction_type=reaction_type))

results = {}
for problem in prediction_problems:
    dataset = problem['dataset']
    importances = [abs(dataset[feature].corr(dataset[problem['target']])) for feature in problem['features']]
    importance_table = pd.DataFrame({'Feature': problem['features'], 'Importance': importances})
    importance_table.sort_values(by='Importance', ascending=False, inplace=True)
    results[problem['name']] = importance_table
    if SAVE_DIR is not None:
        file_name = problem['name'].replace('(', '').replace(')', '').replace(':', '').replace(' ', '_')
        importance_table.to_csv(SAVE_DIR + '/' + file_name + '.csv', index=False)
