"""Materials science reaction types

Compare which reaction types domminate over the course of time.
"""


import os

import matplotlib.pyplot as plt

import ms_datasets

SAVE_DIR = '../paper-cffs-material/Reaction-Fractions/'  # use None if plots should not be saved
SPATIAL_AGGREGATES = ['mean', 'median']  # aggregate over voxels at each time step
REACTION_TYPES = ms_datasets.REACTION_TYPES

sampled_voxel_dataset = ms_datasets.prepare_sampled_voxel_data(subset='none')
sampled_merged_dataset = ms_datasets.prepare_sampled_merged_data(subset='none')
scenarios = [
    {'name': 'sampled_voxel_n', 'dataset': sampled_voxel_dataset,
     'reactions': ['rho_' + x for x in REACTION_TYPES]},  # only renamed to "rho_", but actually a count
    {'name': 'sampled_merged_n', 'dataset': sampled_merged_dataset,
     'reactions': REACTION_TYPES},
    {'name': 'sampled_merged_rho', 'dataset': sampled_merged_dataset,
     'reactions': ['rho_' + x for x in REACTION_TYPES]},
]
if (SAVE_DIR is not None) and (not os.path.isdir(SAVE_DIR)):
    os.makedirs(SAVE_DIR)

for scenario in scenarios:
    reaction_cols = scenario['reactions']
    name = scenario['name']
    dataset = scenario['dataset'][reaction_cols + ['pos_x', 'pos_y', 'pos_z', 'time']].copy()
    dataset['reactions_total'] = dataset[reaction_cols].sum(axis='columns')
    dataset[reaction_cols] = dataset[reaction_cols].apply(lambda x: x / dataset['reactions_total'] * 100)
    for agg_func in SPATIAL_AGGREGATES:
        dataset.groupby('time')[reaction_cols].agg(agg_func).plot(
            title=f'{agg_func} fraction of reaction types for scenario "{name}"').set(ylabel='%')
        plt.tight_layout()
        if SAVE_DIR is not None:
            plt.savefig(f'{SAVE_DIR}Reaction_Fraction_{name}_{agg_func}.pdf')
        else:
            plt.show()
