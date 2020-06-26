"""Materials science reaction types

Compare which reaction types domminate over the course of time.
"""


import matplotlib.pyplot as plt

from ms_datasets import prepare_delta_voxel_data, prepare_sampled_voxel_data

SAVE_DIR = '../paper-cffs-material/Reaction-Fractions/'  # use None if plots should not be saved
PLOT_AGGREGATES = ['mean', 'median']  # names of aggregationn function for DataFrame.agg()

sampled_voxel_dataset = prepare_sampled_voxel_data(delta_steps=20, subset='no')
delta_voxel_dataset = prepare_delta_voxel_data(subset='no')
reaction_types = ['coll', 'cs', 'glissile', 'hirth', 'inplane', 'lomer', 'multiple_coll']
prediction_problems = [
    {'name': 'sampled_voxel_n', 'dataset': sampled_voxel_dataset, 'reactions': reaction_types},
    {'name': 'delta_voxel_n', 'dataset': delta_voxel_dataset, 'reactions': reaction_types},
    {'name': 'delta_voxel_rho', 'dataset': delta_voxel_dataset, 'reactions': ['rho_' + x for x in reaction_types]},
]

for problem in prediction_problems:
    reaction_cols = problem['reactions']
    name = problem['name']
    dataset = problem['dataset'][reaction_cols + ['pos_x', 'pos_y', 'pos_z', 'time']].copy()
    dataset['reactions_total'] = dataset[reaction_cols].sum(axis='columns')
    dataset[reaction_cols] = dataset[reaction_cols].apply(lambda x: x / dataset['reactions_total'] * 100)
    for agg_func in PLOT_AGGREGATES:
        dataset.groupby('time')[reaction_cols].agg(agg_func).plot(
            title=f'{agg_func} fraction of reaction types for scenario "{name}"').set(ylabel='%')
        if SAVE_DIR is not None:
            plt.savefig(f'{SAVE_DIR}Reaction_Fraction_{name}_{agg_func}.pdf')
        else:
            plt.show()
