"""Materials science datasets

Utility functions for loading datasets for our case study in materials science.
"""


import re
from typing import Any, Dict, Optional, Sequence

import pandas as pd


REACTION_TYPES = ['coll', 'cs', 'glissile', 'hirth', 'inplane', 'lomer', 'multiple_coll']


def determine_valid_times(times: pd.Series, delta_steps: int = 0) -> pd.DataFrame:
    result = pd.DataFrame({'time': times.sort_values().unique()})
    # A time is valid if we also have time + prediction horizon available
    result['target_time'] = result['time'] + delta_steps * 50
    result['is_valid'] = result['target_time'].isin(result['time'])
    # Determine groups of consecutive valid time steps, i.e., group begins with valid time steps
    # where prior time step is invalid or more than 50 time units away
    result['valid_group'] = (result['is_valid'] &
                             (~result['is_valid'].shift().fillna(False) |
                              (result['time'] - result['time'].shift() != 50))).cumsum()
    result.loc[~result['is_valid'], 'valid_group'] = 0
    return result


# Dataset is modified in place
def add_deltas_and_sanitize_time(dataset: pd.DataFrame, delta_quantities: Sequence[str] = REACTION_TYPES,
                                 delta_steps: int = 1, subset: str = 'none'):
    time_validity_info = determine_valid_times(times=dataset['time'], delta_steps=delta_steps)
    # Add delta features (forward delta, difference between reaction density "delta_steps" in future
    # and reaction density now); as there are missing time steps, we cannot use shift()
    dataset.sort_values(by=['time', 'pos_x', 'pos_y', 'pos_z'], inplace=True)
    for time, target_time in zip(time_validity_info.loc[time_validity_info['is_valid'], 'time'],
                                 time_validity_info.loc[time_validity_info['is_valid'], 'target_time']):
        for quantity in delta_quantities:
            dataset.loc[dataset['time'] == time, 'delta_' + quantity] =\
                dataset.loc[dataset['time'] == target_time, quantity].values -\
                    dataset.loc[dataset['time'] == time, quantity].values
    # Remove invalid time steps
    if subset == 'consecutive':  # consider longest sequence without gaps
        largest_group = time_validity_info[time_validity_info['is_valid']].groupby('valid_group').size().idxmax()
        timestamps = time_validity_info.loc[time_validity_info['valid_group'] == largest_group, 'time']
    elif subset == 'complete':  # consider all time steps where prediction can be made
        timestamps = time_validity_info.loc[time_validity_info['is_valid'], 'time']
    else:  # use all time steps
        timestamps = time_validity_info['time']
    dataset.drop(dataset[~dataset['time'].isin(timestamps)].index, inplace=True)


def prepare_delta_voxel_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/delta_sampled_merged_last_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps: int = 1, subset: str = 'none') -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    quantities = ['rho_' + x for x in REACTION_TYPES]
    for quantity in quantities:  # overall density not available per default, so compute it
        dataset[quantity] = dataset[[quantity + '_' + str(i) for i in range(1, 13)]].sum(axis='columns')
    add_deltas_and_sanitize_time(dataset, delta_quantities=quantities,
                                 delta_steps=delta_steps, subset=subset)
    return dataset


def prepare_sampled_voxel_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/sampled_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps: int = 1, subset: str = 'none') -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    dataset = dataset[dataset['time'] % 50 == 0]  # remove irregular time steps
    add_deltas_and_sanitize_time(dataset, delta_quantities=REACTION_TYPES,
                                 delta_steps=delta_steps, subset=subset)
    return dataset


def predict_delta_voxel_data_absolute(dataset: pd.DataFrame, dataset_name: str = 'delta_voxel_data',
                                      reaction_type: str = 'glissile') -> Dict[str, Any]:
    target = 'rho_' + reaction_type
    # Exclude if feature name contains the reaction type (but don't exclude multiple_coll for coll)
    features = [x for x in list(dataset) if (reaction_type not in x) or ('multiple' in x)]
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    features = [x for x in features if 'delta' not in x]  # exclude delta features
    features = [x for x in features if 'pos_' not in x and x != 'time']  # exclude position and time
    return {'dataset_name': dataset_name, 'dataset': dataset[dataset[target] != 0],
            'target': target, 'features': features}


def predict_delta_voxel_data_relative(dataset: pd.DataFrame, dataset_name: str = 'delta_voxel_data',
                                      reaction_type: str = 'glissile') -> Dict[str, Any]:
    target = 'delta_rho_' + reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic feature
    return {'dataset_name': dataset_name, 'dataset': dataset[dataset['rho_' + reaction_type] != 0],
            'target': target, 'features': features}


def predict_sampled_voxel_data_absolute(dataset: pd.DataFrame, dataset_name: str = 'sampled_voxel_data',
                                        reaction_type: str = 'glissile') -> Dict[str, Any]:
    target = reaction_type
    # Exclude if feature name contains the reaction type (but don't exclude multiple_coll for coll)
    features = [x for x in list(dataset) if (reaction_type not in x) or ('multiple' in x)]
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    features = [x for x in features if 'delta' not in x]  # exclude delta features
    features = [x for x in features if 'pos_' not in x and x != 'time']  # exclude position and time
    return {'dataset_name': dataset_name, 'dataset': dataset[dataset[target] != 0],
            'target': target, 'features': features}


def predict_sampled_voxel_data_relative(dataset: pd.DataFrame, dataset_name: str = 'sampled_voxel_data',
                                        reaction_type: str = 'glissile') -> Dict[str, Any]:
    target = 'delta_' + reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    return {'dataset_name': dataset_name, 'dataset': dataset[dataset[reaction_type] != 0],
            'target': target, 'features': features}


def summarize_voxel_data(dataset: pd.DataFrame, outfile: Optional[str] = None) -> pd.DataFrame:
    featureTable = pd.DataFrame({'Feature': dataset.columns})
    # For sampled_voxel_data, first adapt slip system notation, e.g., replace "gs(3)" by "3"
    featureTable['Feature'] = featureTable['Feature'].str.replace(r'gs\(([0-9]+)\)$', r'\1')
    # Neighboring voxels are indicated like "3_feature", slip systems like "feature_3"
    featureTable['Quantity'] = featureTable['Feature'].str.replace('(^[0-9]+_)|(_[0-9]+$)', '')
    featureTable['Slip_System'] = featureTable['Feature'].str.extract('_([0-9]+)$', expand=False)
    featureTable['History_Neighbors'] = featureTable['Feature'].str.extract('^([0-9]+)_', expand=False)
    overviewTable = featureTable.groupby('Quantity').agg(
        Slip_Systems=('Slip_System', 'nunique'),
        Neighbors=('History_Neighbors', 'nunique'),
        Total=('Quantity', 'size'))
    overviewTable.sort_values(by='Quantity')
    if outfile is not None:
        overviewTable.to_csv(outfile)
    return overviewTable
