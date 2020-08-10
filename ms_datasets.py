"""Materials science datasets

Utility functions for loading datasets for our case study in materials science.
"""


import re
from typing import Any, Dict, Optional

import pandas as pd


def prepare_delta_voxel_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/delta_sampled_merged_last_voxel_data_size2400_order2_speedUp2.csv',
        subset: str = 'consecutive') -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    # Add summed reaction densities:
    for quantity in ['coll', 'cs', 'glissile', 'hirth', 'inplane', 'lomer', 'multiple_coll']:
        col = 'rho_' + quantity
        dataset[col] = dataset[[col + '_' + str(i) for i in range(1, 13)]].sum(axis='columns')
        col = 'delta_rho_' + quantity
        dataset[col] = dataset[[col + '_' + str(i) for i in range(1, 13)]].sum(axis='columns')
    # Remove invalid time steps (optional)
    times = pd.DataFrame({'time': dataset['time'].sort_values().unique()})
    times['time_diff'] = times - times.shift()
    if subset == 'consecutive':  # consider longest sequence without gaps
        times['group'] = (times['time_diff'] != 50).cumsum()  # macro-time step bigger than usual
        longest_group = times.groupby('group').size().idxmax()
        timestamps = list(times[times['group'] == longest_group].time)[1:]  # longest sequence, without first element
    elif subset == 'complete':  # consider all time steps where prediction can be made
        timestamps = list(times[times['time_diff'] == 50]['time'])
    else:  # use all time steps
        timestamps = times['time']
    dataset = dataset[dataset.time.isin(timestamps)]
    return dataset


def prepare_sampled_voxel_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/sampled_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps: int = 1, subset: str = 'consecutive') -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    dataset = dataset[dataset['time'] % 50 == 0]  # remove irregular time steps
    # Add delta features
    dataset.sort_values(by='time', inplace=True)
    dataset_grouped = dataset.groupby(['pos_x', 'pos_y', 'pos_z'])
    for quantity in ['coll', 'glissile', 'lomer']:
        dataset['delta_' + quantity] = dataset_grouped[quantity].transform(lambda x: x.shift(-delta_steps) - x)
    # Remove invalid time steps (optional)
    times = pd.DataFrame({'time': dataset['time'].sort_values().unique()})
    times['time_diff'] = times - times.shift()
    times['group'] = (times['time_diff'] != 50).cumsum()  # macro-time step unusual
    if subset == 'consecutive':  # consider longest sequence without gaps
        longest_group = times.groupby('group').size().idxmax()
        timestamps = list(times[times['group'] == longest_group].time)[:-delta_steps]
    elif subset == 'complete':  # consider all time steps where prediction can be made
        # For each group, remove the last "delta_steps" elements:
        timestamps = list(times.groupby('group').apply(lambda x: x.drop(labels=x.index[-delta_steps:]))['time'])
    else:  # use all time steps
        timestamps = times['time']
    dataset = dataset[dataset.time.isin(timestamps)]
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
    features = [x for x in features if re.search('^0_', x) is not None]  # only values from previous time step
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
