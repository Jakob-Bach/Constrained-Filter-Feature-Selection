"""Materials science datasets

Utility functions for loading datasets for our case study in materials science.
"""


import re
from typing import Any, Dict, Optional, Sequence

import pandas as pd


AGGREGATES = ['min', 'max', 'median', 'sum', 'std']
REACTION_TYPES = ['coll', 'cs', 'glissile', 'hirth', 'inplane', 'lomer', 'multiple_col']


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
                                 delta_steps: int = 1, subset: str = 'none') -> None:
    # Remove pre-computed deltas
    old_delta_features = [x for x in list(dataset) if 'delta_' in x]
    dataset.drop(columns=old_delta_features, inplace=True)
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


# Dataset is modified in place
def add_slip_system_aggregates(dataset: pd.DataFrame) -> None:
    # Remove existing aggregates (which have different naming scheme)
    aggregate_quantities = [x for x in list(dataset) if 'rho_tot' in x or 'q_t' in x]
    dataset.drop(columns=aggregate_quantities, inplace=True)
    # Add new aggregates
    slip_quantities = [x for x in list(dataset) if re.search('_[0-9]+$', x) is not None]
    slip_quantities = {re.sub('_[0-9]+$', '', x) for x in slip_quantities}  # set removes dupes
    for quantity in slip_quantities:
        for agg_func in AGGREGATES:  # much faster than passing all functions to agg() simultaneously
            dataset[f'{quantity}_{agg_func}'] =\
                dataset[[f'{quantity}_{i}' for i in range(1, 13)]].agg(agg_func, axis='columns')


def prepare_sampled_merged_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/sampled_merged_last_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps: int = 1, subset: str = 'none') -> pd.DataFrame:
    dataset = pd.read_csv(path, dtype='float64')  # specifying dtype makes reading faster
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    # String matching functions are more difficult to use if name of one reaction type is substring
    # of another one, thus we rename "multiple_coll" to avoid clashes with "coll"
    dataset.rename(columns=lambda x: re.sub('multiple_coll', 'multiple_col', x), inplace=True)
    # Drop count-based reaction features (we use the density-based versions of these features)
    drop_pattern = '([0-9]+_)?(' + '|'.join(REACTION_TYPES) + ')'
    dataset.drop(columns=[x for x in list(dataset) if re.match(drop_pattern, x) is not None], inplace=True)
    # Compute overall reaction density, as not available by default
    for quantity in [f'rho_{x}' for x in REACTION_TYPES]:
        dataset[f'{quantity}_sum'] = dataset[[f'{quantity}_{i}' for i in range(1, 13)]].sum(axis='columns')
    # Add deltas of reaction quantities over time, remove invalid time steps
    add_deltas_and_sanitize_time(dataset, delta_quantities=[f'rho_{x}_sum' for x in REACTION_TYPES],
                                 delta_steps=delta_steps, subset=subset)
    return dataset


def prepare_sampled_voxel_data(
        path: str = 'C:/MyData/Versetzungsdaten/Voxel_Data/sampled_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps: int = 1, subset: str = 'none') -> pd.DataFrame:
    dataset = pd.read_csv(path, dtype='float64')  # specifying dtype makes reading faster
    dataset.drop(columns=dataset.columns[0], inplace=True)  # drop 1st column (unnamed id column)
    # In current datasets, slip systems notation for shear is "shear(1)" instead "shear_gs(1)"
    dataset.rename(columns=lambda x: re.sub('shear', 'shear_gs', x), inplace=True)
    # In strain rate version of dataset, there is one feature with double brackets
    dataset.rename(columns=lambda x: re.sub('\\(\\(', '(', x), inplace=True)
    # String matching functions are more difficult to use if name of one reaction type is substring
    # of another one, thus we rename "multiple_coll" to avoid clashes with "coll"
    dataset.rename(columns=lambda x: re.sub('multiple_coll', 'multiple_col', x), inplace=True)
    # Make naming consistent to (delta) sampled merged data:
    dataset.rename(columns=lambda x: re.sub(r'gs\(([0-9]+)\)$', r'\1', x), inplace=True)
    for reaction_type in REACTION_TYPES:  # though absolute quantities instead of densities, still make consistent
        dataset.rename(columns=lambda x: re.sub(reaction_type, f'rho_{reaction_type}_sum', x), inplace=True)
    # Add deltas of reaction quantities over time, remove invalid time steps
    dataset = dataset[dataset['time'] % 50 == 0]  # remove irregular time steps
    add_deltas_and_sanitize_time(dataset, delta_quantities=[f'rho_{x}_sum' for x in REACTION_TYPES],
                                 delta_steps=delta_steps, subset=subset)
    return dataset


def predict_voxel_data_absolute(dataset: pd.DataFrame, dataset_name: str = '',
                                reaction_type: str = 'glissile', add_aggregates: bool = False) -> Dict[str, Any]:
    target = 'rho_' + reaction_type + '_sum'
    features = [x for x in list(dataset) if reaction_type not in x]  # exclude features of target reaction type
    features = [x for x in features if re.match('[0-9]+_', x) is None]  # exclude historic features
    features = [x for x in features if 'delta' not in x]  # exclude delta features
    dataset = dataset.loc[(dataset[target] != 0) & (~dataset[target].isna()), features + [target]]
    if add_aggregates:
        add_slip_system_aggregates(dataset)  # in-place
        features = [x for x in list(dataset) if x != target]
    return {'dataset_name': dataset_name, 'target': target, 'features': features, 'dataset': dataset}


def predict_voxel_data_relative(dataset: pd.DataFrame, dataset_name: str = '',
                                reaction_type: str = 'glissile', add_aggregates: bool = False) -> Dict[str, Any]:
    target = 'delta_rho_' + reaction_type + '_sum'
    features = [x for x in list(dataset) if 'delta_rho_' + reaction_type not in x]  # exclude target reaction's deltas
    features = [x for x in features if re.match('[0-9]+_', x) is None]  # exclude historic feature
    dataset = dataset.loc[(dataset[f'rho_{reaction_type}_sum'] != 0) & (~dataset[target].isna()), features + [target]]
    if add_aggregates:
        add_slip_system_aggregates(dataset)  # in-place
        features = [x for x in list(dataset) if x != target]
    return {'dataset_name': dataset_name, 'target': target, 'features': features, 'dataset': dataset}


def summarize_voxel_data(dataset: pd.DataFrame, outfile: Optional[str] = None) -> pd.DataFrame:
    featureTable = pd.DataFrame({'Feature': dataset.columns})
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
