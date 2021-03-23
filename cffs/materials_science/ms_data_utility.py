"""Materials science datasets

Utility functions for loading datasets for our case study in materials science.
"""


import pathlib
import re
from typing import Any, Dict, Optional

import pandas as pd


AGGREGATE_FUNCTIONS = ['min', 'max', 'median', 'sum', 'std']
REACTION_TYPES = ['coll', 'cs', 'glissile', 'hirth', 'inplane', 'lomer', 'multiple_col']


# Dataset is modified in place
def add_slip_system_aggregates(dataset: pd.DataFrame) -> None:
    # Remove existing aggregates (which have different naming scheme):
    aggregate_columns = [x for x in list(dataset) if 'rho_tot' in x or 'q_t' in x]
    dataset.drop(columns=aggregate_columns, inplace=True)
    # Add aggregates for all quantities that are available for multiple slip systems:
    slip_quantities = [x for x in list(dataset) if re.search('_[0-9]+$', x) is not None]
    slip_quantities = {re.sub('_[0-9]+$', '', x) for x in slip_quantities}  # set removes duplicates
    for quantity in slip_quantities:
        for agg_func in AGGREGATE_FUNCTIONS:  # much faster than passing all functions to agg() simultaneously
            dataset[f'{quantity}_{agg_func}'] =\
                dataset[[f'{quantity}_{i}' for i in range(1, 13)]].agg(agg_func, axis='columns')


def prepare_voxel_data(path: str) -> pd.DataFrame:
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
    return dataset


def predict_voxel_data_absolute(dataset: pd.DataFrame, dataset_name: str = '',
                                reaction_type: str = 'glissile', add_aggregates: bool = False) -> Dict[str, Any]:
    target = 'rho_' + reaction_type + '_sum'
    # Exclude all reaction-type-related features (as reaction density is our target):
    features = [x for x in list(dataset) if re.search('rho_(' + '|'.join(REACTION_TYPES) + ')', x) is None]
    # Exclude historic/neigboring-voxel features (we want to predict in same voxel, at same time step):
    features = [x for x in features if re.match('[0-9]+_', x) is None]
    # Exclude delta features (we want to predict in same voxel, at same time step):
    features = [x for x in features if 'delta' not in x]
    # Exclude voxels with zero or with missing reaction density:
    dataset = dataset.loc[(dataset[target] != 0) & (~dataset[target].isna()), features + [target]]
    if add_aggregates:
        add_slip_system_aggregates(dataset)  # in-place
        features = [x for x in list(dataset) if x != target]
    return {'dataset_name': dataset_name, 'target': target, 'features': features, 'dataset': dataset}


# Create overview of physical quantities in dataset (including info if quantity is available
# for multiple slip systems and/or neighboring voxels at previous time step)
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
