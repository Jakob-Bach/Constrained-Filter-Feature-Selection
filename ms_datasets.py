import re

import pandas as pd


def prepare_delta_voxel_data(
        path='C:/MyData/Versetzungsdaten/delta_sampled_merged_last_voxel_data_size2400_order2_speedUp2.csv',
        subset='consecutive'):
    dataset = pd.read_csv(path)
    dataset.drop(columns=list(dataset)[0], inplace=True)  # drop 1st column (unnamed id column)
    # Add summed reaction densities:
    for quantity in ['coll', 'glissile', 'lomer']:
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
        path='C:/MyData/Versetzungsdaten/sampled_voxel_data_size2400_order2_speedUp2.csv',
        delta_steps=1, subset='consecutive'):
    dataset = pd.read_csv(path)
    dataset.drop(columns=list(dataset)[0], inplace=True)  # drop 1st column (unnamed id column)
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


def predict_delta_voxel_data_absolute(dataset, dataset_name='delta_voxel_data', reaction_type='glissile'):
    target = 'rho_' + reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    features = [x for x in features if 'delta' not in x]  # exclude delta features
    features = [x for x in features if 'pos_' not in x and x != 'time']  # exclude position and time
    return {'name': dataset_name + ': ' + reaction_type + ' (absolute)', 'dataset': dataset, 'target': target,
            'features': features}


def predict_delta_voxel_data_relative(dataset, dataset_name='delta_voxel_data', reaction_type='glissile'):
    target = 'delta_rho_' + reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^0_', x) is not None]  # only values from previous time step
    return {'name': dataset_name + ': ' + reaction_type + ' (relative)', 'dataset': dataset, 'target': target,
            'features': features}


def predict_sampled_voxel_data_absolute(dataset, dataset_name='sampled_voxel_data', reaction_type='glissile'):
    target = reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    features = [x for x in features if 'delta' not in x]  # exclude delta features
    features = [x for x in features if 'pos_' not in x and x != 'time']  # exclude position and time
    return {'name': dataset_name + ': ' + reaction_type + ' (absolute)', 'dataset': dataset, 'target': target,
            'features': features}


def predict_sampled_voxel_data_relative(dataset, dataset_name='sampled_voxel_data', reaction_type='glissile'):
    target = 'delta_' + reaction_type
    features = [x for x in list(dataset) if target not in x]  # exclude if feature name contains the target string
    features = [x for x in features if re.search('^([0-9]+)_', x) is None]  # exclude historic features
    return {'name': dataset_name + ': ' + reaction_type + ' (relative)', 'dataset': dataset, 'target': target,
            'features': features}
