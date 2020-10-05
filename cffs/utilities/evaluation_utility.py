"""Evaluation utility functions

Functions to be used to evaluate results of prediction pipelines.
"""


import pandas as pd


# Make objective value relative to dataset/split/quality's max objective; in place
def add_normalized_objective(results: pd.DataFrame) -> None:
    results_grouped = results.groupby(['dataset_name', 'split_idx', 'quality_name'])
    results['frac_objective'] = results_grouped['objective_value'].apply(lambda x: x / x.max())


# Make prediction performance relative to dataset/split/quality's max performance; in place
def add_normalized_prediction_performance(results: pd.DataFrame) -> None:
    prediction_metrics = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]
    results_grouped = results.groupby(['dataset_name', 'split_idx', 'quality_name'])
    results[[f'frac_{metric}' for metric in prediction_metrics]] =\
        results_grouped[prediction_metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return results


# Make number of constraints relative to global maximum number of constraints; in place
def add_normalized_num_constraints(results: pd.DataFrame) -> None:
    results['frac_constraints'] = results['num_constraints'] / results['num_constraints'].max()


# Transform prediction data from wide format to long format
def reshape_prediction_data(results: pd.DataFrame) -> pd.DataFrame:
    prediction_metrics = [x for x in results.columns if x.endswith('_r2')]
    results = results[prediction_metrics].melt(var_name='model', value_name='r2')
    # apply() seems to be faster than Series.str operations
    results['split'] = results['model'].apply(lambda x: 'train' if 'train' in x else 'test')
    results['model'] = results['model'].apply(lambda x: x.replace('_train_r2', '').replace('_test_r2', ''))
    return results
