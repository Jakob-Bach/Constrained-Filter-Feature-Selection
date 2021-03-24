"""Utility for evaluating pipeline results

Functions to prepare experimental results of the prediction pipelines for evaluation, e.g.,
normalization and re-naming of metrics.
"""

from typing import Optional, Sequence

import pandas as pd


# Make objective value relative to dataset/split/quality's max objective (which is the value for
# the unconstrained problem in our experiments).
# Be aware that this function does in-place modification.
def add_normalized_objective(results: pd.DataFrame) -> None:
    results_grouped = results.groupby(['dataset_name', 'split_idx', 'quality_name'])
    results['frac_objective'] = results_grouped['objective_value'].apply(lambda x: x / x.max())


# Make prediction performance relative to dataset/split/quality's max performance.
# Be aware that this function does in-place modification.
def add_normalized_prediction_performance(results: pd.DataFrame) -> None:
    prediction_metrics = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]
    results_grouped = results.groupby(['dataset_name', 'split_idx', 'quality_name'])
    results[[f'frac_{metric}' for metric in prediction_metrics]] =\
        results_grouped[prediction_metrics].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return results


# Make several metrics relative to number of variables.
# Be aware that this function does in-place modification.
def add_normalized_variable_counts(results: pd.DataFrame) -> None:
    results['frac_selected'] = results['num_selected'] / results['num_variables']
    results['frac_constrained_variables'] = results['num_constrained_variables'] / results['num_variables']
    results['frac_unique_constrained_variables'] = results['num_unique_constrained_variables'] / results['num_variables']


# Make number of constraints relative to global maximum number of constraints.
# Be aware that this function does in-place modification.
def add_normalized_num_constraints(results: pd.DataFrame) -> None:
    results['frac_constraints'] = results['num_constraints'] / results['num_constraints'].max()


# Transform prediction data from wide format to long format (i.e., use the three columns "model",
# "split" and "r2" instead of having a separate r2 column for each model-split combination).
# Further "additional_columns" might be kept in the result if desired.
def reshape_prediction_data(results: pd.DataFrame, additional_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if additional_columns is None:
        additional_columns = []
    keep_colums = [x for x in results.columns if x.endswith('_r2')] + additional_columns
    # First, only create one column denoting the model-split combination
    results = results[keep_colums].melt(id_vars=additional_columns, var_name='model', value_name='r2')
    # apply() seems to be faster than Series.str operations
    # Second, move split name to a separate column
    results['split'] = results['model'].apply(lambda x: 'train' if 'train' in x else 'test')
    results['model'] = results['model'].apply(lambda x: x.replace('_train_r2', '').replace('_test_r2', ''))
    return results


# Rename columns and also some values for the paper's plots. "long_metric_name" determines
# whether name + formal identifier or just the formal identifier are used for metrics.
# Note that the names used here are refered to in the evaluation scripts, so if you change
# something here, you should also change it there.
def rename_for_plots(results: pd.DataFrame, long_metric_names: bool = False) -> pd.DataFrame:
    results = results.copy()
    if 'split' in results.columns:
        results['split'] = results['split'].replace({'train': 'Train', 'test': 'Test'})
    if 'model' in results.columns:
        results['model'] = results['model'].replace(
            {'linear-regression': 'Linear regression', 'regression-tree': 'Regression tree',
             'xgb-linear': 'Boosted linear', 'xgb-tree': 'Boosted trees'}
        )
    if 'cardinality' in results.columns:
        results['cardinality'] = results['cardinality'].apply(lambda x: '$n_{se}$=' + str(x))
    results.rename(columns={'model': 'Prediction model', 'split': 'Split', 'r2': '$R^2$',
                            'constraint_name': 'Constraint type', 'dataset_name': 'Dataset name',
                            'cardinality': 'Cardinality'}, inplace=True)
    if long_metric_names:
        results.rename(columns={
            'frac_constraints': 'Number of constraints $n_{co}^{norm}$',
            'frac_solutions': 'Number of solutions $n_{so}^{norm}$',
            'frac_selected': 'Number of selected features $n_{se}^{norm}$',
            'frac_objective': 'Objective value $Q^{norm}$',
            'frac_linear-regression_test_r2': 'Prediction $R^{2, norm}_{lreg}$',
            'frac_xgb-tree_test_r2': 'Prediction $R^{2, norm}_{btree}$'
        }, inplace=True)
    else:
        results.rename(columns={
            'frac_constraints': '$n_{co}^{norm}$', 'frac_constrained_variables': '$n_{cf}^{norm}$',
            'frac_unique_constrained_variables': '$n_{ucf}^{norm}$', 'frac_solutions': '$n_{so}^{norm}$',
            'frac_selected': '$n_{se}^{norm}$', 'frac_objective': '$Q^{norm}$',
            'frac_linear-regression_test_r2': '$R^{2, norm}_{lreg}$',
            'frac_xgb-tree_test_r2': '$R^{2, norm}_{btree}$'
        }, inplace=True)
    return results
