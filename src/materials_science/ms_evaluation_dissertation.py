"""Evaluation of the case study in materials science

Script to compute all summary statistics and create all plots used in the dissertation to evaluate
the case study in materials science. Should be run after the experimental pipeline.

Usage: python -m materials_science.ms_evaluation_dissertation --help
"""

import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cffs import feature_qualities
from utilities import data_utility
from utilities import evaluation_utility


plt.rcParams['font.family'] = 'Arial'


# Create and save all plots to evaluate the case study in materials science for the dissertation.
# To that end, read results from the "results_dir" and some dataset information from the
# "data_dir"; save plots to the "plot_dir" and print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    X, y = data_utility.load_dataset(dataset_name=data_utility.list_datasets(directory=data_dir)[0],
                                     directory=data_dir)

    results = data_utility.load_results(directory=results_dir)
    results['cardinality'] = results['constraint_name'].str.extract('_k([0-9]+)$').astype(int)
    results['constraint_name'] = results['constraint_name'].str.replace('_k[0-9]+$', '', regex=True)
    results['selected'] = results['selected'].apply(ast.literal_eval)  # make list string a proper list

    print('\n-------- 5.2 Experimental Design --------')

    print('\n------ 5.2.1 Scenario and Dataset ------')

    print('\n## Table 5.1: Feature overview ##\n')
    feature_overview = pd.DataFrame({'Feature': X.columns})
    aggregates = '(max|min|median|std|sum)'
    feature_overview['Quantity'] = feature_overview['Feature'].str.replace(
        f'_(([0-9]+)|{aggregates})$', '')
    feature_overview['Slip_System'] = feature_overview['Feature'].str.extract(
        '_([0-9]+)$', expand=False)
    feature_overview['Aggregate'] = feature_overview['Feature'].str.extract(
        f'_{aggregates}$', expand=False)
    feature_overview = feature_overview.groupby('Quantity').agg(
        Slip_systems=('Slip_System', 'nunique'),
        Aggregates=('Aggregate', 'nunique'),
        Features=('Quantity', 'size')).reset_index()
    feature_overview.sort_values(by='Quantity', inplace=True)
    print(feature_overview.to_latex(index=False))

    print('\n-------- 5.3 Evaluation --------')

    print('\n------ 5.3.1 Solution Quality ------')

    prediction_data = evaluation_utility.reshape_prediction_data(results, additional_columns=['cardinality'])
    prediction_data = evaluation_utility.rename_for_plots(prediction_data, is_dissertation=True)

    # Figure 5.1a
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', palette='Paired',
                data=prediction_data.drop(columns='Cardinality'))
    plt.xticks(rotation=20)
    plt.ylim(0.8, 1.01)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-prediction-performance-split.pdf')

    # Figure 5.1b
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Cardinality', palette='Paired',
                data=prediction_data[prediction_data['Split'] == 'Test'].drop(columns='Split'))
    plt.xticks(rotation=20)
    plt.ylim(0.8, 1.01)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-prediction-performance-cardinality.pdf')

    print('Objective value aggregated over constraint types:')
    print(results.groupby('cardinality')['objective_value'].describe())
    print('Objective value aggregated over constraint types, excluding "Mixed":')
    print(results[results['constraint_name'] != 'Mixed'].groupby('cardinality')['objective_value'].describe())

    # For comparison: box plot
    # plt.figure(figsize=(4, 3))
    # sns.boxplot(x='cardinality', y='objective_value', data=results)
    # plt.xticks(rotation=20)
    # plt.ylim(0, 10)
    # plt.tight_layout()

    # For comparison: bar chart
    # plt.figure(figsize=(8, 3))
    # sns.barplot(x='constraint_name', hue='cardinality', y='objective_value', data=results)
    # plt.xticks(rotation=20)
    # plt.ylim(0, 10)
    # plt.tight_layout()

    # For comparison: distribution of feature qualities
    max_train_time = X['time'].quantile(q=0.8)  # split from ms_pipeline.py
    X_train = X[X['time'] <= max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time', 'step'])
    y_train = y[X['time'] <= max_train_time]
    qualities = pd.Series(feature_qualities.abs_corr(X=X_train, y=y_train), index=X_train.columns)
    # Make sure we have the correct qualities by computing one objective value with them:
    assert results.loc[0, 'objective_value'] == qualities[results.loc[0, 'selected']].sum()
    print('Distribution of feature qualities:')
    print(qualities.describe())
    print(f'Fraction of feature qualities >= 0.8: {(qualities >= 0.8).sum() / len(qualities):.2}')

    print('\n------ 5.3.2 Selected Features ------')

    # Test whether always the maximum number of possible features is selected:
    print('Always maximum number of features selected? ' +
          str((results['num_selected'] == results['cardinality']).all()))

    # Prepare Figure 5.2:
    # Make sure results have the cardinalities assumed below in the order assumed below:
    assert (([5] * 12 + [10] * 12) == results['cardinality']).all()
    # Compute similarity between feature sets:
    similarity_matrix = np.zeros(shape=(len(results), len(results)))
    for i in range(len(results)):
        for j in range(len(results)):
            similarity_matrix[i, j] = len(set(results['selected'].iloc[i]).intersection(results['selected'].iloc[j]))
    # Use names from dissertation:
    constraint_names = [f'(D{i+1})' for i in range(results['constraint_name'].nunique())] * results['cardinality'].nunique()
    similarity_matrix = pd.DataFrame(similarity_matrix, index=constraint_names, columns=constraint_names)

    # Figure 5.2a
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 13
    sns.heatmap(similarity_matrix.iloc[:12, :12], vmin=0, vmax=5, cmap='YlGnBu',
                annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-selected-similarity-card5.pdf')

    # Figure 5.2b
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 13
    sns.heatmap(similarity_matrix.iloc[12:, 12:], vmin=0, vmax=10, cmap='YlGnBu',
                annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-selected-similarity-card10.pdf')

    # Compute drop in objective value if adding the hardest combination of constraints
    print('Maximum drop in objective value over constraint types:')
    grouping = results.groupby('cardinality')
    print(1 - grouping['objective_value'].min() / grouping['objective_value'].max())


# Parse some command line arguments and run evaluation.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates the dissertation\'s plots and prints statistics to evaluate ' +
        'the case study in materials science.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/ms/', dest='data_dir',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/ms-results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/ms-plots/',
                        dest='plot_dir', help='Output directory for plots.')
    args = parser.parse_args()
    print('Evaluation started.')
    evaluate(**vars(args))
    print('Plots created and saved.')
