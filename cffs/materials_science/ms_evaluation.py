"""Evaluation of the case study in materials science

Script to compute all summary statistics and create all plots used in the paper to evaluate the
case study in materials science. Should be run after the experimental pipeline.

Usage: python -m cffs.materials_science.ms_evaluation --help
"""

import argparse
import ast
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cffs.utilities import data_utility
from cffs.utilities import evaluation_utility
from cffs.utilities import feature_qualities


plt.rcParams['font.family'] = 'Linux Biolinum'


# Create and save plots for the paper. Also, print some statistics that are used in the paper as well.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    results = data_utility.load_results(directory=results_dir)
    results['cardinality'] = results['constraint_name'].str.extract('_k([0-9]+)$').astype(int)
    results['constraint_name'] = results['constraint_name'].str.replace('_k[0-9]+$', '', regex=True)
    results['selected'] = results['selected'].apply(ast.literal_eval)  # make list string a list

    # ---Prediction performance---

    # Figure 6
    prediction_data = evaluation_utility.reshape_prediction_data(results, additional_columns=['cardinality'])
    prediction_data = evaluation_utility.rename_for_plots(prediction_data)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', palette='Paired',
                data=prediction_data.drop(columns='Cardinality'))
    plt.xticks(rotation=20)
    plt.ylim(0.8, 1.01)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-prediction-performance-split.pdf')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 14
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Cardinality', palette='Paired',
                data=prediction_data[prediction_data['Split'] == 'Test'].drop(columns='Split'))
    plt.xticks(rotation=20)
    plt.ylim(0.8, 1.01)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-prediction-performance-cardinality.pdf')

    # For comparison: Aggregate statistics
    # print(results[ORIGINAL_PRED_METRICS].describe().transpose()[['min', '50%', 'max']])
    # assert (results.loc[results['cardinality'] == 5, 'constraint_name'].reset_index(drop=True) ==
    #         results.loc[results['cardinality'] == 10, 'constraint_name'].reset_index(drop=True)).all()
    # print((results.loc[results['cardinality'] == 10, ORIGINAL_PRED_METRICS].reset_index(drop=True) -
    #        results.loc[results['cardinality'] == 5, ORIGINAL_PRED_METRICS].reset_index(drop=True)).
    #       describe().transpose()[['min', '50%', 'max']])

    # ---Objective value---

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
    X, y = data_utility.load_dataset(dataset_name=data_utility.list_datasets(directory=data_dir)[0],
                                     directory=data_dir)
    max_train_time = X['time'].quantile(q=0.8)  # split from ms_pipeline.py
    X_train = X[X['time'] <= max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time'])
    y_train = y[X['time'] <= max_train_time]
    qualities = pd.Series(feature_qualities.abs_corr(X=X_train, y=y_train), index=X_train.columns)
    # Make sure we have the correct qualities by computing one objective value with them:
    assert results.loc[0, 'objective_value'] == qualities[results.loc[0, 'selected']].sum()
    print('Distribution of feature qualities:')
    print(qualities.describe())
    print(f'Fraction of feature qualities >= 0.8: {(qualities >= 0.8).sum() / len(qualities):.2}')

    # ---Selected features---

    # Test whether always the maximum number of possible features is selected:
    print('Always maximum number of features selected? ' +
          str((results['num_selected'] == results['cardinality']).all()))

    # Prepare Figure 7:
    # Make sure results belonging to same cardinality follow each other (is assumed in naming below):
    assert (results['cardinality'].diff() != 0).sum() == results['cardinality'].nunique()
    # Compute similarity between feature sets:
    similarity_matrix = np.zeros(shape=(len(results), len(results)))
    for i in range(len(results)):
        for j in range(len(results)):
            similarity_matrix[i, j] = len(set(results['selected'].iloc[i]).intersection(results['selected'].iloc[j]))
    # Use names from paper:
    constraint_names = [f'(D{i+1})' for i in range(results['constraint_name'].nunique())] * results['cardinality'].nunique()
    similarity_matrix = pd.DataFrame(similarity_matrix, index=constraint_names, columns=constraint_names)

    # Figure 7
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 14
    sns.heatmap(similarity_matrix.iloc[:12, :12], vmin=0, vmax=5, cmap='YlGnBu',
                annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-selected-similarity-card5.pdf')
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 14
    sns.heatmap(similarity_matrix.iloc[12:, 12:], vmin=0, vmax=10, cmap='YlGnBu',
                annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'ms-selected-similarity-card10.pdf')

    # Compute drop in objective value if adding the hardest combination of constraints
    print('Maximum drop in objective value over constraint types:')
    grouping = results.groupby('cardinality')
    print(1 - grouping['objective_value'].min() / grouping['objective_value'].max())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates the paper\'s plots to evaluate the case study in materials science.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/ms/', dest='data_dir',
                        help='Directory with input data. Should contain datasets with two files each (X, y).')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/ms-results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='../paper-cffs-text/plots/',
                        dest='plot_dir', help='Output directory for plots.')
    args = parser.parse_args()
    evaluate(**vars(parser.parse_args()))
    print('Plots created and saved.')