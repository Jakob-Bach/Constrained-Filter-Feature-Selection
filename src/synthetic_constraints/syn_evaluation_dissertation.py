"""Evaluation of the study with synthetic constraints

Script to compute all summary statistics and create all plots used in the dissertation to evaluate
the study with synthetic constraints. Should be run after the experimental pipeline.

Usage: python -m synthetic_constraints.syn_evaluation_dissertation --help
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utilities import data_utility
from utilities import evaluation_utility


plt.rcParams['font.family'] = 'Arial'
DEFAULT_COL_PALETTE = 'YlGnBu'
DEFAULT_COL_SINGLE = sns.color_palette(DEFAULT_COL_PALETTE, 2)[1]


# Create and save all plots to evaluate the study with synthetic constraints for the dissertation.
# To that end, read results from the "results_dir" and some dataset information from the
# "data_dir"; save plots to the "plot_dir" and print some statistics to the console.
def evaluate(data_dir: pathlib.Path, results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    # Load results and add normalized versions of evaluation metrics
    results = data_utility.load_results(directory=results_dir)
    results = results[results['quality_name'] == 'mut_info']  # results for "abs_corr" very similar
    evaluation_utility.add_normalized_objective(results)
    evaluation_utility.add_normalized_variable_counts(results)
    evaluation_utility.add_normalized_prediction_performance(results)
    evaluation_utility.add_normalized_num_constraints(results)

    # Prepare sub-lists of evaluation metrics for certain plots
    ORIGINAL_PRED_METRICS = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]
    EVALUATION_METRICS = ['frac_constraints', 'frac_constrained_variables', 'frac_unique_constrained_variables',
                          'frac_solutions', 'frac_selected', 'frac_objective',
                          'frac_linear-regression_test_r2', 'frac_xgb-tree_test_r2']

    print('\n-------- 4.3 Experimental Design --------')

    print('\n------ 4.3.4 Datasets ------')

    print('\n## Table 4.1: Dataset overview ##\n')
    dataset_overview = pd.read_csv(data_dir / '_data_overview.csv')
    dataset_overview = dataset_overview[['name', 'NumberOfInstances', 'NumberOfNumericFeatures']]
    dataset_overview.rename(columns={'name': 'Dataset', 'NumberOfInstances': 'm',
                                     'NumberOfNumericFeatures': 'n'}, inplace=True)
    dataset_overview[['m', 'n']] = dataset_overview[['m', 'n']].astype(int)
    dataset_overview['n'] = dataset_overview['n'] - 1  # exclude prediction target
    dataset_overview.sort_values(by='Dataset', key=lambda x: x.str.lower(), inplace=True)
    print(dataset_overview.to_latex(index=False))

    print('\n-------- 4.4 Evaluation --------')

    print('\n------ 4.4.1 Comparison of Prediction Performance ------')

    # Figure 4.1a
    prediction_data = evaluation_utility.reshape_prediction_data(results[ORIGINAL_PRED_METRICS])
    prediction_data = evaluation_utility.rename_for_diss_plots(prediction_data)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data,
                palette=DEFAULT_COL_PALETTE, fliersize=0)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    leg = plt.legend(title='Split', loc='upper left', bbox_to_anchor=(0, -0.1), ncol=2,
                     columnspacing=1, edgecolor='white', framealpha=0)
    leg.get_title().set_position((-110, -21))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-prediction-performance-all.pdf')

    # Figure 4.1b
    prediction_data = results.loc[results['constraint_name'] == 'UNCONSTRAINED', ORIGINAL_PRED_METRICS]
    prediction_data = evaluation_utility.reshape_prediction_data(prediction_data)
    prediction_data = evaluation_utility.rename_for_diss_plots(prediction_data)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data,
                palette=DEFAULT_COL_PALETTE, fliersize=0)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    leg = plt.legend(title='Split', loc='upper left', bbox_to_anchor=(0, -0.1), ncol=2,
                     columnspacing=1, edgecolor='white', framealpha=0)
    leg.get_title().set_position((-110, -21))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-prediction-performance-unconstrained.pdf')

    print('\n------ 4.4.2 Relationship Between Evaluation Metrics (Q1) ------')

    # Figure 4.2
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 13
    sns.heatmap(data=evaluation_utility.rename_for_diss_plots(results[EVALUATION_METRICS]).corr(
        method='spearman'), vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-evaluation-metrics-correlation.pdf')

    # Figure 4.3a
    scatter_plot_data = results.sample(n=1000, random_state=25)
    scatter_plot_data = evaluation_utility.rename_for_diss_plots(scatter_plot_data,
                                                                 long_metric_names=True)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Number of selected features $n_{\\mathrm{se}}^{\\mathrm{norm}}$',
                    y='Objective value $Q^{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlabel('Number of selected features $n_{\\mathrm{se}}^{\\mathrm{norm}}$', x=0.4)  # move
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-selected-vs-objective.pdf')

    # Figure 4.3b
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Number of solutions $n_{\\mathrm{so}}^{\\mathrm{norm}}$',
                    y='Objective value $Q^{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-solutions-vs-objective.pdf')

    # Figure 4.3c
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Number of constraints $n_{\\mathrm{co}}$',
                y='Objective value $Q^{\\mathrm{norm}}$',
                data=scatter_plot_data, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraints-vs-objective.pdf')

    # Figure 4.3d
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Prediction $R^{2, \\mathrm{norm}}_{\\mathrm{lin}}$',
                    y='Objective value $Q^{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-frac-linear-regression-r2-vs-objective.pdf')

    print('\n------ 4.4.3 Comparison of Constraint Types (Q2) ------')

    # Figure 4.4a
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Constraint type', y='$n_{\\mathrm{so}}^{\\mathrm{norm}}$',
                data=evaluation_utility.rename_for_diss_plots(results), fliersize=0, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraint-type-vs-solutions.pdf')

    # Figure 4.4b
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Constraint type', y='$Q^{\\mathrm{norm}}$',
                data=evaluation_utility.rename_for_diss_plots(results), fliersize=0, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraint-type-vs-objective.pdf')

    # For comparison: average out repetitions, only show variation between datasets
    # agg_data = results.groupby(['constraint_name', 'dataset_name'])[EVALUATION_METRICS].mean().reset_index()
    # plt.figure(figsize=(4, 3))
    # sns.boxplot(x='constraint_name', y='frac_objective', data=agg_data, fliersize=0)
    # plt.xticks(rotation=60)
    # plt.ylim(-0.1, 1.1)
    # plt.tight_layout()

    # For comparison: prediction performance
    # plt.figure(figsize=(4, 3))
    # sns.boxplot(x='constraint_name', y='frac_linear-regression_test_r2', data=results.replace(float('nan'), 0))
    # plt.xticks(rotation=60)
    # plt.ylim(-0.1, 1.1)
    # plt.tight_layout()
    # plt.figure(figsize=(4, 3))
    # sns.boxplot(x='constraint_name', y='frac_xgb-tree_test_r2', data=results.replace(float('nan'), 0))
    # plt.xticks(rotation=60)
    # plt.ylim(-0.1, 1.1)
    # plt.tight_layout()

    print('\n------ 4.4.4 Comparison of Datasets (Q3) ------')

    # Figure 4.5a
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Dataset', y='Objective value $Q^{\\mathrm{norm}}$', color='black',
                data=evaluation_utility.rename_for_diss_plots(
                    results[['dataset_name', 'frac_objective']], long_metric_names=True),
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks([])
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-objective-value-per-dataset.pdf')

    # Figure 4.5b
    agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
    agg_data = agg_data.drop(columns='frac_constrained_variables')  # not in [0,1]
    agg_data = evaluation_utility.rename_for_diss_plots(agg_data)
    agg_data = pd.melt(agg_data, var_name='Evaluation metric', value_name='Mean per dataset')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Evaluation metric', y='Mean per dataset', data=agg_data,
                color='black', boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-evaluation-metrics-mean-per-dataset.pdf')

    # For comparison: without aggregation (constraint types and generation mechanism not averaged out)
    # for evaluation_metric in EVALUATION_METRICS:
    #     sns.boxplot(x='dataset_name', y=evaluation_metric, data=results)
    #     plt.xticks(rotation=45)
    #     plt.ylim(-0.1, 1.1)
    #     plt.show()


# Parse some command line argument and run evaluation.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates the dissertation\'s plots and prints statistics to evaluate ' +
        'the study with synthetic constraints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/openml/', dest='data_dir',
                        help='Directory with prediction datasets in (X, y) form.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/openml-results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/openml-plots/',
                        dest='plot_dir', help='Output directory for plots.')
    args = parser.parse_args()
    print('Evaluation started.')
    evaluate(**vars(args))
    print('Plots created and saved.')
