"""Evaluation of the study with synthetic constraints for dissertation

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

    # Remove a few faulty results (0.0036 % of rows) -- in small fraction of infeasible scenarios,
    # no features are selected (correct!) but solver still returns a positive objective value
    results = results[~((results['num_selected'] == 0) & (results['objective_value'] > 0))]

    # Set normalized prediction performance for empty feature sets to 0 (consistent with objective)
    NORM_PRED_METRICS = [x for x in results.columns if x.endswith('_r2') and x.startswith('frac_')]
    results.loc[results['num_selected'] == 0, NORM_PRED_METRICS] = 0

    # Prepare sub-lists of evaluation metrics for certain plots
    ORIGINAL_PRED_METRICS = [x for x in results.columns
                             if x.endswith('_r2') and not x.startswith('frac_')]
    EVALUATION_METRICS = ['num_constraints', 'frac_constrained_variables',
                          'frac_unique_constrained_variables', 'frac_solutions', 'frac_selected',
                          'frac_objective', 'frac_linear-regression_test_r2',
                          'frac_xgb-tree_test_r2']

    print('\n-------- 4.3 Experimental Design --------')

    print('\n------ 4.3.6 Datasets ------')

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

    print('\n------ 4.4.1 Comparison of Prediction Models ------')

    # Figure 4.1a: Prediction performance by prediction model, all experimental runs
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

    # Figure 4.1b: Prediction performance by prediction model, unconstrained experimental runs
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

    print('\n------ 4.4.2 Relationship Between Evaluation Metrics ------')

    # Figure 4.2: Correlation between evaluation metrics
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 13
    sns.heatmap(data=evaluation_utility.rename_for_diss_plots(results[EVALUATION_METRICS]).corr(
        method='spearman'), vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-evaluation-metrics-correlation.pdf')

    # Figure 4.3a: Fraction of selected features vs. objective value
    scatter_plot_data = results.sample(n=1000, random_state=25)
    scatter_plot_data = evaluation_utility.rename_for_diss_plots(scatter_plot_data,
                                                                 long_metric_names=True)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Fraction of selected features $\\mathit{frac}_{\\mathrm{se}}$',
                    y='Objective value $Q_{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlabel('Fraction of selected features $\\mathit{frac}_{\\mathrm{se}}$', x=0.4)  # move
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-selected-vs-objective.pdf')

    # Figure 4.3b: Prediction performance vs. objective value
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Prediction $R^{2, \\mathrm{lin}}_{\\mathrm{norm}}$',
                    y='Objective value $Q_{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-frac-linear-regression-r2-vs-objective.pdf')

    # Figure 4.3c: Fraction of solutions vs. objective value
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.scatterplot(x='Fraction of solutions $\\mathit{frac}_{\\mathrm{so}}$',
                    y='Objective value $Q_{\\mathrm{norm}}$',
                    data=scatter_plot_data, color=DEFAULT_COL_SINGLE, s=8)
    plt.xlim(-0.1, 1.1)
    plt.xticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-solutions-vs-objective.pdf')

    # Figure 4.3d: Number of constraints vs. objective value
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 15
    sns.boxplot(x='Number of constraints $\\mathit{num}_{\\mathrm{co}}$',
                y='Objective value $Q_{\\mathrm{norm}}$',
                data=scatter_plot_data, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraints-vs-objective.pdf')

    print('\n------ 4.4.3 Impact of Constraint Types ------')

    # Figure 4.4a: Fraction of solutions by constraint type
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Constraint type', y='$\\mathit{frac}_{\\mathrm{so}}$',
                data=evaluation_utility.rename_for_diss_plots(results), fliersize=0, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraint-type-vs-solutions.pdf')

    # Figure 4.4b: Objective value by constraint type
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 18
    sns.boxplot(x='Constraint type', y='$Q_{\\mathrm{norm}}$',
                data=evaluation_utility.rename_for_diss_plots(results), fliersize=0, color='black',
                boxprops={'facecolor': DEFAULT_COL_SINGLE})
    plt.xticks(rotation=90)
    plt.ylim(-0.1, 1.1)
    plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraint-type-vs-objective.pdf')

    # For comparison: prediction performance
    # plt.figure(figsize=(5, 5))
    # plt.rcParams['font.size'] = 18
    # sns.boxplot(x='Constraint type', y='$R^{2, \\mathrm{lin}}_{\\mathrm{norm}}$',
    #             data=evaluation_utility.rename_for_diss_plots(results), fliersize=0,
    #             color='black', boxprops={'facecolor': DEFAULT_COL_SINGLE})
    # plt.xticks(rotation=90)
    # plt.ylim(-0.1, 1.1)
    # plt.yticks(np.arange(start=0, stop=1.1, step=0.2))
    # plt.tight_layout()


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
    print('\nPlots created and saved.')
