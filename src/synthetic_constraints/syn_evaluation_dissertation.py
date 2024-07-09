"""Evaluation of the study with synthetic constraints

Script to compute all summary statistics and create all plots used in the dissertation to evaluate
the study with synthetic constraints. Should be run after the experimental pipeline.

Usage: python -m synthetic_constraints.syn_evaluation_dissertation --help
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utilities import data_utility
from utilities import evaluation_utility


plt.rcParams['font.family'] = 'Arial'


# Create and save all plots to evaluate the study with synthetic constraints for the dissertation.
# To that end, read a results file from the "results_dir" and save plots to the "plot_dir".
def evaluate(results_dir: pathlib.Path, plot_dir: pathlib.Path) -> None:
    if not plot_dir.is_dir():
        print('Plot directory does not exist. We create it.')
        plot_dir.mkdir(parents=True)
    if len(list(plot_dir.glob('*.pdf'))) > 0:
        print('Plot directory is not empty. Files might be overwritten, but not deleted.')

    # Load results and add normalized versions of evaluation metrics
    results = data_utility.load_results(directory=results_dir)
    results = results[results['quality_name'] == 'abs_corr']  # results for MI very similar
    evaluation_utility.add_normalized_objective(results)
    evaluation_utility.add_normalized_variable_counts(results)
    evaluation_utility.add_normalized_prediction_performance(results)
    evaluation_utility.add_normalized_num_constraints(results)

    # Prepare sub-lists of evaluation metrics for certain plots
    ORIGINAL_PRED_METRICS = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]
    EVALUATION_METRICS = ['frac_constraints', 'frac_constrained_variables', 'frac_unique_constrained_variables',
                          'frac_solutions', 'frac_selected', 'frac_objective',
                          'frac_linear-regression_test_r2', 'frac_xgb-tree_test_r2']

    # ---4.2.1 Comparison of Prediction Performance---

    # Figure 1a
    prediction_data = evaluation_utility.reshape_prediction_data(results[ORIGINAL_PRED_METRICS])
    prediction_data = evaluation_utility.rename_for_plots(prediction_data, is_dissertation=True)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data, palette='Paired', fliersize=0)
    plt.xticks(rotation=20)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-prediction-performance-all.pdf')

    # Figure 1b
    prediction_data = results.loc[results['constraint_name'] == 'UNCONSTRAINED', ORIGINAL_PRED_METRICS]
    prediction_data = evaluation_utility.reshape_prediction_data(prediction_data)
    prediction_data = evaluation_utility.rename_for_plots(prediction_data, is_dissertation=True)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data, palette='Paired', fliersize=0)
    plt.xticks(rotation=20)
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-prediction-performance-unconstrained.pdf')

    # ---4.2.2 Relationship Between Evaluation Metrics (Q1)---

    # Figure 2
    plt.figure(figsize=(5, 5))
    plt.rcParams['font.size'] = 13
    sns.heatmap(data=evaluation_utility.rename_for_plots(
        results[EVALUATION_METRICS], is_dissertation=True).corr(method='spearman'),
        vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-evaluation-metrics-correlation.pdf')

    # Figure 3a
    scatter_plot_data = results.sample(n=1000, random_state=25)
    scatter_plot_data = evaluation_utility.rename_for_plots(
        scatter_plot_data, long_metric_names=True, is_dissertation=True)
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 18
    scatter_plot_data.plot.scatter(
        x='Number of selected features $n_{\\mathrm{se}}^{\\mathrm{norm}}$',
        y='Objective value $Q^{\\mathrm{norm}}$',
        s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-selected-vs-objective.pdf')

    # Figure 3b
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 18
    scatter_plot_data.plot.scatter(
        x='Number of solutions $n_{\\mathrm{so}}^{\\mathrm{norm}}$',
        y='Objective value $Q^{\\mathrm{norm}}$',
        s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-solutions-vs-objective.pdf')

    # Figure 3c
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Number of constraints $n_{\\mathrm{co}}^{\\mathrm{norm}}$',
                y='Objective value $Q^{\\mathrm{norm}}$',
                data=scatter_plot_data, color='black', boxprops={'facecolor': plt.get_cmap('Paired')(0)})
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraints-vs-objective.pdf')

    # Figure 3d
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 18
    scatter_plot_data.plot.scatter(x='Prediction $R^{2, \\mathrm{norm}}_{\\mathrm{lreg}}$',
                                   y='Objective value $Q^{\\mathrm{norm}}$',
                                   s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-frac-linear-regression-r2-vs-objective.pdf')

    # ---4.2.3 Comparison of Constraint Types (Q2)---

    # Figure 4a
    plt.figure(figsize=(4, 4))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Constraint type', y='$n_{\\mathrm{so}}^{\\mathrm{norm}}$', fliersize=0, color='black',
                data=evaluation_utility.rename_for_plots(results, is_dissertation=True),
                boxprops={'facecolor': plt.get_cmap('Paired')(0)})
    plt.xticks(rotation=70)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-constraint-type-vs-solutions.pdf')

    # Figure 4b
    plt.figure(figsize=(4, 4))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Constraint type', y='$Q^{\\mathrm{norm}}$', fliersize=0, color='black',
                data=evaluation_utility.rename_for_plots(results, is_dissertation=True),
                boxprops={'facecolor': plt.get_cmap('Paired')(0)})
    plt.xticks(rotation=70)
    plt.ylim(-0.1, 1.1)
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

    # ---4.2.4 Comparison of Datasets (Q3)---

    # Figure 5a
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Dataset name', y='Objective value $Q^{\\mathrm{norm}}$', color='black',
                data=evaluation_utility.rename_for_plots(
                    results[['dataset_name', 'frac_objective']], long_metric_names=True,
                    is_dissertation=True),
                boxprops={'facecolor': plt.get_cmap('Paired')(0)})
    plt.xticks([])
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(plot_dir / 'syn-objective-value-per-dataset.pdf')

    # Figure 5b
    agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
    agg_data = agg_data.drop(columns='frac_constrained_variables')  # not in [0,1]
    agg_data = evaluation_utility.rename_for_plots(agg_data, is_dissertation=True)
    agg_data = pd.melt(agg_data, var_name='Evaluation metric', value_name='Mean per dataset')
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 11
    sns.boxplot(x='Evaluation metric', y='Mean per dataset', data=agg_data,
                color='black', boxprops={'facecolor': plt.get_cmap('Paired')(0)})
    plt.xticks(rotation=30)
    plt.ylim(-0.1, 1.1)
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
        description='Creates the dissertation\'s plots to evaluate the study with synthetic constraints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/openml-results/',
                        dest='results_dir', help='Directory with experimental results.')
    parser.add_argument('-p', '--plots', type=pathlib.Path, default='data/openml-plots/',
                        dest='plot_dir', help='Output directory for plots.')
    args = parser.parse_args()
    print('Evaluation started.')
    evaluate(**vars(args))
    print('Plots created and saved.')
