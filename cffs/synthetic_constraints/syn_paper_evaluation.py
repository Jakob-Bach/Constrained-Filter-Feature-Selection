"""Paper evaluation for synthetic constraints

Script to compute all summary statistics and create all plots used in the paper.
"""


import os
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cffs.utilities import data_utility
from cffs.utilities import evaluation_utility


RESULTS_PATH = pathlib.Path('data/openml-results/')
PLOT_PATH = '../paper-cffs-text/plots/'
plt.rcParams['font.family'] = 'Linux Biolinum'

results = data_utility.load_results(directory=RESULTS_PATH)
results = results[results['quality_name'] == 'abs_corr']  # results for MI very similar
evaluation_utility.add_normalized_objective(results)
evaluation_utility.add_normalized_variable_counts(results)
evaluation_utility.add_normalized_prediction_performance(results)
evaluation_utility.add_normalized_num_constraints(results)
os.makedirs(PLOT_PATH, exist_ok=True)

ORIGINAL_PRED_METRICS = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]
EVALUATION_METRICS = ['frac_constraints', 'frac_constrained_variables', 'frac_unique_constrained_variables',
                      'frac_solutions', 'frac_selected', 'frac_objective',
                      'frac_linear-regression_test_r2', 'frac_xgb-tree_test_r2']

# ---Comparison of prediction models---

# Figure 1
prediction_data = evaluation_utility.reshape_prediction_data(results[ORIGINAL_PRED_METRICS])
prediction_data = evaluation_utility.rename_for_plots(prediction_data)
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data, palette='Paired', fliersize=0)
plt.xticks(rotation=20)
plt.ylim(-0.1, 1.1)
plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-prediction-performance-all.pdf')
prediction_data = results.loc[results['constraint_name'] == 'UNCONSTRAINED', ORIGINAL_PRED_METRICS]
prediction_data = evaluation_utility.reshape_prediction_data(prediction_data)
prediction_data = evaluation_utility.rename_for_plots(prediction_data)
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Prediction model', y='$R^2$', hue='Split', data=prediction_data, palette='Paired', fliersize=0)
plt.xticks(rotation=20)
plt.ylim(-0.1, 1.1)
plt.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=2, borderpad=0, edgecolor='white')
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-prediction-performance-unconstrained.pdf')

# ---(Q2.1) Relationship between constraint evaluation metrics---

# Figure 2
plt.figure(figsize=(5, 5))
plt.rcParams['font.size'] = 14
sns.heatmap(data=evaluation_utility.rename_for_plots(results[EVALUATION_METRICS]).corr(method='spearman'),
            vmin=-1, vmax=1, cmap='PRGn', annot=True, square=True, cbar=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-evaluation-metrics-correlation.pdf')

# Figure 3
scatter_plot_data = results.sample(n=1000, random_state=25)
scatter_plot_data = evaluation_utility.rename_for_plots(scatter_plot_data, long_metric_names=True)
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 20
scatter_plot_data.plot.scatter(x='Number of selected features $n_{se}^{norm}$', y='Objective value $Q^{norm}$',
                               s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-selected-vs-objective.pdf')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 20
scatter_plot_data.plot.scatter(x='Number of solutions $n_{so}^{norm}$', y='Objective value $Q^{norm}$',
                               s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-solutions-vs-objective.pdf')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Number of constraints $n_{co}^{norm}$', y='Objective value $Q^{norm}$',
            data=scatter_plot_data, color='black', boxprops={'facecolor': plt.get_cmap('Paired')(0)})
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-constraints-vs-objective.pdf')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 20
scatter_plot_data.plot.scatter(x='Prediction $R^{2, norm}_{lreg}$', y='Objective value $Q^{norm}$',
                               s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-frac-linear-regression-r2-vs-objective.pdf')

# ---(Q2.2) Comparison of constraint types---

# Figure 4
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Constraint type', y='$n_{so}^{norm}$', fliersize=0, color='black',
            data=evaluation_utility.rename_for_plots(results),
            boxprops={'facecolor': plt.get_cmap('Paired')(0)})
plt.xticks(rotation=60)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-constraint-type-vs-solutions.pdf')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Constraint type', y='$Q^{norm}$', fliersize=0, color='black',
            data=evaluation_utility.rename_for_plots(results),
            boxprops={'facecolor': plt.get_cmap('Paired')(0)})
plt.xticks(rotation=60)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-constraint-type-vs-objective.pdf')

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

# ---(Q2.3) Comparison of datasets---

# Figure 5
agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
agg_data = agg_data.drop(columns='frac_constrained_variables')  # not in [0,1]
agg_data = evaluation_utility.rename_for_plots(agg_data)
agg_data = pd.melt(agg_data, var_name='Evaluation metric', value_name='Mean per dataset')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Evaluation metric', y='Mean per dataset', data=agg_data,
            color='black', boxprops={'facecolor': plt.get_cmap('Paired')(0)})
plt.xticks(rotation=30)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-evaluation-metrics-mean-per-dataset.pdf')
plt.figure(figsize=(4, 3))
plt.rcParams['font.size'] = 14
sns.boxplot(x='Dataset name', y='Objective value $Q^{norm}$', color='black',
            data=evaluation_utility.rename_for_plots(results[['dataset_name', 'frac_objective']], long_metric_names=True),
            boxprops={'facecolor': plt.get_cmap('Paired')(0)})
plt.xticks([])
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-objective-value-per-dataset.pdf')

# For comparison: without aggregation (constraint types and generation mechanism not averaged out)
# for evaluation_metric in EVALUATION_METRICS:
#     sns.boxplot(x='dataset_name', y=evaluation_metric, data=results)
#     plt.xticks(rotation=45)
#     plt.ylim(-0.1, 1.1)
#     plt.show()
