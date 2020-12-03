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
plt.figure(figsize=(4, 3))
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data, fliersize=0)
plt.xticks(rotation=20)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-prediction-performance-all.pdf')
prediction_data = results.loc[results['constraint_name'] == 'UNCONSTRAINED', ORIGINAL_PRED_METRICS]
prediction_data = evaluation_utility.reshape_prediction_data(prediction_data)
plt.figure(figsize=(4, 3))
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data, fliersize=0)
plt.xticks(rotation=20)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-prediction-performance-unconstrained.pdf')

# ---(Q2.1) Relationship between constraint evaluation metrics---

# Figure 2
plt.figure(figsize=(6, 6))
sns.heatmap(data=results[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True, cbar=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-evaluation-metrics-correlation.pdf')

# Figure 3
scatter_plot_data = results.sample(n=1000, random_state=25)
plt.figure(figsize=(4, 3))
scatter_plot_data.plot.scatter(x='frac_selected', y='frac_objective', s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-selected-vs-objective.pdf')
plt.figure(figsize=(4, 3))
scatter_plot_data.plot.scatter(x='frac_solutions', y='frac_objective', s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-solutions-vs-objective.pdf')
plt.figure(figsize=(4, 3))
sns.boxplot(x='frac_constraints', y='frac_objective', data=scatter_plot_data)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-constraints-vs-objective.pdf')
plt.figure(figsize=(4, 3))
scatter_plot_data.plot.scatter(x='frac_linear-regression_test_r2', y='frac_objective', s=1, xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-frac-linear-regression-r2-vs-objective.pdf')

# ---(Q2.2) Comparison of constraint types---

# Figure 4
plt.figure(figsize=(4, 3))
sns.boxplot(x='constraint_name', y='frac_solutions', data=results, fliersize=0)
plt.xticks(rotation=60)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-constraint-type-vs-solutions.pdf')
plt.figure(figsize=(4, 3))
sns.boxplot(x='constraint_name', y='frac_objective', data=results, fliersize=0)
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
agg_data = pd.melt(agg_data, var_name='evaluation metric', value_name='mean per dataset')
plt.figure(figsize=(4, 3))
sns.boxplot(x='evaluation metric', y='mean per dataset', data=agg_data)
plt.xticks(rotation=30)
plt.ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'syn-evaluation-metrics-mean-per-dataset.pdf')
plt.figure(figsize=(4, 3))
sns.boxplot(x='dataset_name', y='frac_objective', data=results)
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
