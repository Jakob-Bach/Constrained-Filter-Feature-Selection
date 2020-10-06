"""Synthetic constraints evaluation

Script to generate the plots for evaluating our experiments with synthetic constraints.
"""


import pathlib

import matplotlib.pyplot as plt
import seaborn as sns

from cffs.utilities import data_utility
from cffs.evaluation import evaluation_utility


results = data_utility.load_results(directory=pathlib.Path('data/openml-results/'))
evaluation_utility.add_normalized_objective(results)
evaluation_utility.add_normalized_num_constraints(results)

CONSTRAINT_METRICS = ['frac_objective', 'frac_selected', 'frac_constraints', 'frac_solutions']
PREDICTION_METRICS = [x for x in results.columns if x.endswith('_r2')]
EVALUATION_METRICS = CONSTRAINT_METRICS + ['linear-regression_test_r2', 'xgb-tree_test_r2']

# ---Distribution of constraint evaluation metrics, comparing constraint types---

for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x='constraint_name', y=evaluation_metric, data=results)
    plt.xticks(rotation=45)
    plt.ylim(-0.1, 1.1)
    plt.show()

# ---Relationship between constraint evaluation metrics---

for i in range(len(EVALUATION_METRICS) - 1):
    for j in range(i + 1, len(EVALUATION_METRICS)):
        results.plot.scatter(x=EVALUATION_METRICS[i], y=EVALUATION_METRICS[j], s=1,
                             xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
        plt.show()
# All in one: sns.pairplot(results[EVALUATION_METRICS])

sns.heatmap(data=results[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)

# ---Performance of prediction models---

# also interesting for results.loc[results['constraint_name'] == 'UNCONSTRAINED']
prediction_data = evaluation_utility.reshape_prediction_data(results)
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data)
plt.ylim(-0.1, 1.1)
plt.show()

# ---Distribution of constraint evaluation metrics, comparing datasets---

for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x='dataset_name', y=evaluation_metric, data=results)
    plt.xticks(rotation=45)
    plt.ylim(-0.1, 1.1)
    plt.show()

agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x=evaluation_metric, data=agg_data)
    plt.xlim(-0.1, 1.1)
    plt.show()

# ---Relationship between constraint evaluation metrics, aggregated per datasets---

agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
for i in range(len(EVALUATION_METRICS) - 1):
    for j in range(i + 1, len(EVALUATION_METRICS)):
        agg_data.plot.scatter(x=EVALUATION_METRICS[i], y=EVALUATION_METRICS[j],
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
        plt.show()

sns.heatmap(data=agg_data[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)

# ---Distribution of constraint evaluation metrics, comparing constraint types and datasets---

agg_data = results.groupby(['constraint_name', 'dataset_name'])[EVALUATION_METRICS].mean().reset_index()
for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x='constraint_name', y=evaluation_metric, data=agg_data)
    plt.xticks(rotation=45)
    plt.ylim(-0.1, 1.1)
    plt.show()

for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x='dataset_name', y=evaluation_metric, data=agg_data)
    plt.xticks(rotation=45)
    plt.ylim(-0.1, 1.1)
    plt.show()

# ---Relationship between constraint evaluation metrics, aggregated per constraint types and dataset---

agg_data = results.groupby(['constraint_name', 'dataset_name'])[EVALUATION_METRICS].mean()
for i in range(len(EVALUATION_METRICS) - 1):
    for j in range(i + 1, len(EVALUATION_METRICS)):
        agg_data.plot.scatter(x=EVALUATION_METRICS[i], y=EVALUATION_METRICS[j],
                              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))
        plt.show()

sns.heatmap(data=agg_data[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)
