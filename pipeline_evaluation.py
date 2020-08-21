"""Synthetic constraints evaluation

Script to generate the plots for evaluating our experiments with synthetic constraints.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results = pd.read_csv('data/openml-results/results.csv')
CONSTRAINT_METRICS = ['objective_value', 'num_selected', 'num_constraints', 'frac_solutions']
PREDICTTION_METRICS = [x for x in results.columns if x.endswith('_r2')]

# ---Distribution of constraint evaluation metrics, comparing constraint types---

for constraint_metric in CONSTRAINT_METRICS + ['xgb-tree_test_r2']:
    sns.boxplot(x='constraint_name', y=constraint_metric, data=results)
    plt.xticks(rotation=45)
    if constraint_metric.endswith('_r2'):
        plt.ylim(-0.1, 1.1)
    plt.show()

# ---Relationship between constraint evaluation metrics---

evaluation_metrics = CONSTRAINT_METRICS + ['xgb-tree_test_r2']
for i in range(len(evaluation_metrics) - 1):
    for j in range(i + 1, len(evaluation_metrics)):
        results.plot.scatter(x=evaluation_metrics[i], y=evaluation_metrics[j], s=1)
        if evaluation_metrics[j].endswith('_r2'):
            plt.ylim(-0.1, 1.1)
        plt.show()
# All in one: sns.pairplot(results[evaluation_metrics])

sns.heatmap(data=results[evaluation_metrics].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)
