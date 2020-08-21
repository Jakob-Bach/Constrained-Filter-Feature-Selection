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
