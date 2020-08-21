"""Synthetic constraints evaluation

Script to generate the plots for evaluating our experiments with synthetic constraints.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results = pd.read_csv('data/openml-results/results.csv')
CONSTRAINT_METRICS = ['objective_value', 'num_selected', 'num_constraints', 'frac_solutions']
PREDICTION_METRICS = [x for x in results.columns if x.endswith('_r2')]
EVALUATION_METRICS = CONSTRAINT_METRICS + ['linear-regression_test_r2', 'xgb-tree_test_r2']

# ---Distribution of constraint evaluation metrics, comparing constraint types---

for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x='constraint_name', y=evaluation_metric, data=results)
    plt.xticks(rotation=45)
    if evaluation_metric.endswith('_r2'):
        plt.ylim(-0.1, 1.1)
    plt.show()

# ---Relationship between constraint evaluation metrics---

for i in range(len(EVALUATION_METRICS) - 1):
    for j in range(i + 1, len(EVALUATION_METRICS)):
        results.plot.scatter(x=EVALUATION_METRICS[i], y=EVALUATION_METRICS[j], s=1)
        if EVALUATION_METRICS[j].endswith('_r2'):
            plt.ylim(-0.1, 1.1)
        plt.show()
# All in one: sns.pairplot(results[EVALUATION_METRICS])

sns.heatmap(data=results[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)

# ---Performance of prediction models---

prediction_data = results[PREDICTION_METRICS].melt(var_name='model', value_name='r2')
prediction_data['split'] = prediction_data['model'].apply(lambda x: 'train' if 'train' in x else 'test')
prediction_data['model'] = prediction_data['model'].str.replace('_train_r2', '').str.replace('_test_r2', '')
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data)
plt.ylim(-0.1, 1.1)
plt.show()

# ---Relationship between constraint evaluation metrics and datasets---

dataset_agg_data = results.groupby('dataset_name')[EVALUATION_METRICS].mean()
for evaluation_metric in EVALUATION_METRICS:
    sns.boxplot(x=evaluation_metric, data=dataset_agg_data)
    if evaluation_metric.endswith('_r2'):
        plt.xlim(-0.1, 1.1)
    plt.show()

for i in range(len(EVALUATION_METRICS) - 1):
    for j in range(i + 1, len(EVALUATION_METRICS)):
        dataset_agg_data.plot.scatter(x=EVALUATION_METRICS[i], y=EVALUATION_METRICS[j])
        if EVALUATION_METRICS[j].endswith('_r2'):
            plt.ylim(-0.1, 1.1)
        plt.show()

sns.heatmap(data=dataset_agg_data[EVALUATION_METRICS].corr(method='spearman'), vmin=-1, vmax=1,
            cmap='RdYlGn', annot=True, square=True)
