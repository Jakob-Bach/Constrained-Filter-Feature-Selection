"""Synthetic constraints evaluation

Script to generate the plots for evaluating our experiments with synthetic constraints.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results = pd.read_csv('data/openml-results/results.csv')
CONSTRAINT_METRICS = ['frac_objective', 'frac_selected', 'num_constraints', 'frac_solutions']
PREDICTION_METRICS = [x for x in results.columns if x.endswith('_r2')]
EVALUATION_METRICS = CONSTRAINT_METRICS + ['linear-regression_test_r2', 'xgb-tree_test_r2']

# Make objective value relative to dataset's max objective
max_objective_values = results.loc[results['constraint_name'] == 'UNCONSTRAINED',
                                   ['quality_name', 'split_idx', 'dataset_name', 'objective_value']]
assert len(max_objective_values) == results.groupby(['quality_name', 'split_idx', 'dataset_name']).ngroups
max_objective_values.rename(columns={'objective_value': 'max_objective'}, inplace=True)
results = results.merge(max_objective_values)
results['frac_objective'] = results['objective_value'] / results['max_objective']


def reshape_prediction_data(results_data: pd.DataFrame) -> pd.DataFrame:
    results_data = results_data[PREDICTION_METRICS].melt(var_name='model', value_name='r2')
    results_data['split'] = results_data['model'].apply(lambda x: 'train' if 'train' in x else 'test')
    results_data['model'] = results_data['model'].str.replace('_train_r2', '').str.replace('_test_r2', '')
    return results_data


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

# also interesting for results.loc[results['constraint_name'] == 'UNCONSTRAINED']
sns.boxplot(x='model', y='r2', hue='split', data=reshape_prediction_data(results))
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
