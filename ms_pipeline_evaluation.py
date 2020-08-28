"""Materials science evaluation

Script to generate the plots for evaluating our case study in materials science.
The evaluation script for the synthetic-constraints pipeline can also be applied.
"""


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results = pd.read_csv('data/ms-results/results.csv')
CONSTRAINT_METRICS = ['objective_value', 'num_selected', 'num_constraints', 'frac_solutions']
PREDICTION_METRICS = [x for x in results.columns if x.endswith('_r2')]
EVALUATION_METRICS = CONSTRAINT_METRICS + ['linear-regression_test_r2', 'xgb-tree_test_r2']


def reshape_prediction_data(results_data: pd.DataFrame) -> pd.DataFrame:
    results_data = results_data.melt(id_vars=['dataset_name', 'constraint_name'], value_vars=PREDICTION_METRICS,
                                     var_name='model', value_name='r2')
    results_data['split'] = results_data['model'].apply(lambda x: 'train' if 'train' in x else 'test')
    results_data['model'] = results_data['model'].str.replace('_train_r2', '').str.replace('_test_r2', '')
    return results_data


# ---Relationship between constraint evaluation metrics and datasets---
# In contrast to the experiments with syntheic constraints, we have so few datasets that we can
# also compare them directly (not only with boxplots)

results[results['constraint_name'] == 'UNCONSTRAINED'].plot(
    x='dataset_name', y=['xgb-tree_train_r2', 'xgb-tree_test_r2'], kind='bar')
plt.show()

for evaluation_metric in EVALUATION_METRICS:
    sns.barplot(x='dataset_name', y=evaluation_metric, hue='constraint_name', data=results)
    if evaluation_metric.endswith('_r2'):
        plt.ylim(-0.1, 1.1)
    plt.xticks(rotation=90)
    plt.show()
