"""Materials science evaluation snippets

Code snippets for evaluating the results of our case stuy in materials science.
"ms_pipeline.py" needs to be run beforehand.
"""


import matplotlib.pyplot as plt
import seaborn as sns

# Compare impact of constraint types on multiple result quantities
results.groupby('constraint_name').mean().transpose()

# Compare impact of constraint types on one result quantity
sns.boxplot(x='constraint_name', y='objective_value', data=results)
plt.xticks(rotation=45)
plt.show()

# Compare prediction models
prediction_columns = [x for x in results.columns if '_r2' in x]
sns.boxplot(data=results[prediction_columns])
plt.ylim(-1.1, 1.1)
plt.xticks(rotation=45)
plt.show()

# Compare datasets
results[(results['constraint_name'] == 'UNCONSTRAINED') &
        (results['dataset_name'].str.contains('absolute'))].plot(x='dataset_name', y=prediction_columns, kind='bar')
plt.show()

# Compare train against test scores
reshaped_results = results[['dataset_name', 'constraint_name'] + prediction_columns]
reshaped_results = reshaped_results.melt(id_vars=['dataset_name', 'constraint_name'], value_vars=prediction_columns,
                                         value_name='R2')
reshaped_results['model'] = reshaped_results['variable'].str.extract('^([^_]*)')
reshaped_results['split'] = reshaped_results['variable'].str.extract('(train|test)')
reshaped_results.drop(columns='variable', inplace=True)
sns.barplot(x='model', y='R2', hue='split',
            data=reshaped_results[(reshaped_results['constraint_name'] == 'UNCONSTRAINED') &
                                  (reshaped_results['dataset_name'] == 'delta_sampled_2400_absolute_glissile')])
plt.show()
sns.boxplot(x='model', y='R2', hue='split',
            data=reshaped_results[(reshaped_results['constraint_name'] == 'UNCONSTRAINED') &
                                  (reshaped_results['dataset_name'].str.contains('absolute'))])
plt.show()
# Impact of constraint types on test scores
sns.boxplot(x='model', y='R2', hue='constraint_name',
            data=reshaped_results[(reshaped_results['split'] == 'test') &
                                  (reshaped_results['dataset_name'].str.contains('absolute'))])
plt.show()
