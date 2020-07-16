"""Synthetic constraints evaluation snippets

Code snippets for evaluating the results of our experiments with synthetic constraints.
"pipeline.py" needs to be run beforehand.
"""


import matplotlib.pyplot as plt
import seaborn as sns


# Compare impact of constraint types on multiple result quantities
results.groupby('constraint_name').agg('mean')

# Compare impact of constraint types on one result quantity
results.boxplot(column='frac_solutions', by='constraint_name')
plt.xticks(rotation=45)
plt.show()
# Without causing eye cancer:
sns.boxplot(x='constraint_name', y='frac_solutions', data=results)
plt.xticks(rotation=45)
plt.show()

# Analyze multiple result quantities for one constraint type
results[results['constraint_name'] == 'global-AT-MOST'].describe()
results[results['constraint_name'] == 'global-AT-MOST'].plot(kind = 'box')

# Plot two result quantities against each other
print(list(results))  # columns for plotting
for constraint_name in results['constraint_name'].unique():
    results[results['constraint_name'] == constraint_name].plot(
        kind='scatter', x='frac_solutions', y='objective_value')
    plt.title(constraint_name)
    plt.show()
