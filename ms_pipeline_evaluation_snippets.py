"""Materials science evaluation snippets

Code snippets for evaluating the results of our case stuy in materials science.
"ms_pipeline.py" needs to be run beforehand.
"""


import matplotlib.pyplot as plt

# Compare prediction targets
results[(results['model'] == 'xgboost') & (results['num_features_rel'] == 1)].set_index(['name', 'target']).plot(
    y=['train_score', 'test_score'], kind='bar', ylim=(-0.2, 1.2), figsize=(16,9), rot=45)
plt.title('Prediction $R^2$, xgboost, no feature selection')
plt.tight_layout()
plt.savefig('Performance_xgboost_noFS.pdf')

# Compare train against test scores
for scenario in results['name'].unique():
    results[(results['name'] == scenario) & (results['num_features_rel'] == 1)].set_index(['target', 'model']).plot(
        y=['train_score', 'test_score'], kind='bar', ylim=(-0.2, 1.2), title='Scenario: '+ scenario)
    plt.show()

# Compare feature selections
for scenario in results['name'].unique():
    results[(results['name'] == scenario)].pivot_table(
        index=['target', 'model'], columns='num_features_rel', values='test_score').plot(
            kind='bar', ylim=(-0.2, 1.2), title='Scenario: '+ scenario)
    plt.show()
