# Run ms_pipeline.py before
import matplotlib.pyplot as plt

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
