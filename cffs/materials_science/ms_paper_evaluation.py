"""Paper evaluation for materials science part

Script to compute all summary statistics and create all plots used in the paper.
"""


import ast
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cffs.utilities import data_utility
from cffs.utilities import evaluation_utility


RESULTS_PATH = pathlib.Path('data/ms-results/')
PLOT_PATH = '../paper-cffs-text/plots/'

results = data_utility.load_results(directory=RESULTS_PATH)
results['cardinality'] = results['constraint_name'].str.extract('_k([0-9]+)$').astype(int)
results['constraint_name'] = results['constraint_name'].str.replace('_k[0-9]+$', '', regex=True)
results['selected'] = results['selected'].apply(ast.literal_eval)  # make list string a list
os.makedirs(PLOT_PATH, exist_ok=True)

ORIGINAL_PRED_METRICS = [x for x in results.columns if x.endswith('_r2') and not x.startswith('frac_')]

# ---Prediction performance---

# Figure 6
prediction_data = evaluation_utility.reshape_prediction_data(results, additional_columns=['cardinality'])
plt.figure(figsize=(4, 3))
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data.drop(columns='cardinality'))
plt.xticks(rotation=20)
plt.ylim(0.8, 1.01)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'ms-prediction-performance-split.pdf')
plt.figure(figsize=(4, 3))
sns.boxplot(x='model', y='r2', hue='cardinality', data=prediction_data[prediction_data['split'] == 'test'].drop(columns='split'))
plt.xticks(rotation=20)
plt.ylim(0.8, 1.01)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'ms-prediction-performance-cardinality.pdf')

# For comparison: Aggregate statistics
# print(results[ORIGINAL_PRED_METRICS].describe().transpose()[['min', '50%', 'max']])
# assert (results.loc[results['cardinality'] == 5, 'constraint_name'].reset_index(drop=True) ==
#         results.loc[results['cardinality'] == 10, 'constraint_name'].reset_index(drop=True)).all()
# print((results.loc[results['cardinality'] == 10, ORIGINAL_PRED_METRICS].reset_index(drop=True) -
#        results.loc[results['cardinality'] == 5, ORIGINAL_PRED_METRICS].reset_index(drop=True)).
#       describe().transpose()[['min', '50%', 'max']])

# ---Objective value---

print(results.groupby('cardinality')['objective_value'].describe())
print(results[results['constraint_name'] != 'Mixed'].groupby('cardinality')['objective_value'].describe())

# For comparison: box plot
# plt.figure(figsize=(4, 3))
# sns.boxplot(x='cardinality', y='objective_value', data=results)
# plt.xticks(rotation=20)
# plt.ylim(0, 10)
# plt.tight_layout()
# plt.savefig(PLOT_PATH + 'ms-objective-value-cardinality.pdf')

# For comparison: bar chart
# plt.figure(figsize=(8, 3))
# sns.barplot(x='constraint_name', hue='cardinality', y='objective_value', data=results)
# plt.xticks(rotation=20)
# plt.ylim(0, 10)
# plt.tight_layout()

# ---Selected features---

# Test whether always the maximum number of possible features is selected:
print((results['num_selected'] == results['cardinality']).all())

# Prepare Figure 7:
# Make sure results belonging to same cardinality follow each other (is assumed in naming below):
assert (results['cardinality'].diff() != 0).sum() == results['cardinality'].nunique()
# Compute similarity between feature sets:
similarity_matrix = np.zeros(shape=(len(results), len(results)))
for i in range(len(results)):
    for j in range(len(results)):
        similarity_matrix[i, j] = len(set(results['selected'].iloc[i]).intersection(results['selected'].iloc[j]))
# Use names from paper:
constraint_names = [f'(D{i+1})' for i in range(results['constraint_name'].nunique())] * results['cardinality'].nunique()
similarity_matrix = pd.DataFrame(similarity_matrix, index=constraint_names, columns=constraint_names)

# Figure 7
plt.figure(figsize=(5, 5))
sns.heatmap(similarity_matrix.iloc[:12, :12], vmin=0, vmax=5, cmap='RdYlGn',
            annot=True, square=True, cbar=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'ms-selected-similarity-card5.pdf')
plt.figure(figsize=(5, 5))
sns.heatmap(similarity_matrix.iloc[12:, 12:], vmin=0, vmax=10, cmap='RdYlGn',
            annot=True, square=True, cbar=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'ms-selected-similarity-card10.pdf')

# Compute drop in objective value if adding the hardest combination of constraints
grouping = results.groupby('cardinality')
print(1 - grouping['objective_value'].min() / grouping['objective_value'].max())
