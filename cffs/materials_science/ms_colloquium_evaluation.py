"""Plots for the IAM-CMS Collquium on 2020-11-27.

Slightly modified extract of the MS paper plot script.
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
PLOT_PATH = '../paper-cffs-material/IAM-CMS-Colloquium-2020-11-27/'
plt.rcParams.update({'font.size': 14})

results = data_utility.load_results(directory=RESULTS_PATH)
results['cardinality'] = results['constraint_name'].str.extract('_k([0-9]+)$').astype(int)
results['constraint_name'] = results['constraint_name'].str.replace('_k[0-9]+$', '', regex=True)
results['selected'] = results['selected'].apply(ast.literal_eval)  # make list string a list
os.makedirs(PLOT_PATH, exist_ok=True)

# ---Prediction performance---

prediction_data = evaluation_utility.reshape_prediction_data(results[results['cardinality'] == 5])
plt.figure(figsize=(4, 3))
sns.boxplot(x='model', y='r2', hue='split', data=prediction_data)
plt.xticks(rotation=20)
plt.ylim(-0.05, 1.05)
plt.xlabel('Prediction model')
plt.ylabel('Performance ($R^2$)')
plt.tight_layout()
plt.savefig(PLOT_PATH + 'prediction-performance.eps')

# --- Selected features ---
# Make sure results belonging to same cardinality follow each other (is assumed in naming below):
assert (results['cardinality'].diff() != 0).sum() == results['cardinality'].nunique()
# Compute similarity between feature sets:
similarity_matrix = np.zeros(shape=(len(results), len(results)))
for i in range(len(results)):
    for j in range(len(results)):
        similarity_matrix[i, j] = len(set(results['selected'][i]).intersection(results['selected'][j]))
constraint_names = [f'(C{i+1})' for i in range(results['constraint_name'].nunique())] * results['cardinality'].nunique()
similarity_matrix = pd.DataFrame(similarity_matrix, index=constraint_names, columns=constraint_names)
# Figure itself:
plt.figure(figsize=(4, 4))
sns.heatmap(similarity_matrix.iloc[:12, :12], vmin=0, vmax=5, cmap='RdYlGn',
            annot=True, square=True, cbar=False)
plt.tight_layout()
plt.savefig(PLOT_PATH + 'selected-similarity.eps')
