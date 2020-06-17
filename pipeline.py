"""Synthetic constraints pipeline

Main script for our experiments with synthetic constraints.
"""


import pandas as pd
from sklearn.datasets import load_boston
from tqdm import tqdm

import combi_expressions
import combi_solving
import generation


dataset = load_boston()
features = dataset['feature_names']
X = pd.DataFrame(dataset['data'], columns=features)
y = pd.Series(dataset['target'])
variables = [combi_expressions.Variable(name=feature) for feature in features]
qualities = [round(abs(X[feature].corr(y)), 2) for feature in features]
problem = combi_solving.Problem(variables=variables, qualities=qualities)

generators = {
    'group_AT_LEAST': generation.AtLeastGenerator(problem, global_at_most=10),
    'group_AT_MOST': generation.AtMostGenerator(problem),
    'global_AT_MOST': generation.GlobalAtMostGenerator(problem),
    'single_IFF': generation.IffGenerator(problem, global_at_most=10),
    'group_IFF': generation.IffGenerator(problem, global_at_most=10, max_num_variables=5),
    'single_NAND': generation.NandGenerator(problem),
    'group_NAND': generation.NandGenerator(problem, max_num_variables=5),
    'single_XOR': generation.XorGenerator(problem)
}
for generator in generators.values():
    generator.num_repetitions = 10
    generator.min_num_constraints = 1
    generator.max_num_constraints = 10

results = []
for name, generator in tqdm(generators.items()):  # progress bar wrapped around iterable
    result = generator.evaluate_constraints()
    result['constraint_type'] = name
    results.append(result)
results = pd.concat(results)
