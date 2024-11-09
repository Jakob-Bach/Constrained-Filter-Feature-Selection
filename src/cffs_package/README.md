# `cffs` -- A Python Package for Constrained Filter Feature Selection

The package `cffs` contains classes to formulate and solve constrained-feature-selection problems.
In particular, we support univariate filter feature selection as the objective
and constraints from propositional logic and linear arithmetic.
We use an SMT (Satisfiability Modulo Theories) solver to find optimal feature sets under the constraints.

This document provides:

- Steps for [setting up](#setup) the package.
- A short [overview](#functionality) of the (constrained-feature-selection) functionality.
- A [demo](#demo) of the functionality.
- [Guidelines for developers](#developer-info) who want to modify or extend the code base.

If you use this package for a scientific publication, please cite [our paper](https://doi.org/10.1007/s42979-022-01338-z)

```
@article{bach2022empirical,
  title={An Empirical Evaluation of Constrained Feature Selection},
  author={Bach, Jakob and Zoller, Kolja and Trittenbach, Holger and Schulz, Katrin and B{\"o}hm, Klemens},
  journal={SN Computer Science},
  volume={3},
  number={6},
  year={2022},
  doi={10.1007/s42979-022-01338-z}
}
```

## Setup

You can install our package from [PyPI](https://pypi.org/):

```
python -m pip install cffs
```

Alternatively, you can install the package from GitHub:

```bash
python -m pip install git+https://github.com/Jakob-Bach/Constrained-Filter-Feature-Selection.git#subdirectory=src/cffs_package
```

If you already have the source code for the package (i.e., the directory in which this `README` resides)
as a local directory on your computer (e.g., after cloning the project), you can also perform a local install:

```bash
python -m pip install .
```

## Functionality

The package `cffs` contains the following modules:

- `combi_expressions.py`: Formulate constraints in propositional logic and linear arithmetic,
  simultaneously for our own expression classes (`expressions.py`) and the solver `Z3`.
  These constraints may be added to an optimization problem in `combi_solving.py`.
- `combi_solving.py`: Formulate a constrained-filter-feature-selection *optimization* problem
  with constraints from `combi_expressions.py`.
  Count the number of solutions with our own implementation (`solving.py`) and optimize the problem with `Z3`.
- `expressions.py`: Formulate constraints in propositional logic and linear arithmetic,
  using our own expression classes.
  These constraints may be added to a satisfiability problem in `solving.py`.
- `feature_qualities.py`: Compute univariate feature qualities for the (linear) optimization objective.
- `solving.py`: Formulate a constrained-filter-feature-selection *satisfiability* problem
  with constraints from `expressions.py`.
  Count the number of solutions with our own implementation; optimization is not supported.

## Demo

For constrained filter feature selection, we first need to compute an individual quality score for each feature.
(We only consider univariate feature selection, so we ignore interactions between features.)
The objective value is the summed quality of all selected features.
To this end, `cffs.feature_qualities` provides functions for

- the absolute value of Pearson correlation between each feature and the prediction target (`abs_corr()`)
- the mutual information between each feature and the prediction target (`mut_info()`)

Both functions round the qualities to two digits to speed up solving.
(We found that the solver becomes slower the more precise the floats are,
as they are represented as rational numbers in the solver.)
As inputs, the quality functions require a dataset in X-y form (as used in `sklearn`).

After computing feature qualities, we set up an SMT optimization problem from `cffs.combi_solving`.
It is "combi" in the sense that our code wraps an existing SMT solver (`Z3`).
We retrieve the problem's decision variables (one binary variable for each feature) and use them to
formulate constraints with `cffs.combi_expressions`.
These constraints are added to `Z3` but also to our own expression tree,
which we use to count the number of valid solutions in the search space.
Finally, we start optimization.

```python
from cffs.combi_expressions import And, AtMost, Xor
from cffs.combi_solving import Problem
from cffs.feature_qualities import mut_info
import sklearn.datasets

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
feature_qualities = mut_info(X=X, y=y)
problem = Problem(variable_names=X.columns, qualities=feature_qualities)

print('--- Constrained problem ---')
variables = problem.get_variables()
problem.add_constraint(AtMost(variables, 2))
problem.add_constraint(And([Xor(variables[0], variables[1]), Xor(variables[2], variables[3])]))
print(problem.optimize())
print('Number of constraints:', problem.get_num_constraints())
print('Fraction of valid solutions:', problem.compute_solution_fraction())

print('\n--- Unconstrained problem ---')
problem.clear_constraints()
print(problem.optimize())
print('Number of constraints:', problem.get_num_constraints())
print('Fraction of valid solutions:', problem.compute_solution_fraction())
```

The output is the following:

```
--- Constrained problem ---
{'objective_value': 1.48, 'num_selected': 2, 'selected': ['petal width (cm)', 'sepal length (cm)']}
Number of constraints: 2
Fraction of valid solutions: 0.25

--- Unconstrained problem ---
{'objective_value': 2.71, 'num_selected': 4, 'selected': ['petal width (cm)', 'petal length (cm)', 'sepal length (cm)', 'sepal width (cm)']}
Number of constraints: 0
Fraction of valid solutions: 1.0
```

The optimization procedure returns the objective value (summed quality of selected features)
and the feature selection.
To assess how strongly the constraints cut down the space of valid solutions,
we can use `compute_solution_fraction()`.
However, this function iterates over each solution candidate and checks whether it is valid or not,
which becomes very expensive with a growing number of features.
Alternatively, `estimate_solution_fraction()` randomly samples solutions to estimate this quantity.

Our code snippet also shows that you can remove all constraints via `clear_constraints()` without setting up a new optimization problem.
You can also add further constraints after optimization and then optimize again.
The optimizer keeps its state between optimizations, so you may benefit from a warm start.

## Developer Info

New operators / expressions types for constraints in `expressions.py` should (directly or indirectly)
inherit from the top-level superclass `Expression`. Preferably, inherit from one of its subclasses

- `BooleanExpression` (if your operator returns a boolean value; need to override `is_true()`) or
- `ArithmeticExpression` (if your operator returns a numeric value; need to override `get_value()`).

`is_true()` or `get_value()` express the logic of your operator, i.e., how it will be evaluated
(typically depending on the values of child expressions serving as operands).
Also, you need to override `get_children()` to return all (direct) child expressions,
so it is possible to traverse the full expression tree.

`solving.py` supports adding arbitrary `BooleanExpression`s from `expressions.py` as constraints.

New operators / expression types for constraints in `combi_expressions.py` should simultaneously inherit from
the superclass `BooleanExpression` in this module and one expression class from `expressions.py`.
Arithmetic expressions currently do not exist on their own in this module,
but only nested into more complex boolean expressions (like "weighted sum <= some threshold").
You need to initialize the `expressions.py` expression (preferably in the initializer by calling `super().__init__()`)
and store a corresponding `Z3` expression in the field `z3_expr`;
use `get_z3()` to access `Z3` representations of your child expressions serving as operands.

`combi_solving.py` supports adding arbitrary `BooleanExpression`s from `combi_expressions.py` as constraints.
Also, it supports arbitrary univariate feature qualities (e.g., computed with `feature_qualities.py`)
to formulate a univariate (= linear) objective function.

New functions for univariate feature qualities in `feature_qualities.py` should satisfy the following implicit interface:

- Two inputs: Dataset `X` (`pandas.DataFrame`) and prediction target `y` (`pandas.Series`).
- Output is a sequence (e.g., list) of floats whose length equals the number of columns in `X`.
