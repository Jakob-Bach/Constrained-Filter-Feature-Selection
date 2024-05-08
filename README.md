# An Empirical Evaluation of Constrained Feature Selection

This repository contains the code to reproduce the experiments of the paper

> Bach, Jakob, et al. "An Empirical Evaluation of Constrained Feature Selection"

published at the journal [*SN Computer Science*](https://www.springer.com/journal/42979).
You can find the paper [here](https://doi.org/10.1007/s42979-022-01338-z).
You can find the corresponding complete experimental data (inputs as well as results) on [KITopenData](https://doi.org/10.5445/IR/1000148891).
This document describes:

- An outline of the [repo structure](#repo-structure).
- A [demo](#demo) of the core functionality.
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Repo Structure

The code is organized as a Python package called `cffs`, with multiple sub-packages:

- `core`: Code for SMT expressions (to formulate constraints), solving and optimization.
- `materials_science`: Code for our case study with manually-defined constraints in materials science.
- `synthetic_constraints`: Code for our study with synthetically generated constraints on arbitrary datasets.
- `utilities`: Code for the experimental pipelines, like data I/O, computing feature qualities, and predicting.

You can find more information on individual files below, where we describe the steps to reproduce the experiments.

## Demo

For constrained filter feature selection, we first need to compute an individual quality score for each feature.
(We only consider univariate feature selection, so we ignore interactions between features.)
To this end, `cffs.utilities.feature_qualities` provides functions for

- the absolute value of Pearson correlation between each feature and the prediction target (`abs_corr()`)
- the mutual information between each feature and the prediction target (`mut_info()`)

Both functions round the qualities to two digits to speed up solving.
(We found that the solver becomes slower the more precise the floats are,
as they are represented as rational numbers.)
As inputs, the quality functions require a dataset in X-y form (as used in `sklearn`).

After computing feature qualities, we set up an SMT optimization problem from `cffs.core.combi_solving`.
It's "combi" in the sense that our code wraps an existing SMT solver (`Z3`).
We retrieve the problem's decision variables (one binary variable for each feature) and use them to
formulate constraints with `cffs.core.combi_expressions`.
These constraints are added to `Z3` but also to our own expression tree,
which we use to count the number of valid solutions in the search space.
Finally, we start optimization.

```python
from cffs.core.combi_expressions import And, AtMost, Xor
from cffs.core.combi_solving import Problem
from cffs.utilities.feature_qualities import mut_info
import sklearn.datasets

X, y = sklearn.datasets.load_iris(as_frame=True, return_X_y=True)
feature_qualities = mut_info(X=X, y=y)
problem = Problem(variable_names=X.columns, qualities=feature_qualities)

print('--- -Constrained problem ---')
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
--- -Constrained problem ---
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
However, this function iterates over each solution candidate and checks whether it is valid or not.
Alternatively, `estimate_solution_fraction()` randomly samples solutions to estimate this quantity.

Our code snippet also shows that you can remove all constraints without setting up a new optimization problem.
You can also add further constraints after optimization and then optimize again.
The optimizer keeps its state between optimizations, so you may benefit from a warm start.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.7).

### Option 1: `conda` Environment

If you use `conda`, you can install the right Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.7
conda activate <conda-env-name>
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.0) to create an environment for our experiments.
First, make sure you have the right Python version available.
Next, you can install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.0
```

To set up an environment with `virtualenv`, run


```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, simply run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

To leave the environment, run

```bash
deactivate
```

### Optional Dependencies

To use the environment in the IDE `Spyder`, you need to install `spyder-kernels` into the environment.

To run or create notebooks from the environment, you need to install `juypter` into the environment.
Next, you need to install a kernel for the environment:

```bash
ipython kernel install --user --name=<kernel-name>
```

After that, you should see (and be able to select) the kernel when running `Juypter Notebook` with

```bash
jupyter notebook
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
You can reproduce the results of both studies with the same three steps, i.e., by running three scripts each:

1. **Prepare datasets:**
Run the script `prepare_openml_datasets.py` or `prepare_ms_dataset.py` to prepare input data for the experimental pipeline.
(If you use the experimental data linked above, you can skip this step for the materials-science dataset.)
These scripts apply some pre-processing and then save feature data (`X`) and prediction target (`y`) as CSVs for each dataset.
You can specify the output directory.
We recommend `data/openml/` and `data/ms/` as output directories, so the following pipeline scripts work without specifying a directory.
For the materials-science pre-processing, you need to provide the raw voxel dataset `voxel_data.csv` as an input.
For the OpenML pre-processing, you need an internet connection, as the datasets are downloaded first.
`prepare_demo_dataset.py` is a lightweight alternative to test the pipeline for synthetic constraints,
as it just prepares one dataset, which is already part of `sklearn`.
2. **Run experimental pipeline:**
Run the script `syn_pipeline.py` or `ms_pipeline.py` to execute the experimental pipeline.
These scripts save the results as one or more CSV file(s).
A merged results file is available as `results.csv`.
You can specify various options, e.g., output directory, number of cores, number of repetitions, etc.
We recommend using the default output directories `data/openml-results/` and `data/ms-results/`,
so the following evaluation scripts work without specifying a directory.
3. **Run evaluation:**
Run the script `syn_evaluation.py` or `ms_evaluation.py` to create the paper's plots.
These scripts save the plots as PDFs.
You can specify the input and output directory.

Execute all scripts from the top-level directory of this repo like this:

```bash
python -m cffs.synthetic_constraints.prepare_demo_dataset <options>
```

`-m <module.import.syntax>` makes sure imports of sub-packages work.
Note that you need to leave out the file ending `.py` in that call.
Passing `--help` as an option gives you an overview of each script's further options.
