# Constrained Filter Feature Selection

## Structure

The code is organized as a Python package `cffs` with multiple sub-packages:

- `core`: Defining constraints, optimizing and solution counting. Use-case independent.
- `materials_science`: Code for our case study with manually-defined constraints in materials science.
- `synthetic_constraints`: Code for our experiments with synthetically generated constraints on general benchmark datasets.
- `utilities`: Use-case independent code like data I/O, computing feature qualities and predicting.

## Setup

Our code is implemented in Python (version 3.7).
We use [`pipenv`](https://pypi.org/project/pipenv/) (version 2020.6.2) for dependency management.
If already using `conda`, run

```bash
> conda create --name cffs python=3.7
> conda activate cffs
> python -m pip install pipenv==2020.6.2
```

to get the right Python version together with `pipenv`.

To install all dependencies, simply run

```bash
pipenv install
```

If `pipenv` is not picking up the right Python from your computer (e.g., if Python is in a `conda` environment),
you can also specify the exact path or version number, e.g.:

```bash
pipenv install --python C:/Anaconda3/envs/cffs/python.exe
```

If you want to install new packages (not necessary for executing the code), use

```bash
pipenv install <<package_name>>
```

To run or create notebooks from the environment, you need first to install a kernel for it:

```bash
pipenv run ipython kernel install --user --name=cffs_kernel
```

After that, you should see the kernel when running Juypter notebook with

```bash
pipenv run jupyter notebook
```

## Reproducing the Experiments

Both use cases follow the same three steps:

1. Run the script `prepare_.*_dataset.py` to prepare input data for the pipeline.
This applies use-case specific pre-processing and then saves feature data (X) and prediction target (y) as CSVs for each dataset.
You might specify the output directory.
`prepare_demo_dataset.py` is a light-weight alternative to test the pipeline for synthetic constraints, as it just prepares one dataset.
2. Run the script `.*_pipeline.py` to execute the experimental pipeline.
It saves the results as one or more CSV file(s).
The overall results file is named `results.csv`.
You might specify various options, e.g., output directory, number of core, number of repetitions etc.
3. Run the script `.*_paper_evaluation` to create the paper's plots.
It saves the plots as PDFs in a specified directory.

Execute scripts from the top-level directory of the repo like this:

```bash
pipenv run python -m cffs.synthetic_constraints.prepare_demo_dataset <<options>>
```

`pipenv run python` makes sure Python uses our environment.
`-m <<module.import.syntax>>` makes sure imports of sub-packages work.
Note we leave out the file ending `.py`.
Passing `--help` as option gives you an overview of a script's options.
