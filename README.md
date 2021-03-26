# Constrained Filter Feature Selection

This repository contains the code to reproduce the experiments of the paper

> Bach, Jakob, et al. "Evaluating the Impact of Constraints on Feature Selection"

This document describes the repo structure and the steps to reproduce the experiments.
Input data and results data of the experimental pipelines are also available [online](https://bwdatadiss.kit.edu/dataset/xxx).

## Repo Structure

The code is organized as a Python package called `cffs`, with multiple sub-packages:

- `core`: Code for SMT expressions (to formulate constraints), solving and optimization.
- `materials_science`: Code for our case study with manually-defined constraints in materials science.
- `synthetic_constraints`: Code for our study with synthetically generated constraints on arbitrary datasets.
- `utilities`: Code for the experimental pipelines, like data I/O, computing feature qualities and predicting.

You can find more information on individual files below, where we describe the steps to reproduce the experiments.

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
You can reproduces the results of both studies with the same three steps, i.e., by running three scripts each:

1. **Prepare datasets:**
Run the script `prepare_openml_datasets.py` or `prepare_ms_dataset.py` to prepare input data for the experimental pipeline.
These scripts apply some pre-processing and then save feature data (`X`) and prediction target (`y`) as CSVs for each dataset.
You can specify the output directory.
We recommend `data/openml/` and `data/ms/` as output directories, so the following pipeline scripts work without specifying a directory.
For the materials-science pre-processing, you need to provide the raw voxel dataset `voxel_data.csv` as an input.
For the OpenML pre-processing, you need an internet connection, as the datasets are downloaded first.
`prepare_demo_dataset.py` is a light-weight alternative to test the pipeline for synthetic constraints,
as it just prepares one dataset, which is already part of `sklearn`.
2. **Run experimental pipeline:**
Run the script `syn_pipeline.py` or `ms_pipeline.py` to execute the experimental pipeline.
These scripts save the results as one or more CSV file(s).
A merged results file is available as `results.csv`.
You can specify various options, e.g., output directory, number of cores, number of repetitions etc.
We recommend to use the default output directories `data/openml-results/` and `data/ms-results/`,
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
