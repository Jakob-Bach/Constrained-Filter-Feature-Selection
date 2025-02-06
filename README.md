# An Empirical Evaluation of Constrained Feature Selection

This repository contains the code for

- a paper,
- (parts of) a dissertation,
- and the Python package [`cffs`](https://pypi.org/project/cffs/).

This `README` provides:

- An overview of the [related publications](#publications).
- An outline of the [repo structure](#repo-structure).
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Publications

> Bach, Jakob, et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"

is a paper published in the journal [*SN Computer Science*](https://www.springer.com/journal/42979).
You can find the paper [here](https://doi.org/10.1007/s42979-022-01338-z).
You can find the corresponding complete experimental data (inputs as well as results) on [KITopenData](https://doi.org/10.5445/IR/1000148891).
We tagged the commits for reproducing these data:

- Use the tag `syn-pipeline-2021-03-26-paper-accept` to run `prepare_openml_datasets.py` and `syn_pipeline.py`.
- Use the tag `ms-pipeline-2021-03-26-paper-accept` to run `prepare_ms_dataset.py` and `ms_pipeline.py`.
- Use the tag `evaluation-2021-08-10-paper-accept` to run `syn_evaluation_journal.py` and `ms_evaluation_journal.py`.

> Bach, Jakob (2025): "Leveraging Constraints for User-Centric Feature Selection"

is a dissertation at the [Department of Informatics](https://www.informatik.kit.edu/english/index.php) of the [Karlsruhe Institute of Technology](https://www.kit.edu/english/).
You can find the dissertation [here](https://doi.org/10.5445/IR/1000178649).
You can find the corresponding complete experimental data (inputs as well as results) on [*RADAR4KIT*](https://doi.org/10.35097/4kjyeg0z2bxmr6eh).
We tagged the commits for reproducing these data:

- Use the tag `syn-pipeline-2021-03-26-dissertation` to run `prepare_openml_datasets.py` and `syn_pipeline.py` (same commit as for journal version).
- Use the tag `ms-pipeline-2021-03-26-dissertation` to run `prepare_ms_dataset.py` and `ms_pipeline.py` (same commit as for journal version).
- Use the tag `evaluation-2024-11-02-dissertation` to run `syn_evaluation_dissertation.py` and `ms_evaluation_dissertation.py`.

## Repo Structure

On the top level, there are the following (non-code) files:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

The folder `src` contains the code in multiple sub-directories:

- `cffs_package`: Code for SMT expressions (to formulate constraints), solving and optimization.
  Organized as the standalone Python package `cffs` (i.e., can be used without the remaining code).
  See the [corresponding README](src/cffs_package/README.md) for more information.
- `materials_science`: Code for our case study with manually defined constraints in materials science.
- `synthetic_constraints`: Code for our study with synthetically generated constraints on arbitrary datasets.
- `utilities`: Code for the experimental pipelines, like data I/O, computing feature qualities, and predicting.

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.7).

### Option 1: `conda` Environment

If you use `conda`, you can install the correct Python version into a new `conda` environment
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
Run the scripts `syn_evaluation_journal.py` and `ms_evaluation_journal.py` to create the paper's plots or
run the scripts `syn_evaluation_dissertation.py` and `ms_evaluation_dissertation.py` to create the dissertation's plots.
These scripts save the plots as PDFs.
You can specify the input and output directory.

Execute all scripts from the `src` directory of this repo like this:

```bash
python -m synthetic_constraints.prepare_demo_dataset <options>
```

`-m <module.import.syntax>` makes sure imports of sub-packages work.
Note that you need to leave out the file ending `.py` in that call.
Passing `--help` as an option gives you an overview of each script's further options.
