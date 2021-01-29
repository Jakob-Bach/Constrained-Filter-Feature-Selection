# Constrained Filter Feature Selection

## Structure

The code is organized as a Python package `cffs` with multiple sub-packages:

- `core`: Defining constraints, optimizing and solution counting. Use-case independent.
- `materials_science`: Code for our case study with manually-defined constraints in materials science.
- `synthetic_constraints`: Code for our experiments with synthetically generated constraints on general benchmark datasets.
- `utilities`: Use-case independent code like data I/O, computing feature qualities and predicting.

## Setup

Our code is implemented in Python (version 3.7).
We use [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.0) for dependency management.
If already using `conda`, run

```bash
> conda create --name cffs python=3.7
> conda activate cffs
> python -m pip install virtualenv==20.4.0
```

to get the right Python version together with `virtualenv`.

To setup the actual environment, go into the top-level folder of this project and run

```bash
virtualenv -p <<path/to/python/executable>> <<path/to/env/dest>>
```

Activate the environment in Linux with

```bash
source <<path/to/env/dest>>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```
<<path\to\env\dest>>\Scripts\activate
```

After activating the environment, you can use `python` and `pip` as usual.
To install all dependencies, simply run

```bash
pip install -r requirements.txt
```

If you want to install new packages or run updates, use pip as usual.
You can change the requirements file, if you like.

```bash
pip freeze > requirements.txt
```

To leave the environment, run

```bash
deactivate
```

### Extended Setup

To use the environment in the `Spyder` IDE, you need to install `spyder-kernels` into the environment.
To run or create notebooks from the environment, you first need to install `juypter` into the environment.
Next, you need to install a kernel for the environment:

```bash
ipython kernel install --user --name=cffs_kernel
```

After that, you should see the kernel when running Juypter notebook with

```bash
jupyter notebook
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
python -m cffs.synthetic_constraints.prepare_demo_dataset <<options>>
```

Don't forget to activate the environment first.
`-m <<module.import.syntax>>` makes sure imports of sub-packages work.
Note we leave out the file ending `.py`.
Passing `--help` as option gives you an overview of a script's options.
