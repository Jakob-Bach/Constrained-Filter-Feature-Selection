# Constrained Filter Feature Selection

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

To run or create notebooks from the environment, you need first to install a kernel for it:

```bash
pipenv run ipython kernel install --user --name=cffs_kernel
```

After that, you should see the kernel when running Juypter notebook with

```bash
pipenv run jupyter notebook
```
