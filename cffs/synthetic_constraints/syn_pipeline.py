"""Synthetic constraints pipeline

Main script for our experiments with synthetic constraints.

Usage: python syn_pipeline.py --help
"""


import argparse
import multiprocessing
import pathlib
import sys
from typing import Any, Optional, Sequence

import pandas as pd
import tqdm

from cffs.core import combi_solving
from cffs.utilities import data_utility
from cffs.utilities import feature_qualities
from cffs.utilities import prediction_utility
from cffs.synthetic_constraints import syn_constraints


COMMON_GENERATOR_ARGS = {'num_iterations': 1000, 'min_num_constraints': 1, 'max_num_constraints': 10}

GENERATORS = {
    'group-AT-LEAST': {'func': 'AtLeastGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5}},
    'group-AT-MOST': {'func': 'AtMostGenerator', 'args': COMMON_GENERATOR_ARGS},
    'global-AT-MOST': {'func': 'GlobalAtMostGenerator', 'args': COMMON_GENERATOR_ARGS},
    'single-IFF': {'func': 'IffGenerator',
                   'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5, 'max_num_variables': 2}},
    'group-IFF': {'func': 'IffGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5}},
    'single-NAND': {'func': 'NandGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'max_num_variables': 2}},
    'group-NAND': {'func': 'NandGenerator', 'args': COMMON_GENERATOR_ARGS},
    'single-XOR': {'func': 'XorGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'max_num_variables': 2}},
    'MIXED': {'func': 'MixedGenerator', 'args': COMMON_GENERATOR_ARGS},
    'UNCONSTRAINED': {'func': 'NoConstraintGenerator', 'args': COMMON_GENERATOR_ARGS}
}


def evaluate_constraint_type(
        generator_name: str, n_iterations: int, dataset_name: str, data_dir: pathlib.Path,
        quality_names: Sequence[str], model_names: Sequence[str] = None, n_splits: int = 1,
        results_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    if model_names is None:
        model_names = []  # this is more Pythonic than using an empty list as default
    results = []
    X, y = data_utility.load_dataset(dataset_name=dataset_name, directory=data_dir)
    for split_idx, (train_idx, test_idx) in enumerate(prediction_utility.create_split_idx(X, n_splits=n_splits)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        if len(test_idx) > 0:
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
        else:
            X_test = None
            y_test = None
        for quality_name in quality_names:
            qualities = feature_qualities.QUALITIES[quality_name](X_train, y_train)
            problem = combi_solving.Problem(variable_names=list(X_train), qualities=qualities)
            generator_func = getattr(syn_constraints, GENERATORS[generator_name]['func'])
            generator_args = {'problem': problem, **GENERATORS[generator_name]['args']}
            generator_args['num_iterations'] = n_iterations
            generator = generator_func(**generator_args)
            result = generator.evaluate_constraints()
            result['quality_name'] = quality_name
            result['split_idx'] = split_idx
            for model_name in model_names:
                model_dict = prediction_utility.MODELS[model_name]
                model = model_dict['func'](**model_dict['args'])
                if X_test is None:
                    performances = [prediction_utility.evaluate_prediction(
                        X_train=X_train[features], y_train=y_train, X_test=None,
                        y_test=None, model=model) for features in result['selected']]
                else:
                    performances = [prediction_utility.evaluate_prediction(
                        X_train=X_train[features], y_train=y_train, X_test=X_test[features],
                        y_test=y_test, model=model) for features in result['selected']]
                performances = pd.DataFrame(performances)
                performances.rename(columns={x: model_name + '_' + x for x in list(performances)}, inplace=True)
                result = pd.concat([result, performances], axis='columns')
            result.drop(columns='selected', inplace=True)
            results.append(result)
    results = pd.concat(results)
    results['dataset_name'] = dataset_name
    results['constraint_name'] = generator_name
    if results_dir is not None:
        data_utility.save_results(results, dataset_name=dataset_name,
                                  constraint_name=generator_name, directory=results_dir)
    return results


def pipeline(generator_names: Sequence[str], n_iterations: int, data_dir: pathlib.Path,
             quality_names: Sequence[str], model_names: Sequence[str] = None, n_splits: int = 1,
             n_processes: Optional[int] = None, results_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    datasets = [{'dataset_name': x, 'data_dir': data_dir, 'results_dir': results_dir}
                for x in data_utility.list_datasets(data_dir)]

    def update_progress(x: Any):
        progress_bar.update(n=1)

    progress_bar = tqdm.tqdm(total=len(generator_names) * len(datasets))

    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_constraint_type, kwds={
        **dataset_dict, 'generator_name': generator_name, 'n_iterations': n_iterations,
        'quality_names': quality_names, 'model_names': model_names, 'n_splits': n_splits},
        callback=update_progress) for generator_name in generator_names for dataset_dict in datasets]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    return pd.concat([x.get() for x in results])


# use of__ main__ around multiprocessing is required on Windows and recommended on Linux;
# prevents infinite recursion of spawning sub-processes
# furthermore, on Windows, whole file is imported, so everything outside main copied into sub-processes,
# while Linux sub-processes have access to all resources of the parent without copying
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates multiple constraint types on multiple datasets with one or more ' +
        'feature quality measures for each dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/demo/', dest='data_dir',
                        help='Directory for input data. Should contain datasets with two files each (X, y).')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/demo-results/', dest='results_dir',
                        help='Directory for output data. Is used for saving evaluation metrics.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    parser.add_argument('-i', '--iterations', type=int, default=1000, dest='n_iterations',
                        help='Number of repetitions for constraint generation (per constraint type and dataset).')
    parser.add_argument('-s', '--splits', type=int, default=10, dest='n_splits',
                        help='Number of splits used for prediction (at least 0).')
    parser.add_argument('-g', '--generators', type=str, nargs='+', dest='generator_names',
                        choices=list(GENERATORS.keys()), default=list(GENERATORS.keys()),
                        help='Constraint generators to be used.')
    parser.add_argument('-q', '--qualities', type=str, nargs='+', dest='quality_names',
                        choices=list(feature_qualities.QUALITIES.keys()),
                        default=list(feature_qualities.QUALITIES.keys()),
                        help='Feature qualities to be computed.')
    parser.add_argument('-m', '--models', type=str, nargs='*', dest='model_names',
                        choices=list(prediction_utility.MODELS.keys()),
                        default=list(prediction_utility.MODELS.keys()),
                        help='Prediction models to be used.')
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        print('Data directory does not exist.')
        sys.exit(1)
    if len(list(args.data_dir.glob('*'))) == 0:
        print('Data directory is empty.')
        sys.exit(1)
    if not args.results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        args.results_dir.mkdir(parents=True)
    if len(list(args.results_dir.glob('*'))) > 0:
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    results = pipeline(**vars(args))  # extract dict from Namspace and then unpack for call
    data_utility.save_results(results, directory=args.results_dir)
    print('Pipeline executed successfully.')
