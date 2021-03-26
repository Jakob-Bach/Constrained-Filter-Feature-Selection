"""Experimental pipeline for the study with synthetic constraints

Main script (experimental pipeline) for our study with synthetic constraints.
Should be run after preparing one or more dataset(s).

Usage: python -m cffs.synthetic_constraints.syn_pipeline --help
"""

import argparse
import multiprocessing
import pathlib
from typing import Any, Optional

import pandas as pd
import tqdm

from cffs.core import combi_solving
from cffs.utilities import data_utility
from cffs.utilities import feature_qualities
from cffs.utilities import prediction_utility
from cffs.synthetic_constraints import syn_constraints


COMMON_GENERATOR_ARGS = {'num_iterations': 1000, 'min_num_constraints': 1, 'max_num_constraints': 10}

GENERATORS = {  # constraint types
    'Global-AT-MOST': {'func': 'GlobalAtMostGenerator', 'args': COMMON_GENERATOR_ARGS},  # args will have no effect
    'Group-AT-MOST': {'func': 'AtMostGenerator', 'args': COMMON_GENERATOR_ARGS},
    'Group-AT-LEAST': {'func': 'AtLeastGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5}},
    'Single-IFF': {'func': 'IffGenerator',
                   'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5, 'max_num_variables': 2}},
    'Group-IFF': {'func': 'IffGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'global_at_most': 0.5}},
    'Single-NAND': {'func': 'NandGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'max_num_variables': 2}},
    'Group-NAND': {'func': 'NandGenerator', 'args': COMMON_GENERATOR_ARGS},
    'Single-XOR': {'func': 'XorGenerator', 'args': {**COMMON_GENERATOR_ARGS, 'max_num_variables': 2}},
    'Group-MIXED': {'func': 'MixedGenerator', 'args': COMMON_GENERATOR_ARGS},
    'UNCONSTRAINED': {'func': 'UnconstrainedGenerator', 'args': COMMON_GENERATOR_ARGS}  # args will have not effect
}


# Evaluate one constraint type (denoted by "generator_name") on one dataset (denoted by
# "dataset_name", stored in "data_dir"). Iterate over splits of the dataset (depending on
# "n_splits") and a (hard-coded) list of feature-quality measures. Repeat constraint generation
# according to "n_iterations". To evaluate one iteration of constraint generation (which yields
# one feature set), iterate over a (hard-coded) list of prediction models. If "results_dir" is set,
# save a data frame with the evaluation results.
def evaluate_constraint_type(
        generator_name: str, dataset_name: str, data_dir: pathlib.Path, results_dir: Optional[pathlib.Path] = None,
        n_iterations: int = 1000, n_splits: int = 1) -> pd.DataFrame:
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
        for quality_name, quality_func in feature_qualities.QUALITIES.items():
            qualities = quality_func(X_train, y_train)
            problem = combi_solving.Problem(variable_names=list(X_train), qualities=qualities)
            generator_func = getattr(syn_constraints, GENERATORS[generator_name]['func'])
            generator_args = {'problem': problem, **GENERATORS[generator_name]['args']}
            generator_args['num_iterations'] = n_iterations
            generator = generator_func(**generator_args)
            result = generator.evaluate_constraints()  # a data frame, one row per iteration of generation
            for model_name, model_dict in prediction_utility.MODELS.items():
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
            result['split_idx'] = split_idx
            result['quality_name'] = quality_name
            results.append(result)
    results = pd.concat(results)
    results['constraint_name'] = generator_name
    results['dataset_name'] = dataset_name
    if results_dir is not None:
        data_utility.save_results(results, dataset_name=dataset_name,
                                  constraint_name=generator_name, directory=results_dir)
    return results


# Evaluate multiple (hard-coded) constraint types on multiple datasets (stored in "data_dir").
# Optionally, save each dataset-type combination separately as a file in "results_dir". However,
# a data frame with all results is also returned.
# You can vary the number of iterations in constraint generation, the number of splits for
# each dataset, and the number of cores used for parallelization.
def pipeline(data_dir: pathlib.Path, results_dir: Optional[pathlib.Path] = None,
             n_iterations: int = 1000, n_splits: int = 1, n_processes: Optional[int] = None) -> pd.DataFrame:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if len(list(data_dir.glob('*'))) == 0:
        raise FileNotFoundError('Data directory is empty.')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if len(list(results_dir.glob('*'))) > 0:
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    datasets = [{'dataset_name': x, 'data_dir': data_dir, 'results_dir': results_dir}
                for x in data_utility.list_datasets(data_dir)]

    def update_progress(unused: Any):
        progress_bar.update(n=1)

    progress_bar = tqdm.tqdm(total=len(GENERATORS) * len(datasets))

    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_constraint_type, kwds={
        **dataset_dict, 'generator_name': generator_name, 'n_iterations': n_iterations, 'n_splits': n_splits},
        callback=update_progress) for generator_name in GENERATORS.keys() for dataset_dict in datasets]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    return pd.concat([x.get() for x in results])


# Parse some command-line arguments, run the pipeline, and save the results.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates generated constraints from multiple constraint types on arbitary' +
        'datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/openml/', dest='data_dir',
                        help='Directory with input data. Should contain datasets with two files each (X, y).')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/openml-results/', dest='results_dir',
                        help='Directory for output data. Is used for saving evaluation metrics.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    parser.add_argument('-i', '--iterations', type=int, default=1000, dest='n_iterations',
                        help='Number of repetitions for constraint generation (per constraint typ, dataset and split).')
    parser.add_argument('-s', '--splits', type=int, default=10, dest='n_splits',
                        help='Number of splits used for evaluating predictions (at least 0).')
    args = parser.parse_args()
    print('Pipeline started.')
    pipeline_results = pipeline(**vars(args))  # extract dict from Namspace and then unpack for call
    data_utility.save_results(pipeline_results, directory=args.results_dir)
    print('Pipeline executed successfully.')
