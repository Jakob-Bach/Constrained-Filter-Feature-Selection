"""Materials science pipeline

Prediction script for our case study in materials science.

Usage: python ms_pipeline.py --help
"""


import argparse
import multiprocessing
import pathlib
import sys
from typing import Any, Optional, Sequence

import pandas as pd
import tqdm

from cffs.core import combi_solving
from cffs.materials_science import ms_constraints
from cffs.utilities import data_utility
from cffs.utilities import feature_qualities
from cffs.utilities import prediction_utility


EVALUATORS = {
    'Schmid-100': {'func': 'SchmidFactor100Evaluator', 'args': dict()},
    'UNCONSTRAINED': {'func': 'NoConstraintEvaluator', 'args': dict()}
}

DROP_CORRELATION_THRESHOLD = None  # number in [0,1] or None


def evaluate_constraints(
        evaluator_name: str, dataset_name: str, data_dir: pathlib.Path,
        quality_names: Sequence[str], model_names: Sequence[str] = None) -> pd.DataFrame:
    if model_names is None:
        model_names = []  # this is more Pythonic than using an empty list as default
    results = []
    X, y = data_utility.load_dataset(dataset_name=dataset_name, directory=data_dir)
    max_train_time = X['time'].quantile(q=0.8)
    X_train = X[X['time'] <= max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time'])
    y_train = y[X['time'] <= max_train_time]
    X_test = X[X['time'] > max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time'])
    y_test = y[X['time'] > max_train_time]
    if (len(X_train) == 0) or (len(X_test) == 0):
        return None
    X_train, X_test = prediction_utility.drop_correlated_features(
        X_train=X_train, X_test=X_test, threshold=DROP_CORRELATION_THRESHOLD)
    for quality_name in quality_names:
        qualities = feature_qualities.QUALITIES[quality_name](X_train, y_train)
        problem = combi_solving.Problem(variable_names=list(X_train), qualities=qualities)
        evaluator_func = getattr(ms_constraints, EVALUATORS[evaluator_name]['func'])
        evaluator_args = {'problem': problem, **EVALUATORS[evaluator_name]['args']}
        evaluator = evaluator_func(**evaluator_args)
        result = evaluator.evaluate_constraints()  # a dict
        for model_name in model_names:
            model_dict = prediction_utility.MODELS[model_name]
            model = model_dict['func'](**model_dict['args'])
            performances = prediction_utility.evaluate_prediction(
                X_train=X_train[result['selected']], y_train=y_train,
                X_test=X_test[result['selected']], y_test=y_test, model=model)
            for key, value in performances.items():  # multiple eval metrics might be used
                result[f'{model_name}_{key}'] = value
        result.pop('selected')
        result['constraint_name'] = evaluator_name
        result['quality_name'] = quality_name
        results.append(result)
    results = pd.DataFrame(results)
    results['dataset_name'] = dataset_name
    return results


def pipeline(evaluator_names: Sequence[str], data_dir: pathlib.Path, quality_names: Sequence[str],
             model_names: Sequence[str] = None, n_processes: Optional[int] = None,
             results_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    datasets = [{'dataset_name': x, 'data_dir': data_dir} for x in data_utility.list_datasets(data_dir)]

    def update_progress(x: Any):
        progress_bar.update(n=1)

    progress_bar = tqdm.tqdm(total=len(evaluator_names) * len(datasets))

    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_constraints, kwds={
        **dataset_dict, 'evaluator_name': evaluator_name, 'quality_names': quality_names,
        'model_names': model_names}, callback=update_progress)
        for evaluator_name in evaluator_names for dataset_dict in datasets]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    return pd.concat([x.get() for x in results])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates multiple constraint types on multiple datasets with one or more ' +
        'feature quality measures for each dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/ms/', dest='data_dir',
                        help='Directory for input data. Should contain datasets with two files each (X, y).')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/ms-results/', dest='results_dir',
                        help='Directory for output data. Is used for saving evaluation metrics.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    parser.add_argument('-e', '--evaluators', type=str, nargs='+', dest='evaluator_names',
                        choices=list(EVALUATORS.keys()), default=list(EVALUATORS.keys()),
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
