"""Materials science pipeline

Prediction script for our case study in materials science.

Usage: python ms_pipeline.py --help
"""


import argparse
import multiprocessing
import pathlib
import sys
import time
from typing import Any, Optional

import pandas as pd
import tqdm

from cffs.core import combi_solving
from cffs.materials_science import ms_constraints
from cffs.utilities import data_utility
from cffs.utilities import feature_qualities
from cffs.utilities import prediction_utility


BASE_EVALUATORS = {
    'UNCONSTRAINED': {'func': 'NoConstraintEvaluator', 'args': {}},
    'SelectSchmidGroup': {'func': 'SelectSchmidGroupEvaluator', 'args': {}},
    'SelectQuantitySchmidGroup': {'func': 'SelectQuantitySchmidGroupEvaluator', 'args': {}},
    'SelectSchmidGroupRepresentative': {'func': 'SelectSchmidGroupRepresentativeEvaluator', 'args': {}},
    'SelectQuantitySchmidGroupRepresentative': {'func': 'SelectQuantitySchmidGroupRepresentativeEvaluator', 'args': {}},
    # 'SelectWholeSlipSystems': {'func': 'SelectWholeSlipSystemsEvaluator', 'args': {}},  # does not combine well with cardinality
    # 'SelectReactionType': {'func': 'SelectReactionTypeEvaluator', 'args': {}},  # we have removed reaction types from features
    # 'SelectValueOrDelta': {'func': 'SelectValueOrDeltaEvaluator', 'args': {}},  # we use no delta features for predicting absolute quantities
    'SelectStrainTensor': {'func': 'SelectStrainTensorEvaluator', 'args': {}},
    'SelectDislocationDensity': {'func': 'SelectDislocationDensityEvaluator', 'args': {}},
    'SelectStrainRate': {'func': 'SelectStrainRateEvaluator', 'args': {}},
    'SelectAggregate': {'func': 'SelectAggregateEvaluator', 'args': {}},
    'SelectQuantityAggregate': {'func': 'SelectQuantityAggregateEvaluator', 'args': {}},
    'SelectAggregateOrOriginal': {'func': 'SelectAggregateOrOriginalEvaluator', 'args': {}}
}
# Combine all base evaluators with a global cardinality constraint and quality threshold
CARDINALITIES = [5, 10]
DROP_LOW_QUALITY_THRESHOLD = 0.2
EVALUATORS = {}
for cardinality in CARDINALITIES:
    for evaluator_name, evaluator_info in BASE_EVALUATORS.items():
        EVALUATORS[evaluator_name + '_k' + cardinality] = {'func': 'CombinedEvaluator', 'args': {'evaluators': {
            getattr(ms_constraints, evaluator_info['func']): evaluator_info['args'],  # base evaluator
            ms_constraints.GlobalAtMostEvaluator: {'global_at_most': cardinality},
            ms_constraints.QualityThresholdEvaluator: {'threshold': DROP_LOW_QUALITY_THRESHOLD}
        }}}  # "evaluators" is a dict of evaluator type and initialization arguments

DROP_CORRELATION_THRESHOLD = None  # number in [0,1] or None


def evaluate_constraints(evaluator_name: str, dataset_name: str, data_dir: pathlib.Path) -> pd.DataFrame:
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
    for quality_name in feature_qualities.QUALITIES.keys():
        qualities = feature_qualities.QUALITIES[quality_name](X_train, y_train)
        problem = combi_solving.Problem(variable_names=list(X_train), qualities=qualities)
        evaluator_func = getattr(ms_constraints, EVALUATORS[evaluator_name]['func'])
        evaluator_args = {'problem': problem, **EVALUATORS[evaluator_name]['args']}
        evaluator = evaluator_func(**evaluator_args)
        start_time = time.process_time()
        result = evaluator.evaluate_constraints()  # a dict
        end_time = time.process_time()
        for model_name in prediction_utility.MODELS.keys():
            model_dict = prediction_utility.MODELS[model_name]
            model = model_dict['func'](**model_dict['args'])
            performances = prediction_utility.evaluate_prediction(
                X_train=X_train[result['selected']], y_train=y_train,
                X_test=X_test[result['selected']], y_test=y_test, model=model)
            for key, value in performances.items():  # multiple eval metrics might be used
                result[f'{model_name}_{key}'] = value
        result['constraint_name'] = evaluator_name
        result['quality_name'] = quality_name
        result['evaluation_time'] = end_time - start_time
        results.append(result)
    results = pd.DataFrame(results)
    results['dataset_name'] = dataset_name
    return results


def pipeline(data_dir: pathlib.Path, results_dir: Optional[pathlib.Path] = None,
             n_processes: Optional[int] = None) -> pd.DataFrame:
    datasets = [{'dataset_name': x, 'data_dir': data_dir} for x in data_utility.list_datasets(data_dir)]

    def update_progress(unused: Any):
        progress_bar.update(n=1)

    progress_bar = tqdm.tqdm(total=len(EVALUATORS) * len(datasets))

    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_constraints, kwds={
        **dataset_dict, 'evaluator_name': evaluator_name}, callback=update_progress)
        for evaluator_name in EVALUATORS.keys() for dataset_dict in datasets]
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
