"""Experimental pipeline for the case study in materials science

Main script (experimental pipeline) for our case study in materials science.
Should be run after preparing a dataset.

Usage: python -m materials_science.ms_pipeline --help
"""

import argparse
import multiprocessing
import pathlib
import time
from typing import Any, Optional

import pandas as pd
import tqdm

from cffs import combi_solving
from cffs import feature_qualities
from materials_science import ms_constraints
from utilities import data_utility
from utilities import prediction_utility


FEATURE_QUALITIES = {'abs_corr': feature_qualities.abs_corr}

BASE_EVALUATORS = {  # domain-specific constraint types
    'Schmid-group': {'func': 'SchmidGroupEvaluator', 'args': {}},
    'Quantity-Schmid-group': {'func': 'QuantitySchmidGroupEvaluator', 'args': {}},
    'Schmid-group-representative': {'func': 'SchmidGroupRepresentativeEvaluator', 'args': {}},
    'Quantity-Schmid-group-representative': {'func': 'QuantitySchmidGroupRepresentativeEvaluator', 'args': {}},
    'Plastic-strain-tensor': {'func': 'PlasticStrainTensorEvaluator', 'args': {}},
    'Dislocation-density': {'func': 'DislocationDensityEvaluator', 'args': {}},
    'Plastic-strain-rate': {'func': 'PlasticStrainRateEvaluator', 'args': {}},
    'Aggregate': {'func': 'AggregateEvaluator', 'args': {}},
    'Quantity-aggregate': {'func': 'QuantityAggregateEvaluator', 'args': {}},
    'Aggregate-or-original': {'func': 'AggregateOrOriginalEvaluator', 'args': {}},
    'Mixed': {'func': 'CombinedEvaluator', 'args': {'evaluators': {
        ms_constraints.QuantitySchmidGroupRepresentativeEvaluator: {},
        ms_constraints.PlasticStrainTensorEvaluator: {},
        ms_constraints.DislocationDensityEvaluator: {},
        ms_constraints.PlasticStrainRateEvaluator: {},
        ms_constraints.QuantityAggregateEvaluator: {},
        ms_constraints.AggregateOrOriginalEvaluator: {}
    }}},
    'Unconstrained': {'func': 'UnconstrainedEvaluator', 'args': {}}
}
# Combine all base evaluators with a global-cardinality constraint and quality-filter constraint
CARDINALITIES = [5, 10]
QUALITY_FILTER_THRESHOLD = 0.2
EVALUATORS = {}
for cardinality in CARDINALITIES:
    for evaluator_name, evaluator_info in BASE_EVALUATORS.items():
        EVALUATORS[f'{evaluator_name}_k{cardinality}'] = {'func': 'CombinedEvaluator', 'args': {'evaluators': {
            getattr(ms_constraints, evaluator_info['func']): evaluator_info['args'],  # base evaluator
            ms_constraints.GlobalCardinalityEvaluator: {'global_at_most': cardinality},
            ms_constraints.QualityFilterEvaluator: {'threshold': QUALITY_FILTER_THRESHOLD}
        }}}  # "evaluators" is a dict of evaluator type and initialization arguments
INTER_CORRELATION_THRESHOLD = 0.8  # number in [0,1] or None; evaluator added below, depends on X


# Evaluate one constraint type (denoted by "evaluator_name") on one dataset (denoted by
# "dataset_name", stored in "data_dir"). Split the dataset and iterate over a (hard-coded) list
# of feature-quality measures. To evaluate a feature set, iterate over a (hard-coded) list of
# prediction models. Return a data frame with the evaluation results.
def evaluate_constraints(evaluator_name: str, dataset_name: str, data_dir: pathlib.Path) -> pd.DataFrame:
    results = []
    X, y = data_utility.load_dataset(dataset_name=dataset_name, directory=data_dir)
    max_train_time = X['time'].quantile(q=0.8)
    X_train = X[X['time'] <= max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time', 'step'])
    y_train = y[X['time'] <= max_train_time]
    X_test = X[X['time'] > max_train_time].drop(columns=['pos_x', 'pos_y', 'pos_z', 'time', 'step'])
    y_test = y[X['time'] > max_train_time]
    if (len(X_train) == 0) or (len(X_test) == 0):
        raise RuntimeError('Splitting caused empty training or empty test set.')
    for quality_name, quality_func in FEATURE_QUALITIES.items():
        qualities = quality_func(X_train, y_train)
        problem = combi_solving.Problem(variable_names=list(X_train), qualities=qualities)
        evaluator_func = getattr(ms_constraints, EVALUATORS[evaluator_name]['func'])
        evaluator_args = {'problem': problem, **EVALUATORS[evaluator_name]['args']}
        if INTER_CORRELATION_THRESHOLD is not None:
            evaluator_args['evaluators'][ms_constraints.InterCorrelationEvaluator] = {
                'corr_df': X_train.corr().abs(), 'threshold': INTER_CORRELATION_THRESHOLD}
        evaluator = evaluator_func(**evaluator_args)
        start_time = time.process_time()
        result = evaluator.evaluate_constraints()  # a dict, as just one evaluation
        end_time = time.process_time()
        for model_name, model_dict in prediction_utility.MODELS.items():
            model = model_dict['func'](**model_dict['args'])
            performances = prediction_utility.evaluate_prediction(
                X_train=X_train[result['selected']], y_train=y_train,
                X_test=X_test[result['selected']], y_test=y_test, model=model)
            for metric_name, metric_value in performances.items():  # multiple metrics possible
                result[f'{model_name}_{metric_name}'] = metric_value
        result['evaluation_time'] = end_time - start_time
        result['split_idx'] = 0  # for consistency to other study; but we only do one split here
        result['quality_name'] = quality_name
        result['constraint_name'] = evaluator_name
        results.append(result)
    results = pd.DataFrame(results)
    results['dataset_name'] = dataset_name
    return results


# Evaluate multiple (hard-coded) constraint types on multiple datasets (stored in "data_dir").
# A data frame with all results is returned.
# You can set the number of cores used for parallelization.
def pipeline(data_dir: pathlib.Path, n_processes: Optional[int] = None) -> pd.DataFrame:
    if not data_dir.is_dir():
        raise FileNotFoundError('Data directory does not exist.')
    if len(list(data_dir.glob('*'))) == 0:
        raise FileNotFoundError('Data directory is empty.')
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


# Parse some command-line arguments, run the pipeline, and save the results.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates multiple types of manually-defined constraints on one or more ' +
        'materials-science datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='data/ms/', dest='data_dir',
                        help='Directory with input data. Should contain datasets with two files each (X, y).')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='data/ms-results/', dest='results_dir',
                        help='Directory for output data. Is used for saving evaluation metrics.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing (default: all cores).')
    args = vars(parser.parse_args())  # extract dict from Namspace
    results_dir = args.pop('results_dir')
    if not results_dir.is_dir():
        print('Results directory does not exist. We create it.')
        results_dir.mkdir(parents=True)
    if len(list(results_dir.glob('*'))) > 0:
        print('Results directory is not empty. Files might be overwritten, but not deleted.')
    print('Pipeline started.')
    pipeline_results = pipeline(**args)
    data_utility.save_results(pipeline_results, directory=results_dir)
    print('Pipeline executed successfully.')
