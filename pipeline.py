"""Synthetic constraints pipeline

Main script for our experiments with synthetic constraints.

Usage: python pipeline.py --help
"""

import argparse
import multiprocessing
import pathlib
import sys
from typing import Any, Dict, Optional

import pandas as pd
import tqdm

import combi_expressions
import combi_solving
from data_utility import list_datasets, load_qualities, save_results
import generation


def evaluate_constraint_type(
        constraint_type: str, generator_func: str, generator_args: Dict[str, Any], dataset: str,
        data_dir: pathlib.Path, results_dir: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    quality_table = load_qualities(dataset_name=dataset, directory=data_dir)
    quality_table.drop('Feature', axis='columns', inplace=True)
    results = []
    for quality_type in list(quality_table):
        qualities = quality_table[quality_type].to_list()
        variables = [combi_expressions.Variable(name='Feature_' + str(i)) for i in range(len(qualities))]
        problem = combi_solving.Problem(variables=variables, qualities=qualities)
        generator_func = getattr(generation, generator_func)  # get the function object
        generator = generator_func(**{'problem': problem, **generator_args})
        result = generator.evaluate_constraints()
        result['quality_type'] = quality_type
        results.append(result)
    result = pd.concat(results)
    result['dataset'] = dataset
    result['constraint_type'] = constraint_type
    if results_dir is not None:
        save_results(result, dataset_name=dataset, constraint_type=constraint_type, directory=results_dir)
    return result


def pipeline(data_dir: pathlib.Path, n_processes: Optional[int] = None,
             results_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    common_generator_args = {'num_repetitions': 10, 'min_num_constraints': 1, 'max_num_constraints': 10}
    generators = [
        {'constraint_type': 'group-AT-LEAST', 'generator_func': 'AtLeastGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10}},
        {'constraint_type': 'group-AT-MOST', 'generator_func': 'AtMostGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'global-AT-MOST', 'generator_func': 'GlobalAtMostGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'single-IFF', 'generator_func': 'IffGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10}},
        {'constraint_type': 'group-IFF', 'generator_func': 'IffGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10, 'max_num_variables': 5}},
        {'constraint_type': 'single-NAND', 'generator_func': 'NandGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'group-NAND', 'generator_func': 'NandGenerator',
         'generator_args': {**common_generator_args, 'max_num_variables': 5}},
        {'constraint_type': 'single-XOR', 'generator_func': 'XorGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'MIXED', 'generator_func': 'MixedGenerator',
         'generator_args': common_generator_args}
    ]
    datasets = [{'dataset': x, 'data_dir': data_dir, 'results_dir': results_dir} for x in list_datasets(data_dir)]

    def update_progress(x: Any):
        progress_bar.update(n=1)

    progress_bar = tqdm.tqdm(total=len(generators) * len(datasets))

    process_pool = multiprocessing.Pool(processes=n_processes)
    results = [process_pool.apply_async(evaluate_constraint_type, kwds={**x, **y}, callback=update_progress)
               for x in generators for y in datasets]
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
        'feature quality measures for each dataset. Stores the result in the same directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=pathlib.Path, default='.', dest='data_dir',
                        help='Directory for input data. Should contain datasets (X, y) and dataset qualities.')
    parser.add_argument('-r', '--results', type=pathlib.Path, default='.', dest='results_dir',
                        help='Directory for output data. Is used for saving evaluation metrics.')
    parser.add_argument('-p', '--processes', type=int, default=None, dest='n_processes',
                        help='Number of processes for multi-processing.')
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
    save_results(results, directory=args.results_dir)
    print('Pipeline exectued successfully.')
