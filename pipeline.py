"""Synthetic constraints pipeline

Main script for our experiments with synthetic constraints.
"""

import combi_expressions
import combi_solving
import generation


def evaluate(constraint_type, generator_func, generator_args, dataset, qualities):
    variables = [combi_expressions.Variable(name='Feature_' + str(i)) for i in range(len(qualities))]
    problem = combi_solving.Problem(variables=variables, qualities=qualities)
    generator_func = getattr(generation, generator_func)  # get the function object
    generator = generator_func(**{'problem': problem, **generator_args})
    result = generator.evaluate_constraints()
    result['dataset'] = dataset
    result['constraint_type'] = constraint_type
    return result


# use of__ main__ around multiprocessing is required on Windows and recommended on Linux;
# prevents infinite recursion of spawning sub-processes
# furthermore, on Windows, whole file is imported, so everything outside main copied into sub-processes,
# while Linux sub-processes have access to all resources of the parent without copying
if __name__ == '__main__':
    from multiprocessing import Pool

    import pandas as pd
    from sklearn.datasets import load_boston
    from tqdm import tqdm

    dataset = load_boston()
    features = dataset['feature_names']
    X = pd.DataFrame(dataset['data'], columns=features)
    y = pd.Series(dataset['target'])
    qualities = [round(abs(X[feature].corr(y)), 2) for feature in features]
    datasets = [{'dataset': 'boston', 'qualities': qualities}]

    common_generator_args = {'num_repetitions': 10, 'min_num_constraints': 1, 'max_num_constraints': 10}
    generators = [
        {'constraint_type': 'group_AT_LEAST', 'generator_func': 'AtLeastGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10}},
        {'constraint_type': 'group_AT_MOST', 'generator_func': 'AtMostGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'global_AT_MOST', 'generator_func': 'GlobalAtMostGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'single_IFF', 'generator_func': 'IffGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10}},
        {'constraint_type': 'group_IFF', 'generator_func': 'IffGenerator',
         'generator_args': {**common_generator_args, 'global_at_most': 10, 'max_num_variables': 5}},
        {'constraint_type': 'single_NAND', 'generator_func': 'NandGenerator',
         'generator_args': common_generator_args},
        {'constraint_type': 'group_NAND', 'generator_func': 'NandGenerator',
         'generator_args': {**common_generator_args, 'max_num_variables': 5}},
        {'constraint_type': 'single_XOR', 'generator_func': 'XorGenerator',
         'generator_args': common_generator_args}
    ]

    def update_progress(x):
        progress_bar.update(n=1)

    progress_bar = tqdm(total=len(generators))

    process_pool = Pool()
    results = [process_pool.apply_async(evaluate, kwds={**x, **y}, callback=update_progress)
               for x in generators for y in datasets]
    process_pool.close()
    process_pool.join()
    progress_bar.close()
    results = pd.concat([x.get() for x in results])
