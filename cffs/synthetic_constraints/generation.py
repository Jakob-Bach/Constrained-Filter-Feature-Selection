"""Constraint generators

Different generator classes for synthetic SMT constraints.
"""


from abc import ABCMeta, abstractmethod
import random
from typing import Optional, Sequence

import pandas as pd

import combi_expressions as expr
import combi_solving as solv


class ConstraintGenerator(metaclass=ABCMeta):

    def __init__(self, problem: solv.Problem, **kwargs):
        self.problem = problem
        self.min_num_constraints = kwargs.get('min_num_constraints', 1)
        self.max_num_constraints = kwargs.get('max_num_constraints', 1)
        self.min_num_variables = self.make_card_absolute(kwargs.get('min_num_variables', 2))
        self.max_num_variables = self.make_card_absolute(kwargs.get('max_num_variables', None))
        self.num_iterations = kwargs.get('num_iterations', 1)
        self.seed = 25

    @abstractmethod
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('Abstract method.')

    def evaluate_constraints(self) -> pd.DataFrame:
        random.seed(self.seed)
        results = []
        for _ in range(self.num_iterations):
            num_constraints = random.randint(self.min_num_constraints, self.max_num_constraints)
            for _ in range(num_constraints):
                num_variables = random.randint(self.min_num_variables, self.max_num_variables)
                selected_variables = random.sample(self.problem.get_variables(), k=num_variables)
                self.problem.add_constraint(self.generate(selected_variables))
            frac_solutions = self.problem.compute_solution_fraction()
            result = self.problem.optimize()
            result['num_constraints'] = num_constraints
            result['frac_solutions'] = frac_solutions
            results.append(result)
            self.problem.clear_constraints()
        return pd.DataFrame(results)

    def make_card_absolute(self, cardinality: float, pass_none: bool = False) -> int:
        max_cardinality = len(self.problem.get_variables())
        if cardinality is None:
            if pass_none:  # None might be used as default for different purposes
                return None
            cardinality = max_cardinality  # problem-specifc upper bound
        elif 0 < cardinality < 1:
            cardinality = round(cardinality * max_cardinality)  # turn absolute
        cardinality = int(cardinality)
        if (cardinality < 1) or (cardinality > max_cardinality):
            raise ValueError(f'Cardinality of {cardinality} is outside range [1,{max_cardinality}].')
        return cardinality


class AtLeastGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, global_at_most: Optional[int] = None,
                 cardinality: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        self.cardinality = self.make_card_absolute(cardinality, pass_none=True)
        self.global_at_most = self.make_card_absolute(global_at_most)

    # As at-least does not exclude the trivial solution (select everything), we
    # also add a global cardinality constraint
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        result = expr.AtLeast(variables, cardinality)
        if (self.problem.get_num_constraints() == 0) and (self.global_at_most < len(self.problem.get_variables())):
            global_at_most_constraint = expr.AtMost(self.problem.get_variables(), self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


class AtMostGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, cardinality: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        self.cardinality = self.make_card_absolute(cardinality, pass_none=True)

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        return expr.AtMost(variables, cardinality)


class GlobalAtMostGenerator(ConstraintGenerator):

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('evaluate_constraints() goes through all possible constraints anyway.')

    # For each cardinality, there is exactly one way to express a global AT-MOST (one elementary
    # AT-MOST covering all variables), so we just iterate from 1 to k-1 instead of doing repeated
    # evaluation to deal with randomness (which does not exist here)
    def evaluate_constraints(self) -> pd.DataFrame():
        results = []
        generator = AtMostGenerator(self.problem)
        generator.min_num_constraints = 1
        generator.max_num_constraints = 1
        generator.min_num_variables = len(self.problem.get_variables())
        generator.max_num_variables = len(self.problem.get_variables())
        generator.num_iterations = 1
        for cardinality in range(1, len(self.problem.get_variables())):
            generator.cardinality = cardinality
            results.append(generator.evaluate_constraints())
        return pd.concat(results, ignore_index=True)  # re-number the rows (else all have index 0)


class IffGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, global_at_most: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        self.global_at_most = self.make_card_absolute(global_at_most)

    # As iff does not exclude the trivial solution (select everything), we
    # also add a global cardinality constraint
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        result = expr.Iff(variables)
        if (self.problem.get_num_constraints() == 0) and (self.global_at_most < len(self.problem.get_variables())):
            global_at_most_constraint = expr.AtMost(self.problem.get_variables(), self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


class MixedGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, **kwargs):
        super().__init__(problem, **kwargs)
        self.generators = [
            AtLeastGenerator(problem),
            AtMostGenerator(problem),
            IffGenerator(problem),
            NandGenerator(problem),
            XorGenerator(problem)
        ]

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        return random.choice(self.generators).generate(variables)


class NandGenerator(ConstraintGenerator):

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        return expr.Not(expr.And(variables))


# Serves as a baseline
class NoConstraintGenerator(ConstraintGenerator):

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('evaluate_constraints() does not need this method.')

    # If we have no constraints, looping for different solutions does not make sense
    def evaluate_constraints(self) -> pd.DataFrame():
        result = self.problem.optimize()  # without constraints added
        result['num_constraints'] = 0
        result['frac_solutions'] = 1
        return pd.DataFrame([result])


class XorGenerator(ConstraintGenerator):

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        return expr.Xor(variables[0], variables[1])
