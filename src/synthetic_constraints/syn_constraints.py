"""Constraints for the study with synthetic constraints

Classes representing generators of certain types of constraints.
Each class is able to evaluate its constraint type on a given feature-selection problem.
"""

from abc import ABCMeta, abstractmethod
import random
from typing import Optional, Sequence

import pandas as pd

from cffs import combi_expressions as expr
from cffs import combi_solving as solv


# Super-class containing the generation and evaluation procedure for constraints, without defining
# concrete constraints (that is up to the sub-classes).
class ConstraintGenerator(metaclass=ABCMeta):

    # Initialize generator. There are several parameters controlling the generation process,
    # which are passed from the experimental pipeline. To keep the parameter list short,
    # we use **kwargs.
    def __init__(self, problem: solv.Problem, **kwargs):
        self.problem = problem
        self.min_num_constraints = kwargs.get('min_num_constraints', 1)
        self.max_num_constraints = kwargs.get('max_num_constraints', 1)
        self.min_num_variables = self.make_card_absolute(kwargs.get('min_num_variables', 2))
        self.max_num_variables = self.make_card_absolute(kwargs.get('max_num_variables', None))
        self.num_iterations = kwargs.get('num_iterations', 1)

    # Sub-classes should implement this method by generating a boolean expression as constraint.
    # For generation, you should apply logical and/or arithmetic operators to the passed
    # "variables", which serve as operands.
    # This method will be called as a sub-routine from the main evaluation procedure.
    @abstractmethod
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('Abstract method.')

    # Systematically generate constraints by repeatedly picking the number of constraints, the
    # number of variables involved, and the actual variables involved uniformly at random.
    # Solve the optimization problem of constrained feature selection.
    # Return a DataFrame with the core evaluation metrics.
    def evaluate_constraints(self) -> pd.DataFrame:
        random.seed(25)
        results = []
        for _ in range(self.num_iterations):
            num_constraints = random.randint(self.min_num_constraints, self.max_num_constraints)
            for _ in range(num_constraints):
                num_variables = random.randint(self.min_num_variables, self.max_num_variables)
                selected_variables = random.sample(self.problem.get_variables(), k=num_variables)
                self.problem.add_constraint(self.generate(selected_variables))
            frac_solutions = self.problem.compute_solution_fraction()
            constrained_variables = self.problem.get_constrained_variables()
            unique_constrained_variables = set(constrained_variables)  # remove duplicates
            result = self.problem.optimize()  # returns dictionary with some evaluation metrics
            result['num_variables'] = len(self.problem.get_variables())
            result['num_constrained_variables'] = len(constrained_variables)
            result['num_unique_constrained_variables'] = len(unique_constrained_variables)
            result['num_constraints'] = num_constraints
            result['frac_solutions'] = frac_solutions
            results.append(result)
            self.problem.clear_constraints()  # iterations should be independent from each other
        return pd.DataFrame(results)

    # Make sure cardinality is an absolute number by converting fractions and None (the latter
    # might also remain as such if "pass_none" is chosen). Fractions are interpreted relative to
    # the number of variables. None without "pass_none" evaluates to the number of variables.
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


# Generator for Group-AT-LEAST cardinality constraints (combined with a global AT-MOST constraint).
class AtLeastGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, global_at_most: Optional[int] = None,
                 cardinality: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        # For "cardinality", None denotes that it should be picked at random:
        self.cardinality = self.make_card_absolute(cardinality, pass_none=True)
        # In contrast, global cardinality bound is set to number of variables if None
        self.global_at_most = self.make_card_absolute(global_at_most)

    # As AT-LEAST does not exclude the trivial solution (select all variables), we also add a
    # global cardinality constraint.
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        result = expr.AtLeast(variables, cardinality)
        # If this constraint is the first, AND it with a Gloabl-AT-MOST constraint (AND makes sure
        # the number of constraints is not increased by two in one call of generate()).
        if (self.problem.get_num_constraints() == 0) and (self.global_at_most < len(self.problem.get_variables())):
            global_at_most_constraint = expr.AtMost(self.problem.get_variables(), self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


# Generator for Group-AT-MOST cardinality constraints.
class AtMostGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, cardinality: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        # For "cardinality", None denotes that it should be picked at random:
        self.cardinality = self.make_card_absolute(cardinality, pass_none=True)

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        return expr.AtMost(variables, cardinality)


# Generator for Global-AT-MOST cardinality constraints. Deviates from the usual iterated
# generation + evaluation procedure (thus, initialization args are ignored).
class GlobalAtMostGenerator(ConstraintGenerator):

    # Not used, but needs to be implemented so the class can be instantiated.
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('Not necessary for evaluation.')

    # For each cardinality, there is exactly one way to express a global AT-MOST (one elementary
    # AT-MOST covering all variables), so we just iterate from 1 to n-1 instead of doing repeated
    # evaluation, as there is no randomness we would mediate with repetition.
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
            results.append(generator.evaluate_constraints())  # one-row data frame for each cardinality
        return pd.concat(results, ignore_index=True)  # re-number the rows (else all have index 0)


# Generator for (Single-/Group-)IFF constraints (combined with a global AT-MOST constraint).
class IffGenerator(ConstraintGenerator):

    def __init__(self, problem: solv.Problem, global_at_most: Optional[int] = None, **kwargs):
        super().__init__(problem, **kwargs)
        self.global_at_most = self.make_card_absolute(global_at_most)

    # As IFF does not exclude the trivial solution (select all variables), we also add a global
    # cardinality constraint.
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        result = expr.Iff(variables)
        # If this constraint is the first, AND it with a Gloabl-AT-MOST constraint (AND makes sure
        # the number of constraints is not increased by two in one call of generate()).
        if (self.problem.get_num_constraints() == 0) and (self.global_at_most < len(self.problem.get_variables())):
            global_at_most_constraint = expr.AtMost(self.problem.get_variables(), self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


# Generator mixing various constraint types, picking randomly between them.
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


# Generator for (Single-/Group-)NAND constraints.
class NandGenerator(ConstraintGenerator):

    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        return expr.Not(expr.And(variables))


# Generator for empty set of constraints (type "Unconstrained"). Serves as a baseline. Deviates
# from the usual iterated generation + evaluation procedure (thus, initialization args are ignored).
class UnconstrainedGenerator(ConstraintGenerator):

    # If we have no constraints, there is no randomness, so doing repetitions does not make sense.
    # The number of constraints is fixed. We can safely ignore the other generation parameters,
    # so calling the super initializer is unnecessary.
    def __init__(self, problem: solv.Problem, **kwargs):
        self.problem = problem
        self.num_iterations = 1
        self.min_num_constraints = 0
        self.max_num_constraints = 0

    # Not used, but needs to be implemented so the class can be instantiated.
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        raise NotImplementedError('Not necessary for evaluation.')


# Generator for Single-XOR constraints.
class XorGenerator(ConstraintGenerator):

    # Only use two variables, no matter how many are passed.
    def generate(self, variables: Sequence[expr.Variable]) -> expr.BooleanExpression:
        return expr.Xor(variables[0], variables[1])
