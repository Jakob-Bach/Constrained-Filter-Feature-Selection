"""
Different generation functions for synthetic constraints.
"""

import random

import pandas as pd

import combi_expressions as expr


class Constraint_Generator:

    def __init__(self, problem):
        self.problem = problem
        self.min_num_constraints = 1
        self.max_num_constraints = 1
        self.min_num_variables = 2
        self.max_num_variables = 2
        self.num_repetitions = 1
        self.seed = 25

    def generate(self, variables):
        pass

    def evaluate_constraints(self):
        random.seed(self.seed)
        results = []
        for i in range(self.num_repetitions):
            num_constraints = random.randint(self.min_num_constraints, self.max_num_constraints)
            for j in range(num_constraints):
                num_variables = random.randint(self.min_num_variables, self.max_num_variables)
                selected_variables = random.sample(self.problem.get_variables(), k=num_variables)
                self.problem.add_constraint(self.generate(selected_variables))
            frac_solutions = self.problem.compute_solution_fraction()
            result = self.problem.optimize()
            result['num_constraints'] = num_constraints
            result['frac_solutions'] = frac_solutions
            self.problem.clear_constraints()
        return pd.DataFrame(results)


class At_Least_Generator(Constraint_Generator):

    def __init__(self, problem, global_at_most, cardinality=None):
        super().__init__(problem)
        self.cardinality = cardinality
        self.global_at_most = global_at_most

    # As at_most does not exclude the trivial solution (select everything), we
    # also add a global cardinality constraint
    def generate(self, variables):
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        result = expr.Weighted_Sum_Lt_Eq(variables, [1] * len(variables), cardinality)
        if self.problem.num_constraints == 0:
            global_at_most_constraint = expr.Weighted_Sum_Lt_Eq(
                self.problem.get_variables(), [1] * len(self.problem.get_variables()),
                self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


class At_Most_Generator(Constraint_Generator):

    def __init__(self, cardinality=None):
        self.cardinality = cardinality

    def generate(self, variables):
        if self.cardinality is None:
            cardinality = random.randint(1, len(variables) - 1)
        else:
            cardinality = self.cardinality
        return expr.Weighted_Sum_Lt_Eq(variables, [1] * len(variables), cardinality)


class Global_At_Most_Generator(Constraint_Generator):

    # For each cardinality, there is exactly one way to express the constraint,
    # so we iterate over cardinalities without repetitions
    def evaluate_constraints(self):
        results = []
        generator = At_Most_Generator(self.problem)
        generator.min_num_constraints = 1
        generator.max_num_constraints = 1
        generator.min_num_variables = len(self.problem.get_variables())
        generator.max_num_variables = len(self.problem.get_variables())
        generator.num_repetitions = 1
        for cardinality in range(1, len(self.problem.get_variables()) + 1):
            generator.cardinality = cardinality
            results.append(generator.evaluate_constraints())
        return pd.concat(results)


class Iff_Generator(Constraint_Generator):

    def __init__(self, problem, global_at_most):
        super().__init__(problem)
        self.global_at_most = global_at_most

    # As iff does not exclude the trivial solution (select everything), we
    # also add a global cardinality constraint
    def generate(self, variables):
        result = expr.Iff(variables)
        if self.problem.num_constraints == 0:
            global_at_most_constraint = expr.Weighted_Sum_Lt_Eq(
                self.problem.get_variables(), [1] * len(self.problem.get_variables()),
                self.global_at_most)
            result = expr.And([global_at_most_constraint, result])
        return result


class Nand_Generator(Constraint_Generator):

    def generate(self, variables):
        return expr.Not(expr.And(variables))


class Xor_Generator(Constraint_Generator):

    def generate(self, variables):
        return expr.Xor(variables[0], variables[1])
