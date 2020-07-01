"""SMT solving

Description of a constraint satisfcation problem with boolean variables,
including functions to count solutions (not to efficiently find them).
"""


import itertools
import random
from typing import Sequence

import expressions as expr


class Problem:

    def __init__(self, variables: Sequence[expr.Variable]):
        self.__variables = variables
        self.__constraints = []  # several constraints allowed, will be combined by AND

    def get_variables(self) -> Sequence[expr.Variable]:
        return self.__variables

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        self.__constraints.append(constraint)

    # Remove all constraints
    def clear_constraints(self) -> None:
        self.__constraints.clear()

    def num_constraints(self) -> int:
        return len(self.__constraints)

    # Exact procedure for determining fraction of solutions (valid assignments given constraints)
    def compute_solution_fraction(self) -> float:
        solutions = 0
        for assignment in itertools.product([False, True], repeat=len(self.__variables)):
            # Assign
            for i, value in enumerate(assignment):
                self.__variables[i].value = value
            # Check SAT
            satisfied = True
            for constraint in self.__constraints:
                if not constraint.is_true():
                    satisfied = False
                    break
            solutions = solutions + satisfied
        return solutions / 2 ** len(self.__variables)

    # Probabilistic procedure for determining fraction of solutions (valid assignments given constraints)
    def estimate_solution_fraction(self, iterations: int = 1000) -> float:
        solutions = 0
        for _ in range(iterations):
            assignment = [random.random() >= 0.5 for j in range(len(self.__variables))]
            # Assign
            for i, value in enumerate(assignment):
                self.__variables[i].value = value
            # Check SAT
            satisfied = True
            for constraint in self.__constraints:
                if not constraint.is_true():
                    satisfied = False
                    break
            solutions = solutions + satisfied
        return solutions / iterations
