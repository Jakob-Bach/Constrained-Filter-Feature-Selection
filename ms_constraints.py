"""MS constraints

Classes to evaluate specific constraints for materials science use cases.
"""

from abc import ABCMeta, abstractmethod
import re
from typing import Dict

import combi_expressions as expr
import combi_solving as solv


class MSConstraintEvaluator(metaclass=ABCMeta):

    def __init__(self, problem: solv.Problem):
        self.problem = problem

    @abstractmethod
    def add_constraints(self) -> None:
        raise NotImplementedError('Abstract method.')

    def evaluate_constraints(self) -> Dict[str, float]:
        self.add_constraints()
        frac_solutions = self.problem.estimate_solution_fraction(iterations=1000)
        result = self.problem.optimize()
        result['num_constraints'] = self.problem.get_num_constraints()
        result['frac_solutions'] = frac_solutions
        self.problem.clear_constraints()
        return result


class NoConstraintEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        pass


class SchmidFactor100Evaluator(MSConstraintEvaluator):

    SLIP_GROUPS = [[1, 2, 5, 6, 7, 8, 11, 12], [3, 4, 9, 10]]  # for (1 0 0) orientation

    def add_constraints(self) -> None:
        variable_groups = []
        for slip_group in SchmidFactor100Evaluator.SLIP_GROUPS:
            variable_group = []
            for variable in self.problem.get_variables():
                if re.search('_(' + '|'.join([str(i) for i in slip_group]) + ')$', str(variable.z3)) is not None:
                    variable_group.append(variable)
            variable_groups.append(variable_group)
        # Select features from at most one of these groups
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))
