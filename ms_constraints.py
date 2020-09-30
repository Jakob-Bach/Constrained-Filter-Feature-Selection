"""MS constraints

Classes to evaluate specific constraints for materials science use cases.
"""

from abc import ABCMeta, abstractmethod
import re
from typing import Dict

import combi_expressions as expr
import combi_solving as solv
import ms_datasets


SCHMID_GROUPS_100 = [[1, 2, 5, 6, 7, 8, 11, 12], [3, 4, 9, 10]]  # for (1 0 0) orientation


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


# For Schmid factor (1 0 0) grouping, select features from at most one group
class SelectSchmidGroupEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        variable_groups = []
        for slip_group in SCHMID_GROUPS_100:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search('_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                           variable.get_name()) is not None]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))


# For each quantity, for Schmid factor (1 0 0) grouping, select features from at most one group
class SelectQuantitySchmidGroupEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            variable_groups = []
            for slip_group in SCHMID_GROUPS_100:
                variable_group = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                               variable.get_name()) is not None]
                variable_groups.append(variable_group)
            self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))


# For Schmid factor (1 0 0) grouping, select at most one feature from each group
class SelectSchmidGroupRepresentativeEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        variable_groups = []
        for slip_group in SCHMID_GROUPS_100:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search('_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                           variable.get_name()) is not None]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.And([expr.AtMost(x, 1) for x in variable_groups]))


# For each quantity, for Schmid factor (1 0 0) grouping, select at most one feature from each group
class SelectQuantitySchmidGroupRepresentativeEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            variable_groups = []
            for slip_group in SCHMID_GROUPS_100:
                variable_group = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                               variable.get_name()) is not None]
                variable_groups.append(variable_group)
            self.problem.add_constraint(expr.And([expr.AtMost(x, 1) for x in variable_groups]))


# For each slip system, select either all quantities from that slip system or none
class SelectWholeSlipSystemsEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        for slip_system in range(1, 13):
            variable_group = [variable for variable in self.problem.get_variables()
                              if variable.get_name().endswith('_' + slip_system)]
            self.problem.add_constraint(expr.Iff(variable_group))


# From reaction features, select features belonging to at most one reaction type
class SelectReactionTypeEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        variable_groups = []
        for reaction_type in ms_datasets.REACTION_TYPES:
            variable_group = [variable for variable in self.problem.get_variables()
                              if reaction_type in variable.get_name()]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))


# For each quantity, select either absolute value or delta value or none
class SelectValueOrDeltaEvaluator(MSConstraintEvaluator):

    def add_constraints(self) -> None:
        delta_variables = [variable for variable in self.problem.get_variables()
                           if 'delta_' in variable.get_name()]
        for delta_variable in delta_variables:
            variable_name = delta_variable.get_name().replace('delta_', '')
            for variable in self.problem.get_variables():
                if variable.get_name() == variable_name:
                    self.problem.add_constraint(expr.Not(expr.And([variable, delta_variable])))
                    break


# From plastic strain tensor, select at most three directions
class SelectStrainTensorEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        variable_groups = []
        directions = [variable.get_name().replace('eps_', '') for variable in self.problem.get_variables()
                      if re.match('eps_[a-z]{2}$', variable.get_name())]
        for direction in directions:
            variable_group = [variable for variable in self.problem.get_variables()
                              if 'eps_' + direction in variable.get_name()]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 3))


# For dislocation density, select at most one from three feature groups which all describe it
class SelectDislocationDensityEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        variable_groups = []
        quantity_patterns = ['rho_(' + '|'.join(ms_datasets.AGGREGATES) + ')',
                             'mean_free_path', 'free_path_per_voxel']
        for pattern in quantity_patterns:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search(pattern, variable.get_name()) is not None]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))


# For strain rate computation, select at most one type
class SelectStrainRateEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        gamma_variables = [variable for variable in self.problem.get_variables()
                           if 'gamma' in variable.get_name()]
        gamma_abs_variables = [variable for variable in gamma_variables
                               if 'gamma_abs' in variable.get_name()]
        gamma_variables = [variable for variable in gamma_variables
                           if 'gamma_abs' not in variable.get_name()]
        self.problem.add_constraint(expr.Not(expr.And([expr.Or(gamma_variables),
                                                       expr.Or(gamma_abs_variables)])))


# Over all quantities, select at most one type of aggregate
class SelectAggregateEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        variable_groups = []
        for aggregate in ms_datasets.AGGREGATES:
            variable_group = [variable for variable in self.problem.get_variables()
                              if variable.get_name().endswith('_' + aggregate)]
            variable_groups.append(variable_group)
        self.problem.add_constraint(expr.AtMost([expr.Or(x) for x in variable_groups], 1))


# For each quantity, select at most one type of aggregate
class SelectQuantityAggregateEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            aggregate_variables = [variable for variable in self.problem.get_variables()
                                   if re.search(quantity + '_(' + '|'.join(ms_datasets.AGGREGATES) + ')$',
                                                variable.get_name()) is not None]
            if len(aggregate_variables) > 0:
                self.problem.add_constraint(expr.AtMost(aggregate_variables, 1))


# For each quantity, select either aggregates or orignal values or none
class SelectAggregateOrOriginalEvaluator(MSConstraintEvaluator):

    def add_constraints(self):
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            original_variables = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_[0-9]+$', variable.get_name()) is not None]
            aggregate_variables = [variable for variable in self.problem.get_variables()
                                   if re.search(quantity + '_(' + '|'.join(ms_datasets.AGGREGATES) + ')$',
                                                variable.get_name()) is not None]
            if len(original_variables) > 0 and len(aggregate_variables) > 0:
                self.problem.add_constraint(expr.Not(expr.And([expr.Or(original_variables),
                                                               expr.Or(aggregate_variables)])))
