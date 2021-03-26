"""Constraints for the case study in materials science

Classes representing manually-defined constraints for our case study in materials science.
Each class is able to evaluate its constraint set on a given feature-selection problem.
"""

from abc import ABCMeta, abstractmethod
import random
import re
from typing import Any, Dict, Iterable, Type

import pandas as pd

from cffs.core import combi_expressions as expr
from cffs.core import combi_solving as solv
from cffs.materials_science import ms_data_utility


SCHMID_GROUPS_100 = [[1, 2, 5, 6, 7, 8, 11, 12], [3, 4, 9, 10]]  # groups for (1 0 0) orientation of crystal


# Super-class containing the evaluation procedure for constraints, without defining concrete
# constraints (that is up to the sub-classes).
class MSConstraintEvaluator(metaclass=ABCMeta):

    def __init__(self, problem: solv.Problem):
        self.problem = problem

    # Sub-classes should implement this method by generating multiple boolean expressions,
    # representing a set of constraints that should be considered simultaneously.
    # For formulating constraints, you can use the variables in self.problem.get_variables().
    # This method will be called as a sub-routine from the main evaluation procedure.
    @abstractmethod
    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        raise NotImplementedError('Abstract method.')

    # Evaluate a set of constraints by solving the optimization problem of constrained feature
    # selection. Return a dictionary with the core evaluation metrics.
    def evaluate_constraints(self) -> Dict[str, float]:
        random.seed(25)
        for constraint in self.get_constraints():
            self.problem.add_constraint(constraint)
        frac_solutions = self.problem.estimate_solution_fraction(iterations=10000)
        constrained_variables = self.problem.get_constrained_variables()
        unique_constrained_variables = set(constrained_variables)
        result = self.problem.optimize()
        result['num_variables'] = len(self.problem.get_variables())
        result['num_constrained_variables'] = len(constrained_variables)
        result['num_unique_constrained_variables'] = len(unique_constrained_variables)
        result['num_constraints'] = self.problem.get_num_constraints()
        result['frac_solutions'] = frac_solutions
        self.problem.clear_constraints()
        return result


# Combines constraints from multiple sub-evaluators. Does not prescribe a fixed combination.
class CombinedEvaluator(MSConstraintEvaluator):

    # For each sub-evaluator, pass type and dict with initialization arguments; the latter does
    # not need to contain the "problem" itself.
    def __init__(self, problem: solv.Problem, evaluators: Dict[Type[MSConstraintEvaluator], Dict[str, Any]] = None):
        super().__init__(problem=problem)
        self.evaluators = [type_object(**{'problem': problem, **args_dict})  # create evaluator
                           for type_object, args_dict in evaluators.items()]

    # Combine the constraints from all sub-evaluators.
    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        for evaluator in self.evaluators:
            constraints.extend(evaluator.get_constraints())
        return constraints


# The reference case without constraints.
class UnconstrainedEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        return []


# Select only a certain amount of features, i.e., use a global cardinality threshold
# (domain-independent constraint type).
class GlobalCardinalityEvaluator(MSConstraintEvaluator):

    def __init__(self, problem: solv.Problem, global_at_most: int = 10):
        super().__init__(problem=problem)
        self.global_at_most = global_at_most

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        return [expr.AtMost(self.problem.get_variables(), self.global_at_most)]


# Select only features with at least a certain quality (domain-independent constraint type).
class QualityFilterEvaluator(MSConstraintEvaluator):

    def __init__(self, problem: solv.Problem, threshold: float):
        super().__init__(problem=problem)
        self.threshold = threshold

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        return [expr.Not(v) for v, q in zip(self.problem.get_variables(), self.problem.get_qualities())
                if q < self.threshold]  # could also express this with expr.Implies()


# Do not select two features at the same time if they are correlated over a certain threshold.
# (domain-independent constraint type).
class InterCorrelationEvaluator(MSConstraintEvaluator):

    # Prepare a list of pairs of features which pass the correlation threshold.
    def __init__(self, problem: solv.Problem, corr_df: pd.DataFrame, threshold: float):
        # Make sure that correlation matrix refers to the same features as variables in "problem"
        # (though "problem" might change variable order for efficiency reasons):
        sorted_variable_names = sorted([variable.get_name() for variable in problem.get_variables()])
        assert sorted_variable_names == sorted(corr_df.columns)
        assert sorted_variable_names == sorted(corr_df.index)
        super().__init__(problem=problem)
        self.correlation_pairs = []
        for i in range(len(corr_df)):
            variable_1 = problem.get_variables()[i]
            for j in range(i):
                variable_2 = problem.get_variables()[j]
                if corr_df.loc[variable_1.get_name(), variable_2.get_name()] >= threshold:
                    self.correlation_pairs.append((variable_1, variable_2))

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        return [expr.Not(expr.And([v1, v2])) for v1, v2 in self.correlation_pairs]


# For the Schmid factor grouping of slip systems, select features from at most one group.
class SchmidGroupEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        variable_groups = []
        for slip_group in SCHMID_GROUPS_100:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search('_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                           variable.get_name()) is not None]
            variable_groups.append(variable_group)
        return [expr.AtMost([expr.Or(x) for x in variable_groups], 1)]


# For each quantity, for the Schmid factor grouping of slip systems, select features from at most
# one group.
class QuantitySchmidGroupEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            variable_groups = []
            for slip_group in SCHMID_GROUPS_100:
                variable_group = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                               variable.get_name()) is not None]
                variable_groups.append(variable_group)
            constraints.append(expr.AtMost([expr.Or(x) for x in variable_groups], 1))
        return constraints


# For the Schmid factor grouping of slip systems, select at most one feature from each group.
class SchmidGroupRepresentativeEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        for slip_group in SCHMID_GROUPS_100:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search('_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                           variable.get_name()) is not None]
            if len(variable_group) > 0:  # z3.AtMost not defined if applied to empty list
                constraints.append(expr.AtMost(variable_group, 1))
        return constraints


# For each quantity, for the Schmid factor grouping of slip systems, select at most one feature
# from each group.
class QuantitySchmidGroupRepresentativeEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            for slip_group in SCHMID_GROUPS_100:
                variable_group = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_(' + '|'.join([str(i) for i in slip_group]) + ')$',
                                               variable.get_name()) is not None]
                if len(variable_group) > 0:  # z3.AtMost not defined if applied to empty list
                    constraints.append(expr.AtMost(variable_group, 1))
        return constraints


# From plastic strain tensor, select at most three directions.
class PlasticStrainTensorEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        variable_group = [variable for variable in self.problem.get_variables()
                          if re.match('eps_[a-z]{2}$', variable.get_name()) is not None]
        if len(variable_group) == 0:
            return []  # z3.AtMost not defined if applied to empty list
        return [expr.AtMost(variable_group, 3)]


# For dislocation density, aggregated over slip systems, select at most one from the features that
# describe it.
class DislocationDensityEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        pattern = 'rho_(' + '|'.join(ms_data_utility.AGGREGATE_FUNCTIONS) + ')' +\
            '|mean_free_path|free_path_per_voxel'
        variable_group = [variable for variable in self.problem.get_variables()
                          if re.match(pattern, variable.get_name()) is not None]
        if len(variable_group) == 0:
            return []  # z3.AtMost not defined if applied to empty list
        return [expr.AtMost(variable_group, 1)]


# From methods to compute plastic strain rate, select at most one method.
class PlasticStrainRateEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        gamma_variables = [variable for variable in self.problem.get_variables()
                           if 'gamma' in variable.get_name()]
        gamma_abs_variables = [variable for variable in gamma_variables
                               if 'gamma_abs' in variable.get_name()]
        gamma_variables = [variable for variable in gamma_variables
                           if 'gamma_abs' not in variable.get_name()]
        return [expr.Not(expr.And([expr.Or(gamma_variables), expr.Or(gamma_abs_variables)]))]


# Over all quantities, select at most one type of aggregate.
class AggregateEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        variable_groups = []
        for aggregate in ms_data_utility.AGGREGATE_FUNCTIONS:
            variable_group = [variable for variable in self.problem.get_variables()
                              if variable.get_name().endswith('_' + aggregate)]
            variable_groups.append(variable_group)
        return [expr.AtMost([expr.Or(x) for x in variable_groups], 1)]


# For each quantity, select at most one type of aggregate.
class QuantityAggregateEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            variable_group = [variable for variable in self.problem.get_variables()
                              if re.search(quantity + '_(' + '|'.join(ms_data_utility.AGGREGATE_FUNCTIONS) + ')$',
                                           variable.get_name()) is not None]
            if len(variable_group) > 0:  # z3.AtMost not defined if applied to empty list
                constraints.append(expr.AtMost(variable_group, 1))
        return constraints


# For each quantity, select either aggregates or orignal values or none.
class AggregateOrOriginalEvaluator(MSConstraintEvaluator):

    def get_constraints(self) -> Iterable[expr.BooleanExpression]:
        constraints = []
        base_quantities = [variable.get_name().replace('_1', '') for variable in self.problem.get_variables()
                           if variable.get_name().endswith('_1')]
        for quantity in base_quantities:
            original_variables = [variable for variable in self.problem.get_variables()
                                  if re.search(quantity + '_[0-9]+$', variable.get_name()) is not None]
            aggregate_variables = [variable for variable in self.problem.get_variables()
                                   if re.search(quantity + '_(' + '|'.join(ms_data_utility.AGGREGATE_FUNCTIONS) + ')$',
                                                variable.get_name()) is not None]
            constraints.append(expr.Not(expr.And([expr.Or(original_variables), expr.Or(aggregate_variables)])))
        return constraints
