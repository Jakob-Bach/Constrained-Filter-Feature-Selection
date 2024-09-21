"""SMT solving

Despite the name, this module does not contain functionality to solve SMT optimization problems
(see :mod:`combi_solving` for the latter). However, it provide a class representing such problems,
including some auxiliary functionality.

Literature
----------
- Bach et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"
- Barrett & Tinelli (2018): "Satisfiability Modulo Theories"
"""

import itertools
import random
from typing import Sequence

from . import expressions as expr


class Problem:
    """SMT problem

    A class representing an SMT problem with boolean decision variables, including functions to
    count the number of solutions (but not to efficiently find them under an objective).
    """

    def __init__(self, variable_names: Sequence[str]):
        """Initialize problem

        Creates an unconstrained SMT problem and internally stores one binary decision variable for
        each provided variable name.

        Parameters
        ----------
        variable_names : Sequence[str]
            Desired names of the decision variables.
        """

        self.variables = [expr.Variable(name=x) for x in variable_names]
        self.constraints = []  # several constraints allowed, will be combined by AND

    def get_variables(self) -> Sequence[expr.Variable]:
        """Get decision variables

        Returns
        -------
        Sequence[expr.Variable]
            The decision variables of this SMT problem.
        """

        return self.variables

    def get_constrained_variables(self) -> Sequence[expr.Variable]:
        """Get constrained variables

        Iterates over all internally stored constraints and retrieves the decision variables
        involved in the corresponding expressions (including all subexpressions).

        Returns
        -------
        result : Sequence[expr.Variable]
            The found decision variables. A variable may appear multiple times if it occurs in
            multiple subexpressions.
        """

        result = []
        for constraint in self.constraints:
            result.extend(expr.get_involved_variables(constraint))
        return result

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        """Add a constraint

        Adds a constraint to this SMT problem, retaining all existing constraints.

        Parameters
        ----------
        constraint : expr.BooleanExpression
            The constraint to be added, which is an expression evaluating to true or false. Should
            use the decision variables of this problem (:meth`get_variables`).
        """

        self.constraints.append(constraint)

    def clear_constraints(self) -> None:
        """Remove constraints

        Removes all constraints from this SMT problem. The decision variables still exist as-is.
        """

        self.constraints.clear()

    def get_num_constraints(self) -> int:
        """Get number of constraints

        Returns
        -------
        int
            Number of constraints currently stored in this SMT problem.
        """

        return len(self.constraints)

    def compute_solution_fraction(self) -> float:
        """Compute fraction of solutions

        Exactly determine the fraction of solutions to this SMT problem under the current
        constraints, i.e., the ratio between the number of valid assignments to the binary decision
        variables and the total number of potential assignments (2^n). The runtime of this method
        increases exponentially with the number of decision variables (thus, see
        :meth:`estimate_solution_fraction`).

        Returns
        -------
        float
            The fraction of solutions in [0, 1].
        """

        solutions = 0
        for assignment in itertools.product([False, True], repeat=len(self.variables)):
            # Assign
            for i, value in enumerate(assignment):
                self.variables[i].value = value
            # Check SAT
            satisfied = True
            for constraint in self.constraints:
                if not constraint.is_true():
                    satisfied = False
                    break
            solutions = solutions + satisfied
        return solutions / 2 ** len(self.variables)

    def estimate_solution_fraction(self, iterations: int = 1000) -> float:
        """Estimate fraction of solutions

        Approximates the fraction of solutions to this SMT problem (see
        :meth:`compute_solution_fraction`) by randomly sampling (with replacement) variable
        assignments for a fixed number of iterations. In a problem with many variables and strong
        constraints, this method may return zero even if valid solutions exist.

        Returns
        -------
        float
            The fraction of solutions in [0, 1].
        """

        solutions = 0
        for _ in range(iterations):
            assignment = [random.random() >= 0.5 for j in range(len(self.variables))]
            # Assign
            for i, value in enumerate(assignment):
                self.variables[i].value = value
            # Check SAT
            satisfied = True
            for constraint in self.constraints:
                if not constraint.is_true():
                    satisfied = False
                    break
            solutions = solutions + satisfied
        return solutions / iterations
