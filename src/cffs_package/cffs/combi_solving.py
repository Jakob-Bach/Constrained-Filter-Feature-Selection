"""SMT optimization wrapping Z3

This module combines our own SMT-expressions types (:mod:`expressions`) and solution-counting
approaches (:mod:`solving`) with the SMT solver "Z3" for optimization.

Literature
----------
- Bach et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"
- Barrett & Tinelli (2018): "Satisfiability Modulo Theories"
- de Moura & Bjorner (2008): "Z3: An Efficient SMT Solver"
"""

from typing import Dict, Union, Sequence

import z3

from . import combi_expressions as expr
from . import solving


class Problem(solving.Problem):
    """SMT optimization problem

    An SMT optimization problem with boolean decision variables and a linear objective function.
    Represents the problem of constrained (univariate filter) feature selection.
    """

    # Following setting counterintuitively improves evaluation speed of cardinality constraints
    z3.set_param('sat.cardinality.solver', False)

    def __init__(self, variable_names: Sequence[str], qualities: Sequence[float]):
        """Initialize problem

        Creates an unconstrained SMT problem and internally stores one binary decision variable for
        each provided variable (= feature) name. The number of provided (feature) qualities should
        match accordingly. These qualities form the (linear) objective value, i.e., the sum of the
        qualities of the selected features should be optimized.

        Parameters
        ----------
        variable_names : Sequence[str]
            Desired names of the decision variables (which represent feature-selection decisions).
        qualities : Sequence[float]
            Univariate feature qualities; see :mod:`feature_qualities`.
        """

        assert len(variable_names) == len(qualities)
        # Improve optimizer performance by sorting qualities decreasingly; adapt order of variable
        # names accordingly:
        qualities, variable_names = zip(*sorted(zip(qualities, variable_names),
                                                key=lambda x: -x[0]))
        self.qualities = qualities
        self.variables = [expr.Variable(name=x) for x in variable_names]
        self.constraints = []
        self.optimizer = z3.Optimize()
        # Direct multiplication between bool var and real quality returns wrong type (BoolRef) if
        # quality is 1, so we use "If" instead (multiplication is transformed to such an expression
        # anyway):
        objective = z3.Sum([z3.If(var.get_z3(), q, 0)
                            for (q, var) in zip(qualities, self.get_variables())])
        self.objective = self.optimizer.maximize(objective)
        self.optimizer.push()  # restore point for state without constraints

    def get_qualities(self) -> Sequence[float]:
        """Get feature qualities

        Returns
        -------
        Sequence[float]
            The feature qualities provided during initialization, which form the weights in the
            linear objective function with binary decision variables.
        """

        return self.qualities

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        super().add_constraint(constraint)
        self.optimizer.add(constraint.get_z3())  # AttributeError if "z3_expr" not in "constraint"

    def clear_constraints(self) -> None:
        super().clear_constraints()
        self.optimizer.pop()  # go to restore point (state with no constraints)
        self.optimizer.push()  # create new restore point (again, with no constraints)

    def optimize(self) -> Dict[str, Union[float, Sequence[str]]]:
        """Optimize problem

        Run Z3 on the SMT optimization problem and return optimization results.

        Returns
        -------
        Dict[str, Union[float, Sequence[str]]]
            Optimization results: objective value and selected features.
        """

        self.optimizer.check()
        # Object value can have different types, depending on whether result is a whole number;
        # if no valid variable assignment (result of "check()" is "unsat"), objective value is 0
        if self.objective.value().is_int():  # type IntNumRef
            value = self.objective.value().as_long()
        else:  # type RatNumRef
            value = (self.objective.value().numerator_as_long() /
                     self.objective.value().denominator_as_long())
        model = self.optimizer.model()
        # If no valid variable assignment exists, values of all variables in model are None by
        # default; in our code, these values become converted to False
        selected = [var.get_name() for var in self.get_variables()
                    if str(model[var.get_z3()]) == 'True']
        return {'objective_value': value, 'num_selected': len(selected), 'selected': selected}
