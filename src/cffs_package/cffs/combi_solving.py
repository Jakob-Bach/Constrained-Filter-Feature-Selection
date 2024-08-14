"""SMT solving and optimization

A class combining our own SMT solution counter with the Z3 optimizer, to represent the problem of
constrained, univariate filter feature selection.
"""

from typing import Dict, Union, Sequence

import z3

from . import combi_expressions as expr
from . import solving


class Problem(solving.Problem):

    z3.set_param('sat.cardinality.solver', False)  # improves (!) performance of cardinality constraints

    def __init__(self, variable_names: Sequence[str], qualities: Sequence[float]):
        assert len(variable_names) == len(qualities)
        # Improve optimizer performance by sorting qualities decreasingly; adapt order of variable names accordingly:
        qualities, variable_names = zip(*sorted(zip(qualities, variable_names), key=lambda x: -x[0]))
        self.qualities = qualities
        self.variables = [expr.Variable(name=x) for x in variable_names]
        self.constraints = []
        self.optimizer = z3.Optimize()
        # Direct multiplication between bool var and real quality returns wrong type (BoolRef) if quality is 1,
        # so we use "If" instead (multiplication is transformed to such an expression anyway):
        objective = z3.Sum([z3.If(var.get_z3(), q, 0) for (q, var) in zip(qualities, self.get_variables())])
        self.objective = self.optimizer.maximize(objective)
        self.optimizer.push()  # restore point for state without constraints

    def get_qualities(self) -> Sequence[float]:
        return self.qualities

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        super().add_constraint(constraint)
        self.optimizer.add(constraint.get_z3())  # AttributeError if attribute "z3_expr" not set in "constraint"

    # Remove all constraints
    def clear_constraints(self) -> None:
        super().clear_constraints()
        self.optimizer.pop()  # go to restore point (state with no constraints)
        self.optimizer.push()  # create new restore point (again, with no constraints)

    # Run optimization and return result dict
    def optimize(self) -> Dict[str, Union[float, Sequence[str]]]:
        self.optimizer.check()
        # Object value can have different types, depending on whether result is a whole number;
        # if no valid variable assignment (result of "check()" is "unsat"), objective value is 0
        if self.objective.value().is_int():  # type IntNumRef
            value = self.objective.value().as_long()
        else:  # type RatNumRef
            value = self.objective.value().numerator_as_long() / self.objective.value().denominator_as_long()
        model = self.optimizer.model()
        # If no valid variable assignment exists, values of all variables in model are None by
        # default; in our code, these values become converted to False
        selected = [var.get_name() for var in self.get_variables() if str(model[var.get_z3()]) == 'True']
        return {'objective_value': value, 'num_selected': len(selected), 'selected': selected}
