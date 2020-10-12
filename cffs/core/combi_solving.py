"""SMT solving and optimization

Combination of own SMT solution counter and Z3 optimizer.
"""


from typing import Dict, Sequence

import z3

from . import combi_expressions as expr
from . import solving


class Problem(solving.Problem):

    z3.set_param('sat.cardinality.solver', False)  # improves (!) performance of cardinality constraints

    def __init__(self, variable_names: Sequence[str], qualities: Sequence[float]):
        assert len(variable_names) == len(qualities)
        # Improve optimizer performance by sorting qualities decreasingly; order of variable names adapted accordingly
        qualities, variable_names = zip(*sorted(zip(qualities, variable_names), key=lambda x: -x[0]))
        self.variables = [expr.Variable(name=x) for x in variable_names]
        self.constraints = []
        self.optimizer = z3.Optimize()
        # Direct multiplication between bool var and real quality returns wrong type (BoolRef) if quality is 1,
        # so we use "If" instead (to which that multiplication is transformed anyway)
        objective = z3.Sum([z3.If(var.get_z3(), q, 0) for (q, var) in zip(qualities, self.get_variables())])
        self.objective = self.optimizer.maximize(objective)
        self.optimizer.push()  # restore point for state without constraints

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        super().add_constraint(constraint)
        self.optimizer.add(constraint.get_z3())  # AttributeError if "z3" not set in BooleanExpression object

    # Remove all constraints
    def clear_constraints(self) -> None:
        super().clear_constraints()
        self.optimizer.pop()  # go to restore point (no constraints)
        self.optimizer.push()  # create new restore point

    # Run optimization and return result dict
    def optimize(self) -> Dict[str, float]:
        self.optimizer.check()
        # Z3 returns different type, depending on whether result is a whole number
        if self.objective.value().is_int():
            value = self.objective.value().as_long()
        else:
            value = self.objective.value().numerator_as_long() /\
                self.objective.value().denominator_as_long()
        model = self.optimizer.model()
        selected = [var.get_name() for var in self.get_variables() if str(model[var.get_z3()]) == 'True']
        return {'objective_value': value, 'num_selected': len(selected),
                'frac_selected': len(selected) / len(self.get_variables()), 'selected': selected}
