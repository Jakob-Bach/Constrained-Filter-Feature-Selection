"""SMT solving and optimization

Combination of own SMT solution counter and Z3 optimizer.
"""


from typing import Dict, Sequence

import z3

import combi_expressions as expr
import solving


class Problem(solving.Problem):

    def __init__(self, variables: Sequence[expr.Variable], qualities: Sequence[float]):
        super().__init__(variables)
        self.__optimizer = z3.Optimize()
        assert len(variables) == len(qualities)
        # Direct multiplication between bool var and real quality returns wrong type (BoolRef) if quality is 1,
        # so we use "If" instead (to which that multiplication is transformed anyway)
        objective = z3.Sum([z3.If(var.z3, q, 0) for (q, var) in zip(qualities, variables)])
        self.__objective = self.__optimizer.maximize(objective)
        self.__optimizer.push()  # restore point for state without constraints

    def add_constraint(self, constraint: expr.BooleanExpression) -> None:
        super().add_constraint(constraint)
        self.__optimizer.add(constraint.z3)  # AttributeError if "z3" not set in BooleanExpression object

    # Remove all constraints
    def clear_constraints(self) -> None:
        super().clear_constraints()
        self.__optimizer.pop()  # go to restore point (no constraints)
        self.__optimizer.push()  # create new restore point

    # Run optimization and return result dict
    def optimize(self) -> Dict[str, float]:
        self.__optimizer.check()
        # Z3 returns different type, depending on whether result is a whole number
        if self.__objective.value().is_int():
            value = self.__objective.value().as_long()
        else:
            value = self.__objective.value().numerator_as_long() /\
                self.__objective.value().denominator_as_long()
        model = self.__optimizer.model()
        selected = [i for i, var in enumerate(self.__variables) if str(model[var.z3]) == 'True']
        return {'objective_value': value, 'num_selected': len(selected), 'selected': selected}
