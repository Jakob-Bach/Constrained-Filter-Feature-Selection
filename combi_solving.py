"""SMT solving and optimization

Combination of own SMT solution counter and Z3 optimizer.
"""


import z3

import solving


class Problem(solving.Problem):

    def __init__(self, variables, qualities):
        super().__init__(variables)
        self.__optimizer = z3.Optimize()
        objective = z3.Sum([q * var.z3 for (q, var) in zip(qualities, variables)])
        self.__objective = self.__optimizer.maximize(objective)
        self.__optimizer.push()  # restore point for state without constraints

    # Add a Boolean_Expression as constraint
    def add_constraint(self, constraint):
        super().add_constraint(constraint)
        self.__optimizer.add(constraint.z3)

    # Remove all constraints
    def clear_constraints(self):
        super().clear_constraints()
        self.__optimizer.pop()  # go to restore point (no constraints)
        self.__optimizer.push()  # create new restore point

    # Run optimization and return result dict
    def optimize(self):
        self.__optimizer.check()
        # Z3 returns different type, depending on whether result is a whole number
        if self.__objective.value().is_int():
            value = self.__objective.value().as_long()
        else:
            value = self.__objective.value().numerator_as_long() /\
                self.__objective.value().denominator_as_long()
        num_selected = sum([str(self.__optimizer.model()[var.z3]) == 'True' for var in self.__variables])
        return {'objective_value': value, 'num_selected': num_selected}
