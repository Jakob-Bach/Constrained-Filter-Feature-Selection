"""SMT expressions based on Z3

Combination of own expressions (used for model counting) and Z3 expressions
(used for optimization) to have a uniform interface and avoid inconsistencies.
"""


import z3

import expressions


class Variable(expressions.Variable):

    def __init__(self, name):
        super().__init__()
        self.z3 = z3.Bool(name)


class And(expressions.And):

    def __init__(self, bool_expressions):
        super().__init__(bool_expressions)
        self.z3 = z3.And([x.z3 for x in bool_expressions])


class Iff(expressions.Iff):

    def __init__(self, bool_expressions):
        super().__init__(bool_expressions)
        self.z3 = z3.Or(z3.And([x.z3 for x in bool_expressions]),
                        z3.Not(z3.Or([x.z3 for x in bool_expressions])))


class Implies(expressions.Implies):

    def __init__(self, bool_expression1, bool_expression2):
        super().__init__(bool_expression1, bool_expression2)
        self.z3 = z3.Implies(bool_expression1.z3, bool_expression2.z3)


class Not(expressions.Not):

    def __init__(self, bool_expression):
        super().__init__(bool_expression)
        self.z3 = z3.Not(bool_expression.z3)


class Or(expressions.Or):

    def __init__(self, bool_expressions):
        super().__init__(bool_expressions)
        self.z3 = z3.Or([x.z3 for x in bool_expressions])


class WeightedSumEq(expressions.Eq):

    def __init__(self, bool_expressions, weights, value):
        super().__init__(expressions.WeightedSum(bool_expressions, weights),
                         expressions.NumericConstant(value))
        self.z3 = z3.PbEq([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumGtEq(expressions.GtEq):

    def __init__(self, bool_expressions, weights, value):
        super().__init__(expressions.WeightedSum(bool_expressions, weights),
                         expressions.NumericConstant(value))
        self.z3 = z3.PbGe([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumLtEq(expressions.LtEq):

    def __init__(self, bool_expressions, weights, value):
        super().__init__(expressions.WeightedSum(bool_expressions, weights),
                         expressions.NumericConstant(value))
        self.z3 = z3.PbLe([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class Xor(expressions.Xor):

    def __init__(self, bool_expression1, bool_expression2):
        super().__init__(bool_expression1, bool_expression2)
        self.z3 = z3.Xor(bool_expression1.z3, bool_expression2.z3)
