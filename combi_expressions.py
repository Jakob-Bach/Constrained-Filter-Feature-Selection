"""SMT expressions based on Z3

Combination of own expressions (used for model counting) and Z3 expressions
(used for optimization) to have a uniform interface and avoid inconsistencies.
"""


from typing import Sequence

import z3

import expressions as expr


class BooleanExpression(expr.BooleanExpression):

    def get_z3(self) -> z3.BoolRef:
        return self.z3  # sub-classes need to set this value


class Variable(expr.Variable, BooleanExpression):

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.z3 = z3.Bool(name)

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class And(expr.And, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3 = z3.And([x.z3 for x in bool_expressions])


class AtLeast(expr.GtEq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: float):
        super().__init__(expr.Sum(bool_expressions), expr.NumericConstant(value))
        self.z3 = z3.AtLeast(*[e.z3 for e in bool_expressions], value)


class AtMost(expr.LtEq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: float):
        super().__init__(expr.Sum(bool_expressions), expr.NumericConstant(value))
        self.z3 = z3.AtMost(*[e.z3 for e in bool_expressions], value)


class Iff(expr.Iff, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3 = z3.Or(z3.And([x.z3 for x in bool_expressions]),
                        z3.Not(z3.Or([x.z3 for x in bool_expressions])))


class Implies(expr.Implies, BooleanExpression):

    def __init__(self, bool_expression1: expr.BooleanExpression, bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3 = z3.Implies(bool_expression1.z3, bool_expression2.z3)


class Not(expr.Not, BooleanExpression):

    def __init__(self, bool_expression: expr.BooleanExpression):
        super().__init__(bool_expression)
        self.z3 = z3.Not(bool_expression.z3)


class Or(expr.Or, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3 = z3.Or([x.z3 for x in bool_expressions])


class WeightedSumEq(expr.Eq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights),
                         expr.NumericConstant(value))
        self.z3 = z3.PbEq([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumGtEq(expr.GtEq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights),
                         expr.NumericConstant(value))
        self.z3 = z3.PbGe([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumLtEq(expr.LtEq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights),
                         expr.NumericConstant(value))
        self.z3 = z3.PbLe([(e.z3, w) for (e, w) in zip(bool_expressions, weights)], value)


class Xor(expr.Xor, BooleanExpression):

    def __init__(self, bool_expression1: expr.BooleanExpression, bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3 = z3.Xor(bool_expression1.z3, bool_expression2.z3)
