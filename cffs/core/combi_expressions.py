"""SMT expressions based on Z3

Combination of own expressions (used for model counting) and Z3 expressions
(used for optimization) to have a uniform interface and avoid inconsistencies.
"""


from abc import ABCMeta
from typing import Sequence

import z3

from . import expressions as expr


class BooleanExpression(expr.BooleanExpression, metaclass=ABCMeta):

    def get_z3(self) -> z3.BoolRef:
        return self.z3_expr  # sub-classes need to set this value


class BooleanValue(expr.BooleanValue, BooleanExpression):

    def __init__(self, value: bool):
        super().__init__(value=value)
        self.z3_expr = z3.BoolVal(value)


class Variable(expr.Variable, BooleanExpression):

    def __init__(self, name: str):
        super().__init__(name=name)
        self.z3_expr = z3.Bool(name)


class And(expr.And, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.And([x.get_z3() for x in bool_expressions])


class AtLeast(expr.Ge, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: float):
        super().__init__(expr.Sum(bool_expressions), expr.NumericValue(value))
        self.z3_expr = z3.AtLeast(*[e.get_z3() for e in bool_expressions], value)


class AtMost(expr.Le, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: float):
        super().__init__(expr.Sum(bool_expressions), expr.NumericValue(value))
        self.z3_expr = z3.AtMost(*[e.get_z3() for e in bool_expressions], value)


class Iff(expr.Iff, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.Or(z3.And([x.get_z3() for x in bool_expressions]),
                             z3.Not(z3.Or([x.get_z3() for x in bool_expressions])))


class Implies(expr.Implies, BooleanExpression):

    def __init__(self, bool_expression1: expr.BooleanExpression, bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3_expr = z3.Implies(bool_expression1.get_z3(), bool_expression2.get_z3())


class Not(expr.Not, BooleanExpression):

    def __init__(self, bool_expression: expr.BooleanExpression):
        super().__init__(bool_expression)
        self.z3_expr = z3.Not(bool_expression.get_z3())


class Or(expr.Or, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.Or([x.get_z3() for x in bool_expressions])


class WeightedSumEq(expr.Eq, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbEq([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumGe(expr.Ge, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbGe([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)], value)


class WeightedSumLe(expr.Le, BooleanExpression):

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbLe([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)], value)


class Xor(expr.Xor, BooleanExpression):

    def __init__(self, bool_expression1: expr.BooleanExpression, bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3_expr = z3.Xor(bool_expression1.get_z3(), bool_expression2.get_z3())
