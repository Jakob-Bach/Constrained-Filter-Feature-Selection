"""SMT expressions

Logical and arithmetic expressions which allow to formulate constraints.
"""


from __future__ import annotations  # to use a class as a type hint within its own definition

from abc import ABCMeta, abstractmethod
from typing import Sequence


class Expression(metaclass=ABCMeta):

    def get_children(self) -> Sequence[Expression]:
        return []


class BooleanExpression(Expression, metaclass=ABCMeta):

    @abstractmethod
    def is_true(self) -> bool:
        raise NotImplementedError('Abstract method.')


class BooleanValue(BooleanExpression):

    def __init__(self, value: bool):
        self.value = value

    def is_true(self) -> bool:
        return self.value

    def __bool__(self) -> bool:
        return self.is_true()


class Variable(BooleanValue):

    def __init__(self, name: str):
        super().__init__(value=False)
        self.name = name

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


class And(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.bool_expressions:
            if not bool_expression.is_true():
                return False
        return True

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Iff(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        joint_value = self.bool_expressions[0].is_true()
        for bool_expression in self.bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Implies(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return not (self.bool_expression1.is_true() and not self.bool_expression2.is_true())

    def get_children(self) -> Sequence[Expression]:
        return [self.bool_expression1, self.bool_expression2]


class Not(BooleanExpression):

    def __init__(self, bool_expression: BooleanExpression):
        self.expression = bool_expression

    def is_true(self) -> bool:
        return not self.expression.is_true()

    def get_children(self) -> Sequence[Expression]:
        return [self.expression]


class Or(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.bool_expressions:
            if bool_expression.is_true():
                return True
        return False

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Xor(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return self.bool_expression1.is_true() != self.bool_expression2.is_true()

    def get_children(self) -> Sequence[Expression]:
        return [self.bool_expression1, self.bool_expression2]


class ArithmeticExpression(Expression, metaclass=ABCMeta):

    @abstractmethod
    def get_value(self) -> float:
        raise NotImplementedError('Abstract method.')


class NumericValue(ArithmeticExpression):

    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value


class Eq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() == self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Ge(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() >= self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Le(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() <= self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Sum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def get_value(self) -> float:
        result = 0
        for bool_expression in self.bool_expressions:
            if bool_expression.is_true():
                result += 1
        return result

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class WeightedSum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression], weights: Sequence[float]):
        self.bool_expressions = bool_expressions
        self.weights = weights

    def get_value(self) -> float:
        result = 0
        for (bool_expression, weight) in zip(self.bool_expressions, self.weights):
            if bool_expression.is_true():
                result += weight
        return result

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


def get_involved_variables(expression: Expression) -> Sequence[Variable]:
    if isinstance(expression, Variable):
        return [expression]
    elif len(expression.get_children()) == 0:
        return []
    else:
        result = []
        for child_expression in expression.get_children():
            result.extend(get_involved_variables(child_expression))
        return result
