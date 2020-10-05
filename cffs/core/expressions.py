"""SMT expressions

Logical and arithmetic expressions which allow to formulate constraints.
"""


from abc import ABCMeta, abstractmethod
from typing import Sequence


class BooleanExpression(metaclass=ABCMeta):

    @abstractmethod
    def is_true(self) -> bool:
        raise NotImplementedError('Abstract method.')


class Variable(BooleanExpression):

    def __init__(self, name: str):
        self.name = name
        self.value = False

    def is_true(self) -> bool:
        return self.value

    def __bool__(self) -> bool:
        return self.is_true()

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


class Iff(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        joint_value = self.bool_expressions[0].is_true()
        for bool_expression in self.bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True


class Implies(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return not (self.bool_expression1.is_true() and not self.bool_expression2.is_true())


class Not(BooleanExpression):

    def __init__(self, bool_expression: BooleanExpression):
        self.expression = bool_expression

    def is_true(self) -> bool:
        return not self.expression.is_true()


class Or(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.bool_expressions:
            if bool_expression.is_true():
                return True
        return False


class Xor(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return self.bool_expression1.is_true() != self.bool_expression2.is_true()


class ArithmeticExpression(metaclass=ABCMeta):

    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError('Abstract method.')


class NumericConstant(ArithmeticExpression):

    def __init__(self, value: float):
        self.value = value

    def value(self) -> float:
        return self.value


class Eq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.value() == self.arith_expression2.value()


class GtEq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.value() >= self.arith_expression2.value()


class LtEq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.value() <= self.arith_expression2.value()


class Sum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def value(self) -> float:
        result = 0
        for bool_expression in self.bool_expressions:
            if bool_expression.is_true():
                result += 1
        return result


class WeightedSum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression], weights: Sequence[float]):
        self.bool_expressions = bool_expressions
        self.weights = weights

    def value(self) -> float:
        result = 0
        for (bool_expression, weight) in zip(self.bool_expressions, self.weights):
            if bool_expression.is_true():
                result += weight
        return result
