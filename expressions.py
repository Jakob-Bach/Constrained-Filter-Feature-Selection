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

    def __init__(self):
        self.value = False

    def is_true(self) -> bool:
        return self.value


class And(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.__bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.__bool_expressions:
            if not bool_expression.is_true():
                return False
        return True


class Iff(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.__bool_expressions = bool_expressions

    def is_true(self) -> bool:
        joint_value = self.__bool_expressions[0].is_true()
        for bool_expression in self.__bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True


class Implies(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return not (self.__bool_expression1.is_true() and not self.__bool_expression2.is_true())


class Not(BooleanExpression):

    def __init__(self, bool_expression: BooleanExpression):
        self.__expression = bool_expression

    def is_true(self) -> bool:
        return not self.__expression.is_true()


class Or(BooleanExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.__bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                return True
        return False


class Xor(BooleanExpression):

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return self.__bool_expression1.is_true() != self.__bool_expression2.is_true()


class ArithmeticExpression(metaclass=ABCMeta):

    @abstractmethod
    def value(self) -> float:
        raise NotImplementedError('Abstract method.')


class NumericConstant(ArithmeticExpression):

    def __init__(self, value: float):
        self.__value = value

    def value(self) -> float:
        return self.__value


class Eq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.__arith_expression1.value() == self.__arith_expression2.value()


class GtEq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.__arith_expression1.value() >= self.__arith_expression2.value()


class LtEq(BooleanExpression):

    def __init__(self, arith_expression1: ArithmeticExpression, arith_expression2: ArithmeticExpression):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.__arith_expression1.value() <= self.__arith_expression2.value()


class Sum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.__bool_expressions = bool_expressions

    def value(self) -> float:
        result = 0
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                result += 1
        return result


class WeightedSum(ArithmeticExpression):

    def __init__(self, bool_expressions: Sequence[BooleanExpression], weights: Sequence[float]):
        self.__bool_expressions = bool_expressions
        self.__weights = weights

    def value(self) -> float:
        result = 0
        for (bool_expression, weight) in zip(self.__bool_expressions, self.__weights):
            if bool_expression.is_true():
                result += weight
        return result
