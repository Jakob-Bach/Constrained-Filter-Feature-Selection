"""SMT expressions

Logical and arithmetic expressions which allow to formulate constraints.
"""


class BooleanExpression:

    def is_true(self):
        pass  # method not implemented here, but in each sub-class


class Variable(BooleanExpression):

    def __init__(self):
        self.value = False

    def is_true(self):
        return self.value


class And(BooleanExpression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        for bool_expression in self.__bool_expressions:
            if not bool_expression.is_true():
                return False
        return True


class Eq(BooleanExpression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() == self.__arith_expression2.value()


class Iff(BooleanExpression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        joint_value = self.__bool_expressions[0].is_true()
        for bool_expression in self.__bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True


class GtEq(BooleanExpression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() >= self.__arith_expression2.value()


class Implies(BooleanExpression):

    def __init__(self, bool_expression1, bool_expression2):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self):
        return not (self.__bool_expression1.is_true() and not self.__bool_expression2.is_true())


class LtEq(BooleanExpression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() <= self.__arith_expression2.value()


class Not(BooleanExpression):

    def __init__(self, bool_expression):
        self.__expression = bool_expression

    def is_true(self):
        return not self.__expression.is_true()


class Or(BooleanExpression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                return True
        return False


class Xor(BooleanExpression):

    def __init__(self, bool_expression1, bool_expression2):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self):
        return self.__bool_expression1.is_true() != self.__bool_expression2.is_true()


class ArithmeticExpression:

    def value(self):
        pass  # method not implemented here, but in each sub-class


class NumericConstant(ArithmeticExpression):

    def __init__(self, value):
        self.__value = value

    def value(self):
        return self.__value


class Sum(ArithmeticExpression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def value(self):
        result = 0
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                result += 1
        return result


class WeightedSum(ArithmeticExpression):

    def __init__(self, bool_expressions, weights):
        self.__bool_expressions = bool_expressions
        self.__weights = weights

    def value(self):
        result = 0
        for (bool_expression, weight) in zip(self.__bool_expressions, self.__weights):
            if bool_expression.is_true():
                result += weight
        return result
