class Boolean_Expression:

    def is_true(self):
        pass # method not implemented here, but in each sub-class

class Variable(Boolean_Expression):

    def __init__(self):
        self.value = False

    def is_true(self):
        return self.value

class And(Boolean_Expression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        for bool_expression in self.__bool_expressions:
            if not bool_expression.is_true():
                return False
        return True

class Eq(Boolean_Expression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() == self.__arith_expression2.value()

class Iff(Boolean_Expression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        joint_value = self.__bool_expressions[0].is_true()
        for bool_expression in self.__bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True

class Gt_Eq(Boolean_Expression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() >= self.__arith_expression2.value()

class Implies(Boolean_Expression):

    def __init__(self, bool_expression1, bool_expression2):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self):
        return not (self.__bool_expression1.is_true() and not self.__bool_expression2.is_true())

class Lt_Eq(Boolean_Expression):

    def __init__(self, arith_expression1, arith_expression2):
        self.__arith_expression1 = arith_expression1
        self.__arith_expression2 = arith_expression2

    def is_true(self):
        return self.__arith_expression1.value() <= self.__arith_expression2.value()

class Not(Boolean_Expression):

    def __init__ (self, bool_expression):
        self.__expression = bool_expression

    def is_true(self):
        return not self.__expression.is_true()

class Or(Boolean_Expression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def is_true(self):
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                return True
        return False

class Xor(Boolean_Expression):

    def __init__(self, bool_expression1, bool_expression2):
        self.__bool_expression1 = bool_expression1
        self.__bool_expression2 = bool_expression2

    def is_true(self):
        return self.__bool_expression1.is_true() != self.__bool_expression2.is_true()

class Arithmetic_Expression:

    def value(self):
        pass # method not implemented here, but in each sub-class

class Numeric_Constant(Arithmetic_Expression):

    def __init__(self, value):
        self.__value = value

    def value(self):
        return self.__value

class Sum(Arithmetic_Expression):

    def __init__(self, bool_expressions):
        self.__bool_expressions = bool_expressions

    def value(self):
        result = 0
        for bool_expression in self.__bool_expressions:
            if bool_expression.is_true():
                result += 1
        return result

class Weighted_Sum(Arithmetic_Expression):

    def __init__(self, bool_expressions, weights):
        self.__bool_expressions = bool_expressions
        self.__weights = weights

    def value(self):
        result = 0
        for (bool_expression, weight) in zip(self.__bool_expressions, self.__weights):
            if bool_expression.is_true():
                result += weight
        return result
