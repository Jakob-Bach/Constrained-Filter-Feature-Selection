class Expression:

    def is_true(self):
        pass # method not implemented here, but in each sub-class

class Variable(Expression):

    def __init__(self):
        self.value = False

    def is_true(self):
        return self.value

class And(Expression):

    def __init__(self, expressions):
        self.__expressions = expressions

    def is_true(self):
        for expression in self.__expressions:
            if not expression.is_true():
                return False
        return True

class Iff(Expression):

    def __init__(self, expressions):
        self.__expressions = expressions

    def is_true(self):
        joint_value = self.__expressions[0].is_true()
        for expression in self.__expressions[1:]:
            if expression.is_true() != joint_value:
                return False
        return True

class Implies(Expression):

    def __init__(self, expression1, expression2):
        self.__expression1 = expression1
        self.__expression2 = expression2

    def is_true(self):
        return not (self.__expression1.is_true() and not self.__expression2.is_true())

class Not(Expression):

    def __init__ (self, expression):
        self.__expression = expression

    def is_true(self):
        return not self.__expression.is_true()

class Or(Expression):

    def __init__(self, expressions):
        self.__expressions = expressions

    def is_true(self):
        for expression in self.__expressions:
            if expression.is_true():
                return True
        return False

class Xor(Expression):

    def __init__(self, expression1, expression2):
        self.__expression1 = expression1
        self.__expression2 = expression2

    def is_true(self):
        return (self.__expression1.is_true() and not self.__expression2.is_true()) or\
               (not self.__expression1.is_true() and self.__expression2.is_true())
