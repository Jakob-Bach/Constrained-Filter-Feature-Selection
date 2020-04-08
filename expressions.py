class Expression:

    def is_true(self):
        pass # method not implemented here, but in each sub-class

class Variable(Expression):

    def __init__(self):
        self.value = False

    def is_true(self):
        return self.value

class Not(Expression):

    def __init__ (self, expression):
        self.__expression = expression

    def is_true(self):
        return not self.__expression.is_true()

class And(Expression):

    def __init__(self, expressions):
        self.__expressions = expressions

    def is_true(self):
        for expression in self.__expressions:
            if not expression.is_true():
                return False
        return True

class Or(Expression):

    def __init__(self, expressions):
        self.__expressions = expressions

    def is_true(self):
        for expression in self.__expressions:
            if expression.is_true():
                return True
        return False
