"""SMT expressions

Classes representing logical and arithmetic expressions that enable users to formulate constraints
for feature selection. The object-oriented design allows expressions to be nested and their
evaluation to be dispatched dynamically.

Literature
----------
- Bach et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"
- Barrett & Tinelli (2018): "Satisfiability Modulo Theories"
"""

from __future__ import annotations  # to use a class as a type hint within its own definition

from abc import ABCMeta, abstractmethod
from typing import Sequence


class Expression(metaclass=ABCMeta):
    """SMT expression

    Should evaluate to an arithmetic or boolean value (subclasses should implement corresponding
    evaluation methods). Can nest multiple child expressions.
    """

    def get_children(self) -> Sequence[Expression]:
        """Get child expressions

        This function allows to iterate over the expression tree down to the leaves (expressions
        without children).

        Returns
        -------
        Sequence[Expression]
            The child expressions. By default, there are none (e.g., for a constant value).
            For arithmetic or logical operators, the child expressions are its operands, and the
            method should be overridden accordingly in subclasses. Also, you should define an
            initializer to set the child expressions.
        """

        return []


class BooleanExpression(Expression, metaclass=ABCMeta):
    """Boolean expression

    Evaluates to true or false.
    """

    @abstractmethod
    def is_true(self) -> bool:
        """Evaluate boolean expression

        Should determine the value of the boolean expression, typically by evaluating its child
        expressions (operands) first and then applying the operator represented by this expression.
        See the corresponding subclass description for details.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (implementation depends on logical operator).

        Returns
        -------
        bool
            Boolean value of the expression.
        """

        raise NotImplementedError('Abstract method.')


class BooleanValue(BooleanExpression):
    """Boolean value

    Elementary logical expression that evaluates to a user-provided boolean value. Does not have
    child expressions.
    """

    def __init__(self, value: bool):
        self.value = value

    def is_true(self) -> bool:
        return self.value

    def __bool__(self) -> bool:
        return self.is_true()


class Variable(BooleanValue):
    """Boolean variable

    A boolean value with an additional name attribute. Represents a decision variable in the
    optimization problem of constrained feature selection.
    """

    def __init__(self, name: str):
        super().__init__(value=False)
        self.name = name

    def get_name(self) -> str:
        """Get name

        Returns
        -------
        str
            The name of the variable, as set during initialization.
        """

        return self.name

    def __str__(self) -> str:
        return self.name


class And(BooleanExpression):
    """Logical AND

    Evaluates to true if and only if all its child expressions evaluate to true. Can have an
    arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.bool_expressions:
            if not bool_expression.is_true():
                return False
        return True  # no child expression evaluates to False

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Iff(BooleanExpression):
    """Logical IFF (equivalence)

    Evaluates to true if and only if all its child expressions evaluate to the same value, i.e.,
    all evaluate to true or all evaluate to false. Can have an arbitrary number of child
    expressions.
    """

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        joint_value = self.bool_expressions[0].is_true()
        for bool_expression in self.bool_expressions[1:]:
            if bool_expression.is_true() != joint_value:
                return False
        return True  # all child expressions evaluate to same value

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Implies(BooleanExpression):
    """Logical IF (implication)

    Evaluates to true if and only if its first child expression evaluates to false or its second
    child expression evaluates to true. Has exactly two child expressions.
    """

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return (not self.bool_expression1.is_true()) or self.bool_expression2.is_true()

    def get_children(self) -> Sequence[Expression]:
        return [self.bool_expression1, self.bool_expression2]


class Not(BooleanExpression):
    """Logical NOT (negation)

    Evaluates to true if and only if its child expression evaluates to false. Has exactly one child
    expression.
    """

    def __init__(self, bool_expression: BooleanExpression):
        self.bool_expression = bool_expression

    def is_true(self) -> bool:
        return not self.bool_expression.is_true()

    def get_children(self) -> Sequence[Expression]:
        return [self.bool_expression]


class Or(BooleanExpression):
    """Logical (inclusive) OR

    Evaluates to true if and only if at least one of its child expression evaluates to true. Can
    have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[BooleanExpression]):
        self.bool_expressions = bool_expressions

    def is_true(self) -> bool:
        for bool_expression in self.bool_expressions:
            if bool_expression.is_true():
                return True
        return False  # no child expression evaluates to True

    def get_children(self) -> Sequence[Expression]:
        return self.bool_expressions


class Xor(BooleanExpression):
    """Logical exclusive OR

    Evaluates to true if and only if its two child expressions evaluate to different boolean
    values. Has exactly two child expressions.
    """

    def __init__(self, bool_expression1: BooleanExpression, bool_expression2: BooleanExpression):
        self.bool_expression1 = bool_expression1
        self.bool_expression2 = bool_expression2

    def is_true(self) -> bool:
        return self.bool_expression1.is_true() != self.bool_expression2.is_true()

    def get_children(self) -> Sequence[Expression]:
        return [self.bool_expression1, self.bool_expression2]


class ArithmeticExpression(Expression, metaclass=ABCMeta):
    """Arithmetic expression

    Evaluates to a numeric value.
    """

    @abstractmethod
    def get_value(self) -> float:
        """Evaluate arithmetic expression

        Should determine the value of the arithmetic expression, typically by evaluating its child
        expressions (operands) first and then applying the operator represented by this expression.
        See the corresponding subclass description for details.

        Raises
        ------
        NotImplementedError
            Always raised since abstract method (implementation depends on arithmetic operator).

        Returns
        -------
        float
            Numeric value of the expression.
        """

        raise NotImplementedError('Abstract method.')


class NumericValue(ArithmeticExpression):
    """Numeric value

    Elementary arithmetic expression that evaluates to a user-provided numeric value. Does not
    have child expressions.
    """

    def __init__(self, value: float):
        self.value = value

    def get_value(self) -> float:
        return self.value


class Eq(BooleanExpression):
    """Equality (==) of numeric values

    Evaluates to true if and only if its two arithmetic child expressions evaluate to the same
    numeric value. Has exactly two child expressions.
    """

    def __init__(self, arith_expression1: ArithmeticExpression,
                 arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() == self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Ge(BooleanExpression):
    """Greater-or-equal (>=) of numeric values

    Evaluates to true if and only if its first arithmetic child expressions evaluate to the same
    or a greater numeric value than the second. Has exactly two child expressions.
    """

    def __init__(self, arith_expression1: ArithmeticExpression,
                 arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() >= self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Le(BooleanExpression):
    """Less-or-equal (<=) of numeric values

    Evaluates to true if and only if its first arithmetic child expressions evaluate to the same
    or a smaller numeric value than the second. Has exactly two child expressions.
    """

    def __init__(self, arith_expression1: ArithmeticExpression,
                 arith_expression2: ArithmeticExpression):
        self.arith_expression1 = arith_expression1
        self.arith_expression2 = arith_expression2

    def is_true(self) -> bool:
        return self.arith_expression1.get_value() <= self.arith_expression2.get_value()

    def get_children(self) -> Sequence[Expression]:
        return [self.arith_expression1, self.arith_expression2]


class Sum(ArithmeticExpression):
    """Sum (+) of boolean values

    Evaluates to a numeric value that equals the sum of the values to which its child expressions
    evaluate. Can have an arbitrary number of child expressions. Note that this expression only
    supports boolean child expressions, since the constraints we analyzed did not require summing
    actual numeric values.
    """

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
    """Weighted sum of numeric values

    Evaluates to a numeric value that equals the sum of the values to which its child expressions
    evaluate multiplied by user-provided weights (one for each child expression). Can have an
    arbitrary number of child expressions. Note that this expression only supports boolean child
    expressions, since the constraints we analyzed did not require summing actual numeric values.
    """

    def __init__(self, bool_expressions: Sequence[BooleanExpression], weights: Sequence[float]):
        assert len(bool_expressions) == len(weights)
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
    """Get variables involved in expression

    Traverses the tree formed by the expression (i.e., all child expressions recursively) and
    extracts all boolean expressions representing decision variables.

    Parameters
    ----------
    expression : Expression
        The expression in which variables should be searched.

    Returns
    -------
    Sequence[Variable]
        The found variables. A variable may appear multiple times if it occurs in multiple
        subexpressions.
    """

    if isinstance(expression, Variable):
        return [expression]
    if len(expression.get_children()) == 0:
        return []
    result = []
    for child_expression in expression.get_children():
        result.extend(get_involved_variables(child_expression))
    return result
