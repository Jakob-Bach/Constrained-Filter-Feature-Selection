"""SMT expressions wrapping Z3

Classes combining our own SMT-expression types from :mod:`expressions` (used for counting number of
solutions in :mod:`solving`) and equivalent Z3 expressions (used for optimization in
:mod:`combi_solving`) to have a uniform interface and avoid inconsistencies.

Literature
----------
- Bach et al. (2022): "An Empirical Evaluation of Constrained Feature Selection"
- Barrett & Tinelli (2018): "Satisfiability Modulo Theories"
- de Moura & Bjorner (2008): "Z3: An Efficient SMT Solver"
"""

from abc import ABCMeta
from typing import Sequence

import z3

from . import expressions as expr


class BooleanExpression(expr.BooleanExpression, metaclass=ABCMeta):
    """Boolean expression

    Evaluates to true or false. Subclasses need to implement the corresponding logic via the method
    :meth:`is_true`, defining an operator that combines all child expressions (operands).
    """

    def get_z3(self) -> z3.BoolRef:
        """Get wrapped Z3 expression

        Returns
        -------
        z3.BoolRef
            Get the Z3 representation of the expression. For this method to work, subclasses should
            set the field :attr:`z3_expr`, ideally in their initializer.
        """

        return self.z3_expr


class BooleanValue(expr.BooleanValue, BooleanExpression):
    """Boolean value

    Elementary logical expression that evaluates to a user-provided boolean value. Does not have
    child expressions.
    """

    def __init__(self, value: bool):
        super().__init__(value=value)
        self.z3_expr = z3.BoolVal(value)


class Variable(expr.Variable, BooleanExpression):
    """Boolean variable

    A boolean value with an additional name attribute. Represents a decision variable in the
    optimization problem of constrained feature selection.
    """

    def __init__(self, name: str):
        super().__init__(name=name)
        self.z3_expr = z3.Bool(name)


class And(expr.And, BooleanExpression):
    """Logical AND

    Evaluates to true if and only if all its child expressions evaluate to true. Can have an
    arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.And([x.get_z3() for x in bool_expressions])


class AtLeast(expr.Ge, BooleanExpression):
    """At-least cardinality constraint

    Evaluates to true if and only if at least the user-provided number of child expressions
    evaluates to true. Can have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: int):
        super().__init__(expr.Sum(bool_expressions), expr.NumericValue(value))
        self.z3_expr = z3.AtLeast(*[e.get_z3() for e in bool_expressions], value)


class AtMost(expr.Le, BooleanExpression):
    """At-most cardinality constraint

    Evaluates to true if and only if at most the user-provided number of child expressions
    evaluates to true. Can have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression], value: int):
        super().__init__(expr.Sum(bool_expressions), expr.NumericValue(value))
        self.z3_expr = z3.AtMost(*[e.get_z3() for e in bool_expressions], value)


class Iff(expr.Iff, BooleanExpression):
    """Logical IFF (equivalence)

    Evaluates to true if and only if all its child expressions evaluate to the same value, i.e.,
    all evaluate to true or all evaluate to false. Can have an arbitrary number of child
    expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.Or(z3.And([x.get_z3() for x in bool_expressions]),
                             z3.Not(z3.Or([x.get_z3() for x in bool_expressions])))


class Implies(expr.Implies, BooleanExpression):
    """Logical IF (implication)

    Evaluates to true if and only if its first child expression evaluates to false or its second
    child expression evaluates to true. Has exactly two child expressions.
    """

    def __init__(self, bool_expression1: expr.BooleanExpression,
                 bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3_expr = z3.Implies(bool_expression1.get_z3(), bool_expression2.get_z3())


class Not(expr.Not, BooleanExpression):
    """Logical NOT (negation)

    Evaluates to true if and only if its child expression evaluates to false. Has exactly one child
    expression.
    """

    def __init__(self, bool_expression: expr.BooleanExpression):
        super().__init__(bool_expression)
        self.z3_expr = z3.Not(bool_expression.get_z3())


class Or(expr.Or, BooleanExpression):
    """Logical (inclusive) OR

    Evaluates to true if and only if at least one of its child expression evaluates to true. Can
    have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression]):
        super().__init__(bool_expressions)
        self.z3_expr = z3.Or([x.get_z3() for x in bool_expressions])


class WeightedSumEq(expr.Eq, BooleanExpression):
    """Equality (==) of a weighted sum of boolean values

    Represents a linear equality with the form "w_1 * x_1 + ... + w_n * x_n == value", where the
    weights "w" are numeric constants and the variables "x" are boolean expressions. Evaluates to
    true if and only if the sum of its boolean child expressions multiplied by the user-provided
    weights evaluate to the user-provided threshold value. Can have an arbitrary number of child
    expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression],
                 weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbEq([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)],
                               value)


class WeightedSumGe(expr.Ge, BooleanExpression):
    """Greater-or-equal (>=) of a weighted sum of boolean values

    Represents a linear inequality with the form "w_1 * x_1 + ... + w_n * x_n >= value", where
    the weights "w" are numeric constants and the variables "x" are boolean expressions.
    Evaluates to true if and only if the sum of its boolean child expressions multiplied by the
    user-provided weights evaluate to the same or a greater numeric value than the user-provided
    threshold value. Can have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression],
                 weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbGe([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)],
                               value)


class WeightedSumLe(expr.Le, BooleanExpression):
    """Less-or-equal (<=) of a weighted sum of boolean values

    Represents a linear inequality with the form "w_1 * x_1 + ... + w_n * x_n <= value", where
    the weights "w" are numeric constants and the variables "x" are boolean expressions.
    Evaluates to true if and only if the sum of its boolean child expressions multiplied by the
    user-provided weights evaluate to the same or a smaller numeric value than the user-provided
    threshold value. Can have an arbitrary number of child expressions.
    """

    def __init__(self, bool_expressions: Sequence[expr.BooleanExpression],
                 weights: Sequence[float], value: float):
        super().__init__(expr.WeightedSum(bool_expressions, weights), expr.NumericValue(value))
        self.z3_expr = z3.PbLe([(e.get_z3(), w) for (e, w) in zip(bool_expressions, weights)],
                               value)


class Xor(expr.Xor, BooleanExpression):
    """Logical exclusive OR

    Evaluates to true if and only if its two child expressions evaluate to different boolean
    values. Has exactly two child expressions.
    """

    def __init__(self, bool_expression1: expr.BooleanExpression,
                 bool_expression2: expr.BooleanExpression):
        super().__init__(bool_expression1, bool_expression2)
        self.z3_expr = z3.Xor(bool_expression1.get_z3(), bool_expression2.get_z3())
