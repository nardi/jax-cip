"""Descriptions of optimization problems over pytrees of variables."""

import math
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

from jaxtyping import Array, ArrayLike, Float, PyTree

from jaxcip._types import Domain, Sense


class ParametricFunction(Protocol):
    """A JAX function `f(x, params)` of the variables and the parameters of a problem.

    `x` has the structure of `Problem.variables` with an array in place of every `Variable`.
    `params` is the pytree passed to `compile` and `solve`.
    """

    def __call__(self, x: Any, params: Any, /) -> Float[Array, "*shape"]: ...


class RowFunction(Protocol):
    """A JAX function `f(x, params)` of one row of `rows`.

    `x` is the vector of the row's entries and `params` is the row's slice of the
    parameters of `rows`.
    """

    def __call__(self, x: Float[Array, " k"], params: Any, /) -> Float[Array, ""]: ...


class Variable(NamedTuple):
    """An array of decision variables with bounds and a domain.

    Its shape is the broadcast of `shape` and the shapes of the bounds. Estimators of a
    nonlinear function need finite bounds on its variables.
    """

    lower: ArrayLike = -math.inf
    upper: ArrayLike = math.inf
    domain: Domain = Domain.CONTINUOUS
    shape: tuple[int, ...] = ()


class Constraint(NamedTuple):
    """The constraint `lower ≤ function(x, params) ≤ upper`, one row per output entry.

    `lower` and `upper` broadcast to the shape of the output.
    """

    function: ParametricFunction
    lower: ArrayLike = -math.inf
    upper: ArrayLike = math.inf


class Problem(NamedTuple):
    """An optimization problem over a pytree of variables.

    Without an objective the problem is a feasibility problem.
    """

    variables: PyTree[Variable]
    """A pytree with a `Variable` at every leaf, which fixes the structure of `x`."""
    objective: ParametricFunction | None = None
    """A function with a scalar output."""
    sense: Sense = Sense.MINIMIZE
    constraints: tuple[Constraint, ...] = ()


def is_variable(value: object) -> bool:
    """Returns whether a value is a `Variable`, for use as the `is_leaf` of tree functions."""
    return isinstance(value, Variable)


Unflatten = Callable[[Float[Array, " n"]], Any]
"""Maps the flat vector of all variables to the structure of `Problem.variables`."""
