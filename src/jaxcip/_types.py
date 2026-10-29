"""Enumerations and records shared by the extension modules and the Python layer."""

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from cyxla import Executable

    from jaxcip._block import BlockStateRef
    from jaxcip._handler import FunctionHandle


class Sense(IntEnum):
    """Direction of optimization, with SCIP's `SCIP_OBJSENSE` values."""

    MAXIMIZE = -1
    MINIMIZE = 1


class Domain(IntEnum):
    """Values a variable may take, with SCIP's `SCIP_VARTYPE` values."""

    BINARY = 0
    INTEGER = 1
    CONTINUOUS = 3


class Status(IntEnum):
    """Why SCIP stopped solving, with SCIP's `SCIP_STATUS` values."""

    UNKNOWN = 0
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    INFEASIBLE_OR_UNBOUNDED = 4
    USER_INTERRUPT = 10
    TERMINATE = 11
    NODE_LIMIT = 20
    TOTAL_NODE_LIMIT = 21
    STALL_NODE_LIMIT = 22
    TIME_LIMIT = 23
    MEMORY_LIMIT = 24
    GAP_LIMIT = 25
    PRIMAL_LIMIT = 26
    DUAL_LIMIT = 27
    SOLUTION_LIMIT = 28
    BEST_SOLUTION_LIMIT = 29
    RESTART_LIMIT = 30


class Emphasis(IntEnum):
    """SCIP's predefined parameter settings, with its `SCIP_PARAMEMPHASIS` values."""

    DEFAULT = 0
    CP_SOLVER = 1
    EASY_CIP = 2
    FEASIBILITY = 3
    HARD_LP = 4
    OPTIMALITY = 5
    COUNTER = 6
    PHASE_FEASIBILITY = 7
    PHASE_IMPROVE = 8
    PHASE_PROOF = 9
    NUMERICS = 10
    BENCHMARK = 11


class Verbosity(IntEnum):
    """How much SCIP prints, as the values of its `display/verblevel` parameter."""

    QUIET = 0
    ERRORS = 1
    WARNINGS = 2
    NORMAL = 3
    VERBOSE = 4
    FULL = 5


class ParameterType(IntEnum):
    """Value types of SCIP parameters, with SCIP's `SCIP_PARAMTYPE` values."""

    BOOL = 0
    INT = 1
    LONGINT = 2
    REAL = 3
    CHAR = 4
    STRING = 5


class Callback(IntEnum):
    """Callbacks of the expression handler, in the order of its counters."""

    EVAL = 0
    """Evaluates the function."""
    BWDIFF = 1
    """Returns one partial derivative."""
    FWDIFF = 2
    """Returns a directional derivative."""
    BWFWDIFF = 3
    """Returns one entry of a Hessian-vector product."""
    INTEVAL = 4
    """Encloses the range of the function over a box."""
    ESTIMATE = 5
    """Returns a linear under- or overestimator at a point."""
    INITESTIMATES = 6
    """Returns linear estimators for the initial LP relaxation."""
    MONOTONICITY = 7
    """Reports monotonicity in one input."""
    REVERSEPROP = 8
    """Tightens the bounds of the inputs from bounds on the function."""


class Program(IntEnum):
    """The programs compiled for each function, in the order of `jaxcip._handler.Programs`."""

    VALUE_AND_GRADIENT = 0
    """Maps `(x, params)` to the value and the gradient in `x`."""
    HESSIAN_VECTOR = 1
    """Maps `(x, direction, params)` to the Hessian in `x` times `direction`."""
    CONVEX = 2
    """Maps `(x, lower, upper, params)` to a convex relaxation's value and subgradient."""
    CONCAVE = 3
    """Maps `(x, lower, upper, params)` to a concave relaxation's value and supergradient."""
    INTERVAL = 4
    """Maps `(lower, upper, params)` to bounds on the value over the box."""
    GRADIENT_INTERVAL = 5
    """Maps `(lower, upper, params)` to bounds on the gradient over the box."""


class CallbackStatistics(NamedTuple):
    """Counters of one callback of the expression handler."""

    calls: int
    """Times SCIP called the callback."""
    hits: int
    """Calls answered from a cache."""
    runs: int
    """Program executions."""
    seconds: float
    """Seconds spent in program executions."""


class HandlerStatistics(NamedTuple):
    """Counters of every callback of the expression handler for one function."""

    eval: CallbackStatistics
    bwdiff: CallbackStatistics
    fwdiff: CallbackStatistics
    bwfwdiff: CallbackStatistics
    inteval: CallbackStatistics
    estimate: CallbackStatistics
    initestimates: CallbackStatistics
    monotonicity: CallbackStatistics
    reverseprop: CallbackStatistics


class Estimator(NamedTuple):
    """A linear estimator `constant + coefficients · x` of a function over a box."""

    success: bool
    """Whether the estimator is finite and its coefficients are in range."""
    constant: float
    coefficients: Float[np.ndarray, " n"]


class Interval(NamedTuple):
    """Bounds `lower ≤ y ≤ upper` on a scalar or on each entry of an array."""

    lower: float | Float[np.ndarray, " n"]
    upper: float | Float[np.ndarray, " n"]


class ScipVersion(NamedTuple):
    """Version of the linked SCIP library."""

    major: int
    minor: int
    technical: int


class Monotonicity(Enum):
    """Monotonicity of a function in one input, with SCIP's `SCIP_MONOTONE` values."""

    UNKNOWN = 0
    INCREASING = 1
    DECREASING = 2
    CONSTANT = 3


class Programs(NamedTuple):
    """The executables of one function, in the order of `Program`.

    The Hessian-vector and the gradient interval programs are optional. Without them the
    handler leaves Hessians and monotonicity to SCIP's defaults.
    """

    value_and_gradient: "Executable"
    hessian_vector: "Executable | None"
    convex: "Executable"
    concave: "Executable"
    interval: "Executable"
    gradient_interval: "Executable | None"


class JaxTerm(NamedTuple):
    """One term `coefficient · f(x[inputs], params)` of a constraint, with `f` compiled."""

    handle: "FunctionHandle"
    params: Float[np.ndarray, " m"]
    """Parameter values of this term, as many as the handle takes."""
    inputs: Int[np.ndarray, " n"]
    """Indices of the term's input variables, as `np.intc`."""
    coefficient: float


class BlockTerm(NamedTuple):
    """One term `coefficient · f(x, params)[entry]` of a constraint, an entry of a block."""

    block: "BlockStateRef"
    """The state of the block in the SCIP instance, shared by all its entries."""
    entry: int
    coefficient: float
