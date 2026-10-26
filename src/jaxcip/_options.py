"""Options of compiling and solving."""

from enum import Enum
from typing import NamedTuple

from jaxcip._scip_parameters import ScipParameter
from jaxcip._types import Emphasis, Verbosity


class HandlerOptions(NamedTuple):
    """Optional features of the expression handler, all on by default.

    Each switch exists so that its effect on solving can be measured. Switching one off never
    makes results wrong, only slower or less tight.
    """

    caches: bool = True
    """Whether each expression keeps its recent values, gradients, bounds and estimators."""
    init_estimates: bool = True
    """Whether the initial LP relaxation gets estimators at three points of the box."""
    monotonicity: bool = True
    """Whether monotonicity is reported from interval bounds of the gradient."""
    global_estimators: bool = True
    """Whether estimators over the global box are marked as globally valid."""
    reverse_propagation: bool = True
    """Whether bounds on a function tighten the bounds of its inputs."""
    hessian: bool = True
    """Whether SCIP gets exact Hessian-vector products of each function."""
    zero_copy: bool = True
    """Whether executions read and write host memory directly when the plugin allows it."""
    prefetch: bool = True
    """Whether the first entry of a block that needs a run fills the inputs of all entries.

    The run then answers the callbacks of the other entries at the same point or box."""


class RelaxationMode(Enum):
    """How the gradients of the convex and concave relaxations are computed."""

    FORWARD = "forward"
    """Forward mode, which gives smaller programs for functions of few inputs."""
    REVERSE = "reverse"
    """Reverse mode, which is cheaper when a function has many inputs."""


class RelaxationOptions(NamedTuple):
    """Options of the McCormick relaxations that JAXormick builds for each function."""

    mode: RelaxationMode = RelaxationMode.FORWARD
    alpha_bb: bool = False
    """Whether relaxations are intersected with αBB relaxations from Hessian bounds."""
    tighten_bounds: bool = False
    """Whether intermediate intervals are narrowed by affine relaxations at the box middle."""


class CompileOptions(NamedTuple):
    """Options of turning the functions of a problem into SCIP expressions."""

    relaxation: RelaxationOptions = RelaxationOptions()
    handler: HandlerOptions = HandlerOptions()
    split_sums: bool = True
    """Whether a sum of nonlinear terms becomes one SCIP expression per term."""
    gradient_interval_limit: int = 300
    """Largest function, in jaxpr equations, that gets gradient bounds for monotonicity.

    XLA takes tens of seconds to compile the interval derivative of large functions, and
    monotonicity is optional.
    """
    disk_cache: bool = True
    """Whether lowered programs and executables are stored in the cache directory.

    The directory is `JAXCIP_CACHE_DIR` if set, and `~/.cache/jaxcip` otherwise.
    """
    threads: int | None = None
    """Number of threads that lower and compile programs, or `None` for one per CPU core.

    With 1, the programs are built one after another on the calling thread.
    """


class ScipSetting(NamedTuple):
    """A value for one SCIP parameter."""

    parameter: ScipParameter
    value: bool | int | float | str


class SolveOptions(NamedTuple):
    """Limits and settings of one solve."""

    time_limit: float | None = None
    """Seconds after which SCIP stops, or `None` for no limit."""
    gap_limit: float | None = None
    """Relative gap at which SCIP stops, or `None` for SCIP's default."""
    node_limit: int | None = None
    """Number of nodes after which SCIP stops, or `None` for no limit."""
    emphasis: Emphasis = Emphasis.DEFAULT
    verbosity: Verbosity = Verbosity.QUIET
    settings: tuple[ScipSetting, ...] = ()
    """Further parameter values, applied after the fields above."""
