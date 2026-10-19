"""Options of compiling and solving."""

from typing import NamedTuple


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
    eval_gradient: bool = True
    """Whether evaluating computes the gradient as well, which SCIP usually asks for next."""
    reverse_propagation: bool = True
    """Whether bounds on a function tighten the bounds of its inputs."""
    hessian: bool = True
    """Whether SCIP gets exact Hessian-vector products of each function."""
    zero_copy: bool = True
    """Whether executions read and write host memory directly when the plugin allows it."""
