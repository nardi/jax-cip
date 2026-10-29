"""JAX functions as nonlinear expressions in the SCIP optimization solver."""

from jaxcip import _cache, _compile
from jaxcip._cache import BuildStatistics
from jaxcip._compile import CompiledProblem, compile  # noqa: A004
from jaxcip._options import (
    CompileOptions,
    HandlerOptions,
    RelaxationMode,
    RelaxationOptions,
    ScipSetting,
    SolveOptions,
)
from jaxcip._problem import forward_blas, scip_version
from jaxcip._problem_types import (
    Constraint,
    ParametricFunction,
    Problem,
    RowFunction,
    Variable,
)
from jaxcip._rows import rows
from jaxcip._scip_parameters import ScipParameter
from jaxcip._solve import FileSolution, Solution, solve, solve_file
from jaxcip._types import (
    CallbackStatistics,
    Domain,
    Emphasis,
    HandlerStatistics,
    ScipVersion,
    Sense,
    Status,
    Verbosity,
)

forward_blas()
# SCIP works in double precision, so arrays made after the import have 64-bit floats.
_cache.enable_x64()


def clear_caches() -> None:
    """Forgets the terms compiled in this process, so later builds use the disk cache."""
    _compile.compiled_terms.clear()


__all__ = [
    "BuildStatistics",
    "CallbackStatistics",
    "CompileOptions",
    "CompiledProblem",
    "Constraint",
    "Domain",
    "Emphasis",
    "FileSolution",
    "HandlerOptions",
    "HandlerStatistics",
    "ParametricFunction",
    "Problem",
    "RelaxationMode",
    "RelaxationOptions",
    "RowFunction",
    "ScipParameter",
    "ScipSetting",
    "ScipVersion",
    "Sense",
    "SolveOptions",
    "Solution",
    "Status",
    "Variable",
    "Verbosity",
    "clear_caches",
    "compile",
    "rows",
    "scip_version",
    "solve",
    "solve_file",
]
