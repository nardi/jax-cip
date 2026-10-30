"""Solving compiled problems with SCIP."""

import bisect
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from jaxtyping import Float, PyTree

from jaxcip._block import BlockHandle, BlockStateRef
from jaxcip._cache import BuildStatistics
from jaxcip._compile import (
    Block,
    CompiledProblem,
    CompiledTerm,
    LinearRows,
    block_inputs,
    compile,  # noqa: A004
    linear_rows,
    term_rows,
)
from jaxcip._handler import FunctionHandle
from jaxcip._options import CompileOptions, HandlerOptions, SolveOptions
from jaxcip._problem import Scip
from jaxcip._problem_types import Problem
from jaxcip._scip_parameters import ScipParameter
from jaxcip._types import (
    BlockTerm,
    CallbackStatistics,
    Domain,
    HandlerStatistics,
    JaxTerm,
    ParameterType,
    Sense,
    Status,
)

SETTING_TYPES: dict[ParameterType, tuple[type, ...]] = {
    ParameterType.BOOL: (bool,),
    ParameterType.INT: (int,),
    ParameterType.LONGINT: (int,),
    ParameterType.REAL: (int, float),
    ParameterType.CHAR: (str,),
    ParameterType.STRING: (str,),
}
"""Python types accepted for each type of SCIP parameter."""


class Solution(NamedTuple):
    """The result of solving a problem."""

    x: PyTree[np.ndarray] | None
    """The best solution in the structure of the variables, or `None` if none was found."""
    objective: float
    """Objective value of `x`, or infinity in the direction of optimization without one."""
    status: Status
    dual_bound: float
    """Bound on the optimal objective value that SCIP proved."""
    gap: float
    """Relative gap between `objective` and `dual_bound`."""
    nodes: int
    """Branch-and-bound nodes processed."""
    solving_seconds: float
    statistics: HandlerStatistics
    """Callback counters of the expression handler, summed over all functions."""
    build: BuildStatistics
    """Time and counts of compiling the problem, zero if it was compiled before."""


def solve(
    problem: CompiledProblem | Problem,
    params: PyTree[Any] = None,
    options: SolveOptions = SolveOptions(),
    compile_options: CompileOptions = CompileOptions(),
) -> Solution:
    """Solves a problem for the given parameter values.

    A `Problem` is compiled first with `compile_options`. Terms compiled earlier in the
    process are reused, so this is cheap after the first call. A `CompiledProblem` ignores
    `compile_options`.
    """
    build = BuildStatistics()
    if isinstance(problem, Problem):
        problem = compile(problem, params, compile_options)
        build = problem.statistics
    scip = build_scip(problem, params)
    apply_options(scip, options)
    handles = distinct_handles(problem)
    before = [handle.statistics() for handle in handles]
    scip.solve()
    after = [handle.statistics() for handle in handles]
    best = scip.best_solution()
    x = None
    if best is not None:
        x = problem.variables.unflatten(best[1][: problem.variables.size])
    return Solution(
        x=x,
        objective=scip.primal_bound,
        status=scip.status,
        dual_bound=scip.dual_bound,
        gap=scip.gap,
        nodes=scip.nodes,
        solving_seconds=scip.solving_time,
        statistics=statistics_difference(before, after),
        build=build,
    )


class FileSolution(NamedTuple):
    """The result of solving a problem file with SCIP alone."""

    x: Float[np.ndarray, " n"] | None
    """Values of the variables in file order, or `None` if no solution was found."""
    objective: float
    status: Status
    dual_bound: float
    gap: float
    nodes: int
    solving_seconds: float


def solve_file(path: Path, options: SolveOptions = SolveOptions()) -> FileSolution:
    """Solves a problem file with SCIP's own readers and expression handlers.

    The result is a reference for JAX-CIP's results on the same problem.
    """
    scip = Scip.read(path)
    apply_options(scip, options)
    scip.solve()
    best = scip.best_solution()
    return FileSolution(
        x=None if best is None else best[1],
        objective=scip.primal_bound,
        status=scip.status,
        dual_bound=scip.dual_bound,
        gap=scip.gap,
        nodes=scip.nodes,
        solving_seconds=scip.solving_time,
    )


def build_scip(compiled: CompiledProblem, params: PyTree[Any]) -> Scip:
    """Returns a SCIP problem with the variables and rows of a compiled problem."""
    param_leaves = compiled.params.leaves(params)
    variables = compiled.variables
    handler = compiled.options.handler
    scip = Scip("jaxcip")
    scip.set_sense(compiled.sense)
    objective = np.zeros(variables.size)
    offset = 0.0
    objective_block = compiled.objective
    if objective_block is not None:
        linear = linear_rows(objective_block, variables, param_leaves)
        indices, coefficients = linear.row(0)
        np.add.at(objective, indices, coefficients)
        offset = float(linear.constants[0])
    for lower, upper, coefficient, domain in zip(
        variables.lower, variables.upper, objective, flat_domains(compiled), strict=True
    ):
        scip.add_variable(float(lower), float(upper), float(coefficient), domain)
    scip.add_objective_offset(offset)
    if objective_block is not None and objective_block.terms:
        epigraph = scip.add_variable(-math.inf, math.inf, 1.0, Domain.CONTINUOUS)
        match compiled.sense:
            case Sense.MINIMIZE:
                bounds = (-math.inf, 0.0)
            case Sense.MAXIMIZE:
                bounds = (0.0, math.inf)
            case _:
                raise AssertionError(f"Unknown sense {compiled.sense}.")
        terms = row_terms(objective_block, variables, param_leaves, handler, linear)[0]
        scip.add_jax_constraint(
            terms,
            np.array([epigraph], np.intc),
            np.array([-1.0]),
            0.0,
            *bounds,
            handler,
        )
    for block in compiled.constraints:
        add_block(scip, block, compiled, param_leaves, handler)
    return scip


def flat_domains(compiled: CompiledProblem) -> list[Domain]:
    """Returns the domain of every scalar variable."""
    variables = compiled.variables
    return [
        domain
        for shape, domain in zip(variables.shapes, variables.domains, strict=True)
        for _ in range(math.prod(shape))
    ]


def row_terms(
    block: Block,
    variables: Any,
    param_leaves: list[np.ndarray],
    handler: HandlerOptions,
    linear: LinearRows,
) -> list[list[JaxTerm | BlockTerm]]:
    """Returns the nonlinear terms of every row of a block, with their coefficients.

    A row is a sum of entries of the cuts, each with a coefficient, and a linear part. An
    entry of a block term shares its state with the other entries in the new SCIP instance.
    """
    offsets = [0]
    makers: list[Callable[[int, float], JaxTerm | BlockTerm]] = []
    for compiled in block.terms:
        offsets.append(offsets[-1] + compiled.entries)
        makers.append(term_maker(compiled, block, variables, param_leaves, handler))
    rows: list[list[JaxTerm | BlockTerm]] = []
    for row in range(block.rows):
        columns, coefficients = linear.cut_row(row)
        terms: list[JaxTerm | BlockTerm] = []
        for column, coefficient in zip(columns, coefficients, strict=True):
            cut = bisect.bisect_right(offsets, int(column)) - 1
            terms.append(makers[cut](int(column) - offsets[cut], float(coefficient)))
        rows.append(terms)
    return rows


def term_maker(
    compiled: CompiledTerm,
    block: Block,
    variables: Any,
    param_leaves: list[np.ndarray],
    handler: HandlerOptions,
) -> Callable[[int, float], JaxTerm | BlockTerm]:
    """Returns a function that makes the term of one entry of a cut with a coefficient."""
    if isinstance(compiled.handle, BlockHandle):
        block_variables, block_params = block_inputs(
            compiled, block, variables, param_leaves
        )
        state = BlockStateRef(compiled.handle, block_params, block_variables, handler)
        return lambda entry, coefficient: BlockTerm(state, entry, coefficient)
    handle = compiled.handle
    assert isinstance(handle, FunctionHandle)
    inputs, params = term_rows(compiled, block, variables, param_leaves)
    return lambda entry, coefficient: JaxTerm(
        handle, params[entry], inputs[entry], coefficient
    )


def add_block(
    scip: Scip,
    block: Block,
    compiled: CompiledProblem,
    param_leaves: list[np.ndarray],
    handler: HandlerOptions,
) -> None:
    """Adds the rows of a constraint block."""
    variables = compiled.variables
    linear = linear_rows(block, variables, param_leaves)
    constant = linear.constants
    terms = row_terms(block, variables, param_leaves, handler, linear)
    for row in range(block.rows):
        lower, upper = block.lower[row], block.upper[row]
        if lower == -math.inf and upper == math.inf:
            continue
        row_indices, row_coefficients = linear.row(row)
        if terms[row]:
            scip.add_jax_constraint(
                terms[row],
                row_indices,
                row_coefficients,
                float(constant[row]),
                float(lower),
                float(upper),
                handler,
            )
        else:
            scip.add_linear_constraint(
                row_indices,
                row_coefficients,
                float(lower - constant[row]),
                float(upper - constant[row]),
            )


def apply_options(scip: Scip, options: SolveOptions) -> None:
    """Applies the emphasis, limits, verbosity and further settings of a solve."""
    scip.set_emphasis(options.emphasis)
    if options.time_limit is not None:
        scip.set_parameter(ScipParameter.LIMITS_TIME.path, float(options.time_limit))
    if options.gap_limit is not None:
        scip.set_parameter(ScipParameter.LIMITS_GAP.path, float(options.gap_limit))
    if options.node_limit is not None:
        scip.set_parameter(ScipParameter.LIMITS_NODES.path, int(options.node_limit))
    scip.set_parameter(ScipParameter.DISPLAY_VERBLEVEL.path, int(options.verbosity))
    for setting in options.settings:
        parameter = setting.parameter
        accepted = SETTING_TYPES[parameter.parameter_type]
        is_bool_as_number = isinstance(setting.value, bool) and bool not in accepted
        if not isinstance(setting.value, accepted) or is_bool_as_number:
            raise TypeError(
                f"{parameter.name} takes a {parameter.parameter_type.name.lower()} value,"
                f" not {setting.value!r}."
            )
        scip.set_parameter(parameter.path, setting.value)


def distinct_handles(
    compiled: CompiledProblem,
) -> list[FunctionHandle | BlockHandle]:
    """Returns every handle of a compiled problem once."""
    blocks = [compiled.objective, *compiled.constraints]
    handles = {
        id(term.handle): term.handle
        for block in blocks
        if block is not None
        for term in block.terms
    }
    return list(handles.values())


def statistics_difference(
    before: list[HandlerStatistics], after: list[HandlerStatistics]
) -> HandlerStatistics:
    """Returns the callback counters accumulated between two snapshots, summed over handles."""
    totals = [CallbackStatistics(0, 0, 0, 0.0) for _ in HandlerStatistics._fields]
    for first, second in zip(before, after, strict=True):
        totals = [
            CallbackStatistics(
                calls=total.calls + new.calls - old.calls,
                hits=total.hits + new.hits - old.hits,
                runs=total.runs + new.runs - old.runs,
                seconds=total.seconds + new.seconds - old.seconds,
            )
            for total, old, new in zip(totals, first, second, strict=True)
        ]
    return HandlerStatistics(*totals)
