"""Compiling a problem into blocks of rows with compiled terms.

Compiling traces every function of a problem once, splits it into an affine part and
nonlinear terms, and builds the programs of each term. Nothing here depends on parameter
values, so one compiled problem solves for many of them.
"""

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from jaxcip import _cache
from jaxcip._analysis import Analysis, Origin, Term, analyse, evaluate_term
from jaxcip._handler import FunctionHandle
from jaxcip._options import CompileOptions
from jaxcip._problem_types import (
    BatchedConstraint,
    Constraint,
    ParametricFunction,
    Problem,
    is_variable,
)
from jaxcip._programs import (
    ProgramOptions,
    TermSignature,
    lower_programs,
    term_signature,
)
from jaxcip._types import Domain, Programs

COMPILED_TERMS_LIMIT = 256
"""Number of terms whose handles the process keeps after their problems are gone.

Every executable holds memory mappings of its machine code, and Linux allows about 65,000
mappings per process. A process that compiles many different problems would reach that
limit if it kept every term.
"""

compiled_terms: OrderedDict[str, FunctionHandle] = OrderedDict()
"""Handles of the terms compiled in this process, by `_cache.term_key`, oldest first."""


class VariableLayout(NamedTuple):
    """The variables of a problem as one flat vector."""

    structure: jax.tree_util.PyTreeDef
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    """Position of each leaf's first entry in the flat vector."""
    lower: Float[np.ndarray, " n"]
    upper: Float[np.ndarray, " n"]
    domains: tuple[Domain, ...]
    """Domain of each leaf."""

    @property
    def size(self) -> int:
        """Number of scalar variables."""
        return self.lower.shape[0]

    def indices(self, leaf: int) -> Int[np.ndarray, " size"]:
        """Returns the flat indices of the entries of one leaf."""
        size = math.prod(self.shapes[leaf])
        return np.arange(self.offsets[leaf], self.offsets[leaf] + size, dtype=np.intc)

    def unflatten(self, values: Float[np.ndarray, " n"]) -> PyTree[np.ndarray]:
        """Returns flat values in the structure of the variables.

        Values of integer and binary variables are rounded.
        """
        leaves = []
        for offset, shape, domain in zip(
            self.offsets, self.shapes, self.domains, strict=True
        ):
            leaf = values[offset : offset + math.prod(shape)].reshape(shape)
            leaves.append(leaf if domain is Domain.CONTINUOUS else np.round(leaf) + 0.0)
        return jax.tree.unflatten(self.structure, leaves)


class ParamsLayout(NamedTuple):
    """Structure, shapes and types of the parameters a problem was compiled for."""

    structure: jax.tree_util.PyTreeDef
    avals: tuple[jax.ShapeDtypeStruct, ...]

    def leaves(self, params: PyTree[Any]) -> list[np.ndarray]:
        """Returns the leaves of parameters, checking them against the layout."""
        leaves, structure = jax.tree.flatten(params)
        if structure != self.structure:
            raise ValueError(
                f"The parameters have structure {structure}, compiled for {self.structure}."
            )
        arrays = [np.asarray(leaf) for leaf in leaves]
        for array, aval in zip(arrays, self.avals, strict=True):
            if array.shape != aval.shape:
                raise ValueError(
                    f"A parameter has shape {array.shape}, compiled for {aval.shape}."
                )
        return arrays


class CompiledTerm(NamedTuple):
    """A nonlinear term with its compiled programs."""

    term: Term
    signature: TermSignature
    handle: FunctionHandle


class Block(NamedTuple):
    """Rows of a problem that come from one function.

    A `Constraint` or the objective has one row per entry of its output and the same
    inputs in every row. A `BatchedConstraint` has one row per batch entry, each with its
    own inputs and parameters.
    """

    analysis: Analysis
    terms: tuple[CompiledTerm, ...]
    lower: Float[np.ndarray, " rows"]
    upper: Float[np.ndarray, " rows"]
    batch: BatchedConstraint | None
    """The batched constraint of the block, with `inputs` and `params` as arrays."""
    affine: Callable[..., tuple[Array, list[Array]]] | None
    """Maps parameter leaves to the constants and the Jacobians of the affine part."""

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.lower.shape[0]


class CompiledProblem(NamedTuple):
    """A problem whose functions are analysed and compiled, ready to solve."""

    problem: Problem
    options: CompileOptions
    variables: VariableLayout
    params: ParamsLayout
    objective: Block | None
    constraints: tuple[Block, ...]
    statistics: _cache.BuildStatistics


def compile(  # noqa: A001
    problem: Problem,
    params: PyTree[Any] = None,
    options: CompileOptions = CompileOptions(),
) -> CompiledProblem:
    """Compiles the functions of a problem for parameters of the given structure and shapes.

    `params` may hold arrays or `jax.ShapeDtypeStruct`s, since only their structure,
    shapes and types matter. The result solves with any parameter values of that layout.
    """
    _cache.enable_x64()
    _cache.initialize()
    variables = variable_layout(problem.variables)
    leaves, structure = jax.tree.flatten(params)
    params_layout = ParamsLayout(structure, tuple(as_aval(leaf) for leaf in leaves))
    statistics = _cache.BuildStatistics()
    objective = None
    if problem.objective is not None:
        objective, built = function_block(
            problem.objective, -math.inf, math.inf, variables, params_layout, options
        )
        statistics = statistics.add(built)
        if objective.analysis.shape != ():
            raise ValueError(
                f"The objective must return a scalar, not shape {objective.analysis.shape}."
            )
    blocks = []
    for constraint in problem.constraints:
        match constraint:
            case Constraint(function, lower, upper):
                block, built = function_block(
                    function, lower, upper, variables, params_layout, options
                )
            case BatchedConstraint():
                block, built = batched_block(constraint, variables, options)
            case _:
                raise TypeError(
                    "A constraint must be a Constraint or a BatchedConstraint, "
                    f"not {type(constraint).__name__}."
                )
        statistics = statistics.add(built)
        blocks.append(block)
    return CompiledProblem(
        problem=problem,
        options=options,
        variables=variables,
        params=params_layout,
        objective=objective,
        constraints=tuple(blocks),
        statistics=statistics,
    )


def as_aval(leaf: Any) -> jax.ShapeDtypeStruct:
    """Returns the shape and type of a parameter leaf."""
    if isinstance(leaf, jax.ShapeDtypeStruct):
        return leaf
    array = jnp.asarray(leaf)
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def variable_layout(variables: PyTree[Any]) -> VariableLayout:
    """Returns the flat layout of a pytree of `Variable`s."""
    leaves, structure = jax.tree.flatten(variables, is_leaf=is_variable)
    shapes: list[tuple[int, ...]] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    offsets: list[int] = []
    offset = 0
    for leaf in leaves:
        if not is_variable(leaf):
            raise TypeError(
                f"Every leaf of the variables must be a Variable, not {leaf!r}."
            )
        shape = np.broadcast_shapes(
            leaf.shape, np.shape(leaf.lower), np.shape(leaf.upper)
        )
        lower = np.broadcast_to(np.asarray(leaf.lower, np.float64), shape).ravel()
        upper = np.broadcast_to(np.asarray(leaf.upper, np.float64), shape).ravel()
        if leaf.domain is Domain.BINARY:
            lower, upper = np.maximum(lower, 0.0), np.minimum(upper, 1.0)
        if np.any(lower > upper):
            raise ValueError("A variable has a lower bound above its upper bound.")
        shapes.append(shape)
        lowers.append(lower)
        uppers.append(upper)
        offsets.append(offset)
        offset += lower.shape[0]
    return VariableLayout(
        structure=structure,
        shapes=tuple(shapes),
        offsets=tuple(offsets),
        lower=np.concatenate(lowers) if lowers else np.zeros(0),
        upper=np.concatenate(uppers) if uppers else np.zeros(0),
        domains=tuple(leaf.domain for leaf in leaves),
    )


def function_block(
    function: ParametricFunction,
    lower: Any,
    upper: Any,
    variables: VariableLayout,
    params: ParamsLayout,
    options: CompileOptions,
) -> tuple[Block, _cache.BuildStatistics]:
    """Returns the block of a function of all variables and parameters."""
    num_leaves = len(variables.shapes)

    def flat(*leaves: Array) -> Array:
        x = jax.tree.unflatten(variables.structure, leaves[:num_leaves])
        return jnp.asarray(
            function(x, jax.tree.unflatten(params.structure, leaves[num_leaves:]))
        )

    variable_avals = [
        jax.ShapeDtypeStruct(shape, jnp.float64) for shape in variables.shapes
    ]
    analysis = analyse(flat, variable_avals, params.avals, options.split_sums)
    terms, statistics = compile_terms(analysis, variable_avals, params.avals, options)
    affine = None
    if analysis.affine is not None:
        affine = jax.jit(affine_function(analysis, variable_avals, batched=False))
    return Block(
        analysis=analysis,
        terms=terms,
        lower=row_bounds(lower, analysis.shape),
        upper=row_bounds(upper, analysis.shape),
        batch=None,
        affine=affine,
    ), statistics


def batched_block(
    constraint: BatchedConstraint, variables: VariableLayout, options: CompileOptions
) -> tuple[Block, _cache.BuildStatistics]:
    """Returns the block of a batched constraint."""
    inputs = np.asarray(constraint.inputs, dtype=np.intc)
    if inputs.ndim != 2:
        raise ValueError(
            f"The inputs of a batched constraint need two axes, not {inputs.ndim}."
        )
    rows, width = inputs.shape
    if np.any(inputs < 0) or np.any(inputs >= variables.size):
        raise IndexError("The inputs of a batched constraint index missing variables.")
    leaves, structure = jax.tree.flatten(constraint.params)
    arrays = [np.asarray(leaf) for leaf in leaves]
    if any(array.ndim == 0 or array.shape[0] != rows for array in arrays):
        raise ValueError(f"Every parameter of a batched constraint needs {rows} rows.")
    row_avals = [jax.ShapeDtypeStruct(array.shape[1:], array.dtype) for array in arrays]

    def flat(x: Array, *row_params: Array) -> Array:
        return jnp.asarray(
            constraint.function(x, jax.tree.unflatten(structure, list(row_params)))
        )

    variable_avals = [jax.ShapeDtypeStruct((width,), jnp.float64)]
    analysis = analyse(flat, variable_avals, row_avals, options.split_sums)
    if analysis.shape != ():
        raise ValueError(
            f"The function of a batched constraint must return a scalar, not {analysis.shape}."
        )
    terms, statistics = compile_terms(analysis, variable_avals, row_avals, options)
    affine = None
    if analysis.affine is not None:
        affine = jax.jit(affine_function(analysis, variable_avals, batched=True))
    return Block(
        analysis=analysis,
        terms=terms,
        lower=row_bounds(constraint.lower, (rows,)),
        upper=row_bounds(constraint.upper, (rows,)),
        batch=constraint._replace(inputs=inputs, params=arrays),
        affine=affine,
    ), statistics


def row_bounds(bound: Any, shape: tuple[int, ...]) -> Float[np.ndarray, " rows"]:
    """Returns bounds broadcast to an output shape and flattened."""
    return np.broadcast_to(np.asarray(bound, np.float64), shape).ravel().copy()


def program_options(options: CompileOptions) -> ProgramOptions:
    """Returns the options that decide which programs a term gets."""
    return ProgramOptions(
        relaxation=options.relaxation,
        hessian=options.handler.hessian,
        gradient_interval=options.handler.monotonicity,
        gradient_interval_limit=options.gradient_interval_limit,
    )


def input_avals(
    term: Term,
    analysis: Analysis,
    variable_avals: Sequence[jax.ShapeDtypeStruct],
    param_avals: Sequence[jax.ShapeDtypeStruct],
) -> list[jax.ShapeDtypeStruct]:
    """Returns the shapes and types of the inputs of a term."""
    sources = {
        Origin.CONSTANT: [
            jax.ShapeDtypeStruct(constant.shape, constant.dtype)
            for constant in analysis.constants
        ],
        Origin.VARIABLE: list(variable_avals),
        Origin.PARAMETER: list(param_avals),
    }
    return [sources[ref.origin][ref.index] for ref in term.inputs]


def compile_terms(
    analysis: Analysis,
    variable_avals: Sequence[jax.ShapeDtypeStruct],
    param_avals: Sequence[jax.ShapeDtypeStruct],
    options: CompileOptions,
) -> tuple[tuple[CompiledTerm, ...], _cache.BuildStatistics]:
    """Builds the programs of the nonlinear terms of an analysis."""
    statistics = _cache.BuildStatistics()
    terms = []
    for term in analysis.nonlinear:
        avals = input_avals(term, analysis, variable_avals, param_avals)
        signature = term_signature(term, analysis.shape, avals)
        handle, built = build_handle(term, signature, options)
        statistics = statistics.add(built)
        terms.append(CompiledTerm(term, signature, handle))
    return tuple(terms), statistics


def build_handle(
    term: Term, signature: TermSignature, options: CompileOptions
) -> tuple[FunctionHandle, _cache.BuildStatistics]:
    """Returns the compiled programs of a term, from memory, the disk cache or a new build."""
    programs = program_options(options)
    key = _cache.term_key(str(term.jaxpr), term.signs, signature, programs)
    handle = compiled_terms.get(key)
    if handle is not None:
        compiled_terms.move_to_end(key)
        return handle, _cache.BuildStatistics(reused=1)
    statistics = _cache.BuildStatistics()
    start = time.perf_counter()
    lowered = _cache.load_lowered(key) if options.disk_cache else None
    if lowered is None:
        lowered, skipped = lower_programs(term, signature, programs)
        statistics = statistics.add(
            _cache.BuildStatistics(
                lower_seconds=time.perf_counter() - start, lowered=1, skipped=skipped
            )
        )
        if options.disk_cache:
            _cache.store_lowered(key, lowered)
    else:
        statistics = statistics.add(
            _cache.BuildStatistics(load_seconds=time.perf_counter() - start)
        )
    executables = []
    for program in lowered:
        if program is None:
            executables.append(None)
            continue
        executable, built = _cache.compile_program(program, options.disk_cache)
        statistics = statistics.add(built)
        executables.append(executable)
    handle = FunctionHandle(
        signature.num_inputs, signature.num_params, Programs._make(executables)
    )
    compiled_terms[key] = handle
    # Problems that still use an evicted handle keep it alive themselves.
    while len(compiled_terms) > COMPILED_TERMS_LIMIT:
        compiled_terms.popitem(last=False)
    return handle, statistics


def affine_function(
    analysis: Analysis, variable_avals: Sequence[jax.ShapeDtypeStruct], batched: bool
) -> Callable[..., tuple[Array, list[Array]]]:
    """Returns a function from parameter leaves to the constants and Jacobians of the affine part.

    The affine part is exact at any point, so it is linearized at zero. The constants have
    one entry per row and each Jacobian has one row per row and one column per entry of a
    variable leaf. With `batched`, the parameter leaves have a leading row axis.
    """
    affine = analysis.affine
    assert affine is not None
    zeros = [jnp.zeros(aval.shape, aval.dtype) for aval in variable_avals]

    def one(params: Sequence[Array]) -> tuple[Array, list[Array]]:
        def value(leaves: list[Array]) -> Array:
            return evaluate_term(
                affine, analysis.shape, analysis.constants, leaves, params
            ).ravel()

        constant = value(zeros)
        jacobians = jax.jacfwd(value)(zeros)
        rows = constant.shape[0]
        return constant, [jacobian.reshape(rows, -1) for jacobian in jacobians]

    if not batched:
        return lambda *params: one(params)

    def batch(*params: Array) -> tuple[Array, list[Array]]:
        constant, jacobians = jax.vmap(lambda *row: one(row))(*params)
        return constant[:, 0], [jacobian[:, 0, :] for jacobian in jacobians]

    return batch


def term_rows(
    compiled: CompiledTerm,
    block: Block,
    variables: VariableLayout,
    param_leaves: Sequence[np.ndarray],
) -> tuple[Int[np.ndarray, "rows n"], Float[np.ndarray, "rows m"]]:
    """Returns the input indices and the parameter values of a term in every row of a block.

    The rows of a batched block take their parameters from the block itself.
    """
    rows = block.rows
    leaves = param_leaves if block.batch is None else block.batch.params
    index_parts: list[np.ndarray] = []
    param_parts: list[np.ndarray] = []
    for ref, item in zip(compiled.term.inputs, compiled.signature.inputs, strict=True):
        match ref.origin:
            case Origin.VARIABLE:
                if block.batch is None:
                    leaf_indices = variables.indices(ref.index)
                    indices = np.broadcast_to(leaf_indices, (rows, leaf_indices.size))
                else:
                    indices = np.asarray(block.batch.inputs)
                if item.selection is not None:
                    indices = indices[:, list(item.selection)]
                index_parts.append(np.broadcast_to(indices, (rows, indices.shape[-1])))
            case Origin.CONSTANT:
                value = np.asarray(
                    block.analysis.constants[ref.index], np.float64
                ).ravel()
                param_parts.append(np.broadcast_to(value, (rows, value.shape[0])))
            case Origin.PARAMETER:
                value = np.asarray(leaves[ref.index], np.float64)
                value = (
                    value.reshape(rows, -1)
                    if block.batch is not None
                    else value.ravel()
                )
                param_parts.append(np.broadcast_to(value, (rows, value.shape[-1])))
            case _:
                raise AssertionError(f"Unknown input origin {ref.origin}.")
    if compiled.signature.num_entries > 1:
        param_parts.append(np.arange(rows, dtype=np.float64)[:, None])
    params = np.concatenate(param_parts, axis=1) if param_parts else np.zeros((rows, 0))
    if params.shape[1] < compiled.signature.num_params:
        params = np.pad(
            params, ((0, 0), (0, compiled.signature.num_params - params.shape[1]))
        )
    inputs = np.concatenate(index_parts, axis=1)
    return np.ascontiguousarray(inputs, np.intc), np.ascontiguousarray(
        params, np.float64
    )


def linear_rows(
    block: Block, variables: VariableLayout, param_leaves: Sequence[np.ndarray]
) -> tuple[
    Float[np.ndarray, " rows"], Int[np.ndarray, "rows k"], Float[np.ndarray, "rows k"]
]:
    """Returns the constants, variable indices and coefficients of the affine part of a block."""
    rows = block.rows
    if block.affine is None:
        return np.zeros(rows), np.zeros((rows, 0), np.intc), np.zeros((rows, 0))
    leaves = param_leaves if block.batch is None else block.batch.params
    constant, jacobians = block.affine(*leaves)
    index_parts: list[np.ndarray] = []
    coefficient_parts: list[np.ndarray] = []
    for leaf, jacobian in enumerate(jacobians):
        if block.batch is None:
            leaf_indices = variables.indices(leaf)
            indices = np.broadcast_to(leaf_indices, (rows, leaf_indices.size))
        else:
            indices = np.asarray(block.batch.inputs)
        index_parts.append(indices)
        coefficient_parts.append(np.asarray(jacobian, np.float64))
    return (
        np.asarray(constant, np.float64),
        np.concatenate(index_parts, axis=1).astype(np.intc),
        np.concatenate(coefficient_parts, axis=1),
    )
