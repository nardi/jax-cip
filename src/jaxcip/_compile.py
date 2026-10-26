"""Compiling a problem into blocks of rows with compiled terms.

Compiling traces every function of a problem once, splits it into an affine part and
nonlinear terms, and builds the programs of each term. Nothing here depends on parameter
values, so one compiled problem solves for many of them.
"""

import math
import os
import time
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any, NamedTuple, Self

import asdex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, PyTree

from jaxcip import _cache
from jaxcip._analysis import Analysis, Origin, Term, analyse, evaluate_term
from jaxcip._handler import FunctionHandle
from jaxcip._options import CompileOptions
from jaxcip._problem_types import (
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
from jaxcip._runtime import Executable
from jaxcip._types import Domain, Programs

COMPILED_TERMS_LIMIT = 64
"""Number of terms whose handles the process keeps after their problems are gone.

The executables of a term hold about 200 memory mappings of machine code, and Linux allows
65,530 mappings per process by default. A process that compiles many different problems
would reach that limit if it kept every term, so the cache keeps at most a fifth of it.
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


class RequestedTerm(NamedTuple):
    """A nonlinear term whose programs a `HandleBuilder` is building."""

    term: Term
    signature: TermSignature
    key: str
    """The key of the term's programs, from which the builder returns the handle."""


class BuiltPrograms(NamedTuple):
    """The compiled programs of a term, in the order of `Program`, and the cost of building them."""

    executables: tuple[Executable | None, ...]
    statistics: _cache.BuildStatistics


class InlineExecutor(Executor):
    """An executor that runs each call at once on the calling thread."""

    def submit[**Parameters, Result](
        self,
        function: Callable[Parameters, Result],
        /,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> Future[Result]:
        """Runs a call and returns a future that is already done."""
        future: Future[Result] = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class HandleBuilder:
    """Builds the handles of terms, lowering and compiling their programs on threads.

    `request` starts building a term and `resolve` returns its handle. Terms with the same
    key share one build. Lowering a term runs on one pool and compiles each of its programs
    on another, so a term waits for its compiles without blocking other compiles. Use the
    builder as a context manager, so that the threads stop when building fails.
    """

    def __init__(self, options: CompileOptions) -> None:
        self.options = options
        self.programs = program_options(options)
        threads = options.threads or os.cpu_count() or 1
        self.lowering: Executor
        self.compiling: Executor
        if threads > 1:
            self.lowering = ThreadPoolExecutor(threads, "jaxcip-lower")
            self.compiling = ThreadPoolExecutor(threads, "jaxcip-compile")
        else:
            self.lowering = self.compiling = InlineExecutor()
        self.builds: dict[str, Future[BuiltPrograms]] = {}
        """Builds that were started, by term key."""
        self.handles: dict[str, FunctionHandle] = {}
        """Handles that are ready, by term key."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.lowering.shutdown(wait=True, cancel_futures=True)
        self.compiling.shutdown(wait=True, cancel_futures=True)

    def request(self, term: Term, signature: TermSignature) -> RequestedTerm:
        """Starts building the programs of a term unless they are ready or being built."""
        if self.options.relaxation.tighten_bounds:
            # Tightening the bounds of intermediate values cannot reuse enclosures of roots.
            signature = signature._replace(enclosure_size=0)
        key = _cache.term_key(str(term.jaxpr), term.signs, signature, self.programs)
        if key not in self.handles and key not in self.builds:
            handle = compiled_terms.get(key)
            if handle is not None:
                compiled_terms.move_to_end(key)
                self.handles[key] = handle
            else:
                self.builds[key] = self.lowering.submit(
                    self.build_programs, key, term, signature
                )
        return RequestedTerm(term, signature, key)

    def resolve(
        self, requested: RequestedTerm
    ) -> tuple[FunctionHandle, _cache.BuildStatistics]:
        """Waits for the programs of a requested term and returns its handle.

        Only the first call for a key returns the cost of the build. Later calls return a
        reused handle.
        """
        handle = self.handles.get(requested.key)
        if handle is not None:
            return handle, _cache.BuildStatistics(reused=1)
        built = self.builds[requested.key].result()
        signature = requested.signature
        handle = FunctionHandle(
            signature.num_inputs,
            signature.num_params,
            Programs._make(built.executables),
            signature.state_size,
            signature.enclosure_size,
        )
        self.handles[requested.key] = handle
        compiled_terms[requested.key] = handle
        # Problems that still use an evicted handle keep it alive themselves.
        while len(compiled_terms) > COMPILED_TERMS_LIMIT:
            compiled_terms.popitem(last=False)
        return handle, built.statistics

    def build_programs(
        self, key: str, term: Term, signature: TermSignature
    ) -> BuiltPrograms:
        """Lowers a term, or loads it from the disk cache, and compiles its programs."""
        disk_cache = self.options.disk_cache
        statistics = _cache.BuildStatistics()
        start = time.perf_counter()
        lowered = _cache.load_lowered(key) if disk_cache else None
        if lowered is None:
            lowered, skipped = lower_programs(term, signature, self.programs)
            statistics = statistics.add(
                _cache.BuildStatistics(
                    lower_seconds=time.perf_counter() - start,
                    lowered=1,
                    skipped=skipped,
                )
            )
            if disk_cache:
                _cache.store_lowered(key, lowered)
        else:
            statistics = statistics.add(
                _cache.BuildStatistics(load_seconds=time.perf_counter() - start)
            )
        compiles = [
            None
            if program is None
            else self.compiling.submit(_cache.compile_program, program, disk_cache)
            for program in lowered
        ]
        executables: list[Executable | None] = []
        for compiled in compiles:
            if compiled is None:
                executables.append(None)
                continue
            executable, built = compiled.result()
            statistics = statistics.add(built)
            executables.append(executable)
        return BuiltPrograms(tuple(executables), statistics)


class Block(NamedTuple):
    """Rows of a problem that come from one function.

    A `Constraint` or the objective has one row per entry of its output. The terms of a row
    read the variables of the function, or for terms of `rows` the variables of their own
    row.
    """

    analysis: Analysis
    terms: tuple[CompiledTerm, ...]
    lower: Float[np.ndarray, " rows"]
    upper: Float[np.ndarray, " rows"]
    affine: "AffineFunction | None"
    """Maps parameter leaves to the constants and the Jacobian entries of the affine part."""

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.lower.shape[0]


class AffineValues(NamedTuple):
    """The constants and the Jacobian entries of the affine part of a block."""

    constants: Float[np.ndarray, " rows"]
    rows: Int[np.ndarray, " nnz"]
    """Row of each entry."""
    columns: Int[np.ndarray, " nnz"]
    """Variable index of each entry."""
    values: Float[np.ndarray, " nnz"]
    """Coefficient of each entry, which may be zero."""


AffineFunction = Callable[[Sequence[np.ndarray]], AffineValues]
"""Maps parameter leaves to `AffineValues`."""


class LinearRows(NamedTuple):
    """The affine part of a block as constants and a sparse matrix with one row per row."""

    constants: Float[np.ndarray, " rows"]
    pointers: Int[np.ndarray, " rows_plus_one"]
    """Start of each row in `indices` and `coefficients`, then their total number."""
    indices: Int[np.ndarray, " nnz"]
    """Variable index of each entry, as `np.intc`, in the order of the rows."""
    coefficients: Float[np.ndarray, " nnz"]
    """Coefficient of each entry. No entry is zero."""

    def row(self, row: int) -> tuple[Int[np.ndarray, " k"], Float[np.ndarray, " k"]]:
        """Returns the variable indices and coefficients of one row."""
        span = slice(self.pointers[row], self.pointers[row + 1])
        return self.indices[span], self.coefficients[span]


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
    start = time.perf_counter()
    variables = variable_layout(problem.variables)
    leaves, structure = jax.tree.flatten(params)
    params_layout = ParamsLayout(structure, tuple(as_aval(leaf) for leaf in leaves))
    # All blocks are analysed and their programs requested before any term is waited for,
    # so the builder lowers and compiles terms of different blocks at the same time.
    with HandleBuilder(options) as builder:
        objective_request = None
        if problem.objective is not None:
            objective_request = function_block(
                problem.objective,
                -math.inf,
                math.inf,
                variables,
                params_layout,
                options,
                builder,
            )
            if objective_request.block.analysis.shape != ():
                raise ValueError(
                    "The objective must return a scalar, "
                    f"not shape {objective_request.block.analysis.shape}."
                )
        requests = []
        for constraint in problem.constraints:
            match constraint:
                case Constraint(function, lower, upper):
                    requests.append(
                        function_block(
                            function,
                            lower,
                            upper,
                            variables,
                            params_layout,
                            options,
                            builder,
                        )
                    )
                case _:
                    raise TypeError(
                        f"A constraint must be a Constraint, not {type(constraint).__name__}."
                    )
        statistics = _cache.BuildStatistics()
        objective = None
        if objective_request is not None:
            objective, built = finish_block(objective_request, builder)
            statistics = statistics.add(built)
        blocks = []
        for request in requests:
            block, built = finish_block(request, builder)
            statistics = statistics.add(built)
            blocks.append(block)
    return CompiledProblem(
        problem=problem,
        options=options,
        variables=variables,
        params=params_layout,
        objective=objective,
        constraints=tuple(blocks),
        statistics=statistics._replace(wall_seconds=time.perf_counter() - start),
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


class RequestedBlock(NamedTuple):
    """A block whose terms are being built, which `finish_block` completes."""

    block: Block
    """The block without its compiled terms."""
    terms: tuple[RequestedTerm, ...]


def function_block(
    function: ParametricFunction,
    lower: Any,
    upper: Any,
    variables: VariableLayout,
    params: ParamsLayout,
    options: CompileOptions,
    builder: HandleBuilder,
) -> RequestedBlock:
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
    terms = request_terms(analysis, builder)
    affine = None
    if analysis.affine is not None:
        affine = sparse_affine_function(analysis, variables, params)
    return RequestedBlock(
        Block(
            analysis=analysis,
            terms=(),
            lower=row_bounds(lower, analysis.shape),
            upper=row_bounds(upper, analysis.shape),
            affine=affine,
        ),
        terms,
    )


def finish_block(
    requested: RequestedBlock, builder: HandleBuilder
) -> tuple[Block, _cache.BuildStatistics]:
    """Waits for the programs of a block's terms and returns the block with its compiled terms."""
    statistics = _cache.BuildStatistics()
    terms = []
    for term in requested.terms:
        handle, built = builder.resolve(term)
        statistics = statistics.add(built)
        terms.append(CompiledTerm(term.term, term.signature, handle))
    return requested.block._replace(terms=tuple(terms)), statistics


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


def request_terms(
    analysis: Analysis, builder: HandleBuilder
) -> tuple[RequestedTerm, ...]:
    """Starts building the programs of the nonlinear terms of an analysis."""
    # A term of `rows` is the function of one row, so its value is a scalar.
    return tuple(
        builder.request(
            term, term_signature(term, () if term.gather else analysis.shape)
        )
        for term in analysis.nonlinear
    )


def sparse_affine_function(
    analysis: Analysis, variables: VariableLayout, params: ParamsLayout
) -> AffineFunction:
    """Returns a function from parameter leaves to the constants and Jacobian of the affine part.

    The affine part is exact at any point, so it is linearized at zero. ASDEX finds which
    entries of the Jacobian can be nonzero and colors its columns once, so a solve only
    evaluates one derivative per color. The values of the entries depend on the parameters.
    """
    affine = analysis.affine
    assert affine is not None
    zeros = jnp.zeros(variables.size)
    sample = [jnp.zeros(aval.shape, aval.dtype) for aval in params.avals]

    def value(x: Array, leaves: Sequence[Array]) -> Array:
        parts = [
            x[offset : offset + math.prod(shape)].reshape(shape)
            for offset, shape in zip(variables.offsets, variables.shapes, strict=True)
        ]
        return evaluate_term(
            affine, analysis.shape, analysis.constants, parts, leaves
        ).ravel()

    with warnings.catch_warnings():
        # A dense affine part, such as a matrix product, is correct and costs what the
        # dense derivative would, so the warning about the missing speedup is no use here.
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        coloring = asdex.jacobian_coloring(value, zeros, sample, argnums=0)
    jacobian = jax.jit(asdex.jacobian_from_coloring(value, coloring))
    constant = jax.jit(lambda leaves: value(zeros, leaves))

    def evaluate(leaves: Sequence[np.ndarray]) -> AffineValues:
        arguments = [jnp.asarray(leaf) for leaf in leaves]
        entries = jacobian(zeros, arguments)
        indices = np.asarray(entries.indices)
        return AffineValues(
            constants=np.asarray(constant(arguments), np.float64),
            rows=indices[:, 0],
            columns=indices[:, 1],
            values=np.asarray(entries.data, np.float64),
        )

    return evaluate


def term_rows(
    compiled: CompiledTerm,
    block: Block,
    variables: VariableLayout,
    param_leaves: Sequence[np.ndarray],
) -> tuple[Int[np.ndarray, "rows n"], Float[np.ndarray, "rows m"]]:
    """Returns the input indices and the parameter values of a term in every row of a block.

    A term of `rows` reads its own variables in every row and takes the row's slice of the
    inputs that have a row axis. Other terms read the same inputs in every row.
    """
    rows = block.rows
    index_parts: list[np.ndarray] = []
    param_parts: list[np.ndarray] = []
    term = compiled.term
    for position, (ref, selection) in enumerate(
        zip(term.inputs, term.selections, strict=True)
    ):
        shared = 1 if not term.row_axes or not term.row_axes[position] else rows
        match ref.origin:
            case Origin.VARIABLE:
                part = variables.indices(ref.index)[None]
            case Origin.GATHER:
                assert term.gather is not None
                part = (
                    np.asarray(variables.offsets)[term.gather.leaves]
                    + term.gather.entries
                )
            case Origin.CONSTANT:
                constant = np.asarray(block.analysis.constants[ref.index], np.float64)
                part = constant.reshape(shared, -1)
            case Origin.PARAMETER:
                part = np.asarray(param_leaves[ref.index], np.float64).reshape(
                    shared, -1
                )
            case _:
                raise AssertionError(f"Unknown input origin {ref.origin}.")
        if selection is not None:
            part = part[:, list(selection)]
        part = np.broadcast_to(part, (rows, part.shape[1]))
        parts = (
            index_parts
            if ref.origin in (Origin.VARIABLE, Origin.GATHER)
            else param_parts
        )
        parts.append(part)
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
) -> LinearRows:
    """Returns the constants and the nonzero coefficients of the affine part of a block."""
    rows = block.rows
    if block.affine is None:
        return LinearRows(
            np.zeros(rows),
            np.zeros(rows + 1, np.intp),
            np.zeros(0, np.intc),
            np.zeros(0),
        )
    entries = block.affine(param_leaves)
    used = entries.values != 0.0
    entry_rows, columns = entries.rows[used], entries.columns[used]
    order = np.argsort(entry_rows, kind="stable")
    pointers = np.zeros(rows + 1, np.intp)
    np.cumsum(np.bincount(entry_rows, minlength=rows), out=pointers[1:])
    return LinearRows(
        constants=entries.constants,
        pointers=pointers,
        indices=columns[order].astype(np.intc),
        coefficients=entries.values[used][order],
    )
