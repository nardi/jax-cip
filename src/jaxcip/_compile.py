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
from typing import Any, NamedTuple, Never, ParamSpec, Self, TypeVar

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
from cyxla import Executable
from jaxtyping import Array, Float, Int, PyTree

from jaxcip import _cache
from jaxcip._analysis import Analysis, Origin, Term, analyse, evaluate_term
from jaxcip._block import BlockHandle
from jaxcip._blocks import (
    BlockPattern,
    LoweredTerm,
    detection_lock,
    is_dense,
    lower_block_programs,
)
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
from jaxcip._types import Domain, Programs, Sense

SubmittedParameters = ParamSpec("SubmittedParameters")
"""The parameters of a function that an executor runs."""

SubmittedResult = TypeVar("SubmittedResult")
"""The result of a function that an executor runs."""

COMPILED_TERMS_LIMIT = 64
"""Number of terms whose handles the process keeps after their problems are gone.

The executables of a term hold about 200 memory mappings of machine code, and Linux allows
65,530 mappings per process by default. A process that compiles many different problems
would reach that limit if it kept every term, so the cache keeps at most a fifth of it.
"""

compiled_terms: OrderedDict[str, FunctionHandle | BlockHandle] = OrderedDict()
"""Handles of the terms compiled in this process, by `_cache.term_key`, oldest first."""


def remember_handle(key: str, handle: FunctionHandle | BlockHandle) -> None:
    """Keeps a handle for later builds of terms with the same key, within the limit."""
    compiled_terms[key] = handle
    # Problems that still use an evicted handle keep it alive themselves.
    while len(compiled_terms) > COMPILED_TERMS_LIMIT:
        compiled_terms.popitem(last=False)


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
    fingerprint: tuple[cyxla.LeafInfo, ...]
    """The path and shape of every leaf, which a tree given to `load` has to match."""

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
    fingerprint: tuple[cyxla.LeafInfo, ...]
    """The path and shape of every leaf, which a tree given to `load` has to match."""

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
    """A cut of a function with its compiled programs."""

    term: Term
    signature: TermSignature
    handle: FunctionHandle | BlockHandle
    key: str = ""
    """The key of the term's programs, under which the handle is shared and stored."""

    @property
    def entries(self) -> int:
        """Number of entries of the cut, each of which is one expression in SCIP."""
        gather = self.term.gather
        return len(gather.leaves) if gather is not None else self.signature.num_entries


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
    pattern: BlockPattern | None = None
    """For a block term, which variables each entry reads."""


class InlineExecutor(Executor):
    """An executor that runs each call at once on the calling thread."""

    def submit(
        self,
        function: Callable[SubmittedParameters, SubmittedResult],
        /,
        *args: SubmittedParameters.args,
        **kwargs: SubmittedParameters.kwargs,
    ) -> Future[SubmittedResult]:
        """Runs a call and returns a future that is already done."""
        future: Future[SubmittedResult] = Future()
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
        self.handles: dict[str, FunctionHandle | BlockHandle] = {}
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
    ) -> tuple[FunctionHandle | BlockHandle, _cache.BuildStatistics]:
        """Waits for the programs of a requested term and returns its handle.

        Only the first call for a key returns the cost of the build. Later calls return a
        reused handle.
        """
        handle = self.handles.get(requested.key)
        if handle is not None:
            return handle, _cache.BuildStatistics(reused=1)
        built = self.builds[requested.key].result()
        signature = requested.signature
        programs = Programs._make(built.executables)
        created: FunctionHandle | BlockHandle
        if built.pattern is None:
            created = FunctionHandle(
                signature.num_inputs,
                signature.num_params,
                programs,
                signature.state_size,
                signature.enclosure_size,
            )
        else:
            created = BlockHandle(
                signature.num_inputs,
                signature.num_params,
                built.pattern.pointers,
                built.pattern.columns,
                programs,
            )
        self.handles[requested.key] = created
        remember_handle(requested.key, created)
        return created, built.statistics

    def build_programs(
        self, key: str, term: Term, signature: TermSignature
    ) -> BuiltPrograms:
        """Lowers a term, or loads it from the disk cache, and compiles its programs."""
        disk_cache = self.options.disk_cache
        statistics = _cache.BuildStatistics()
        start = time.perf_counter()
        lowered = _cache.load_lowered(key) if disk_cache else None
        if lowered is None:
            if signature.num_entries > 1:
                lowered, skipped = lower_block_programs(term, signature, self.programs)
            else:
                programs, skipped = lower_programs(term, signature, self.programs)
                lowered = LoweredTerm(programs, None)
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
            for program in lowered.programs
        ]
        executables: list[Executable | None] = []
        for compiled in compiles:
            if compiled is None:
                executables.append(None)
                continue
            executable, built = compiled.result()
            statistics = statistics.add(built)
            executables.append(executable)
        return BuiltPrograms(tuple(executables), statistics, lowered.pattern)


class Block(NamedTuple):
    """Rows of a problem that come from one function.

    A `Constraint` or the objective has one row per entry of its output. Each row is an
    affine function of the entries of the cuts of the function and of the variables. The
    terms are the cuts, and the cut entries come one cut after the other.
    """

    analysis: Analysis
    terms: tuple[CompiledTerm, ...]
    lower: Float[np.ndarray, " rows"]
    upper: Float[np.ndarray, " rows"]
    affine: "AffineEvaluator | None"
    """Maps parameter leaves to the constants and the Jacobian entries of the remainder.

    It is `None` if the output is the only cut, so that each row is one entry of the cut.
    """

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self.lower.shape[0]


class AffineValues(NamedTuple):
    """The constants and the Jacobian entries of the affine remainder of a block."""

    constants: Float[np.ndarray, " rows"]
    rows: Int[np.ndarray, " nnz"]
    """Row of each entry."""
    columns: Int[np.ndarray, " nnz"]
    """Entry of the cuts, then variable, that each entry is the derivative with respect to."""
    values: Float[np.ndarray, " nnz"]
    """Coefficient of each entry, which may be zero."""


class AffineEvaluator(NamedTuple):
    """The affine remainder of a block as a function of the parameters.

    Which entries of its Jacobian can be nonzero does not depend on the parameters, so the
    pattern is fixed and only the values are computed per solve. A program of cyxla computes
    them from the flat vector of parameters. A remainder that reads no parameter has its
    values stored instead.
    """

    rows: Int[np.ndarray, " nnz"]
    """Row of each entry."""
    columns: Int[np.ndarray, " nnz"]
    """Entry of the cuts, then variable, that each entry is the derivative with respect to."""
    program: Executable | None
    """Maps the flat parameters to the constants and the entries, or is `None` if they are fixed.

    The program has one output, the constants, if there are no entries."""
    constants: Float[np.ndarray, " rows"] | None
    values: Float[np.ndarray, " nnz"] | None

    def __call__(self, leaves: Sequence[np.ndarray]) -> AffineValues:
        """Returns the constants and the entries for the given parameter leaves."""
        if self.program is None:
            assert self.constants is not None and self.values is not None
            return AffineValues(self.constants, self.rows, self.columns, self.values)
        flat = np.concatenate([np.asarray(leaf, np.float64).ravel() for leaf in leaves])
        outputs = self.program.run([flat])
        values = outputs[1] if len(outputs) > 1 else np.zeros(0)
        return AffineValues(
            outputs[0].astype(np.float64), self.rows, self.columns, values
        )


class LinearRows(NamedTuple):
    """The remainder of a block as constants and two sparse matrices with one row per row."""

    constants: Float[np.ndarray, " rows"]
    pointers: Int[np.ndarray, " rows_plus_one"]
    """Start of each row in `indices` and `coefficients`, then their total number."""
    indices: Int[np.ndarray, " nnz"]
    """Variable index of each entry, as `np.intc`, in the order of the rows."""
    coefficients: Float[np.ndarray, " nnz"]
    """Coefficient of each entry. No entry is zero."""
    cut_pointers: Int[np.ndarray, " rows_plus_one"]
    """Start of each row in `cut_columns` and `cut_coefficients`, then their total number."""
    cut_columns: Int[np.ndarray, " cut_nnz"]
    """Entry of the cuts of each entry, in the order of the rows."""
    cut_coefficients: Float[np.ndarray, " cut_nnz"]
    """Coefficient of each cut entry. No entry is zero."""

    def row(self, row: int) -> tuple[Int[np.ndarray, " k"], Float[np.ndarray, " k"]]:
        """Returns the variable indices and coefficients of one row."""
        span = slice(self.pointers[row], self.pointers[row + 1])
        return self.indices[span], self.coefficients[span]

    def cut_row(
        self, row: int
    ) -> tuple[Int[np.ndarray, " k"], Float[np.ndarray, " k"]]:
        """Returns the cut entries and coefficients of one row."""
        span = slice(self.cut_pointers[row], self.cut_pointers[row + 1])
        return self.cut_columns[span], self.cut_coefficients[span]


class CompiledProblem(NamedTuple):
    """A problem whose functions are analysed and compiled, ready to solve."""

    problem: Problem | None
    """The problem that was compiled, or `None` for one that `load` returned."""
    options: CompileOptions
    sense: Sense
    variables: VariableLayout
    params: ParamsLayout
    objective: Block | None
    constraints: tuple[Block, ...]
    statistics: _cache.BuildStatistics

    def __reduce__(self) -> Never:
        """Refuses to pickle, since a compiled problem holds native handles.

        `jaxcip.dumps` and `jaxcip.loads` move a compiled problem between processes. The
        trees of the variables and parameters are not part of the file, so unpickling could
        not return a problem that is ready to use.
        """
        raise TypeError(
            "A compiled problem cannot be pickled. Use jaxcip.dumps to get bytes, and "
            "jaxcip.loads with the variables and parameters to read them."
        )


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
    cyxla.initialize(platform="cpu")
    start = time.perf_counter()
    variables = variable_layout(problem.variables)
    leaves, structure = jax.tree.flatten(params)
    params_layout = ParamsLayout(
        structure,
        tuple(as_aval(leaf) for leaf in leaves),
        cyxla.tree_fingerprint(params, lambda leaf: as_aval(leaf).shape),
    )
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
        sense=problem.sense,
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


def leaf_shape(leaf: Any) -> tuple[int, ...]:
    """Returns the shape of a leaf of the variables, which is a `Variable` or an array."""
    if is_variable(leaf):
        return np.broadcast_shapes(
            leaf.shape, np.shape(leaf.lower), np.shape(leaf.upper)
        )
    return tuple(np.shape(leaf))


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
        shape = leaf_shape(leaf)
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
        fingerprint=cyxla.tree_fingerprint(variables, leaf_shape, is_leaf=is_variable),
    )


class RequestedBlock(NamedTuple):
    """A block whose terms are being built, which `finish_block` completes."""

    block: Block
    """The block without its compiled terms and its affine evaluator."""
    terms: tuple[RequestedTerm, ...]
    affine: "Future[tuple[AffineEvaluator, _cache.BuildStatistics]] | None" = None
    """The build of the affine evaluator, if the block has an affine remainder."""


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
    analysis = analyse(flat, variable_avals, params.avals, options.split_sums, is_dense)
    terms = request_terms(analysis, builder)
    affine = None
    if analysis.affine is not None:
        affine = builder.lowering.submit(
            build_affine, analysis, variables, params, options.disk_cache
        )
    return RequestedBlock(
        Block(
            analysis=analysis,
            terms=(),
            lower=row_bounds(lower, analysis.shape),
            upper=row_bounds(upper, analysis.shape),
            affine=None,
        ),
        terms,
        affine,
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
        terms.append(CompiledTerm(term.term, term.signature, handle, term.key))
    block = requested.block._replace(terms=tuple(terms))
    if requested.affine is not None:
        affine, built = requested.affine.result()
        statistics = statistics.add(built)
        block = block._replace(affine=affine)
    return block, statistics


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
        builder.request(term, term_signature(term, () if term.gather else shape))
        for term, shape in zip(analysis.nonlinear, analysis.cut_shapes, strict=True)
    )


def build_affine(
    analysis: Analysis,
    variables: VariableLayout,
    params: ParamsLayout,
    disk_cache: bool,
) -> tuple[AffineEvaluator, _cache.BuildStatistics]:
    """Returns the evaluator of the affine remainder of a block with the cost of building it.

    The remainder is exact at any point, so it is linearized at zero, with respect to the
    entries of the cuts and then the variables. ASDEX finds which entries of the Jacobian can
    be nonzero and colors its columns once, so a solve only evaluates one derivative per
    color. The values of the entries depend on the parameters, so a program computes them,
    unless the remainder reads none. The pattern and the program are kept in the disk cache,
    which lets a warm build skip the detection and the lowering.
    """
    remainder = analysis.affine
    assert remainder is not None
    start = time.perf_counter()
    depends = any(ref.origin is Origin.PARAMETER for ref in remainder.inputs)
    key = _cache.affine_key(
        str(remainder.jaxpr),
        repr((remainder.signs, remainder.inputs, remainder.selections)),
        repr((remainder.linearity, remainder.row_axes, analysis.shape)),
        repr((analysis.cut_shapes, variables.shapes, variables.offsets)),
        repr([(aval.shape, str(aval.dtype)) for aval in params.avals]),
        *(
            np.asarray(constant, np.float64).tobytes()
            for constant in analysis.constants
        ),
    )
    if disk_cache:
        entries = _cache.load_affine(key)
        lowered = cyxla.load_lowered(key) if depends else ()
        if entries is not None and lowered is not None:
            program = None
            statistics = _cache.BuildStatistics(
                load_seconds=time.perf_counter() - start
            )
            if depends:
                assert lowered[0] is not None
                program, compiled = cyxla.compile_program(lowered[0])
                statistics = statistics.add(seconds_of(compiled))
            return affine_from(entries, program), statistics
    value, jacobian, zeros = affine_functions(analysis, variables, params)
    sample = [jnp.zeros(aval.shape, aval.dtype) for aval in params.avals]
    entries = jacobian(zeros, sample)
    pattern = _cache.AffineEntries(
        np.asarray(entries.indices)[:, 0], np.asarray(entries.indices)[:, 1]
    )
    statistics = _cache.BuildStatistics(lower_seconds=time.perf_counter() - start)
    if not depends:
        pattern = pattern._replace(
            constants=np.asarray(value(zeros, sample), np.float64),
            values=np.asarray(entries.data, np.float64),
        )
        if disk_cache:
            _cache.store_affine(key, pattern)
        return affine_from(pattern, None), statistics

    offsets = np.cumsum([0, *(math.prod(aval.shape) for aval in params.avals)])

    def program_function(flat: Array) -> tuple[Array, ...]:
        leaves = [
            flat[offsets[index] : offsets[index + 1]]
            .reshape(aval.shape)
            .astype(aval.dtype)
            for index, aval in enumerate(params.avals)
        ]
        constants = value(zeros, leaves)
        if pattern.rows.size == 0:
            return (constants,)
        return (constants, jacobian(zeros, leaves).data)

    program = cyxla.lower(
        program_function,
        [jax.ShapeDtypeStruct((int(offsets[-1]),), jnp.float64)],
        platform="cpu",
    )
    statistics = statistics._replace(lower_seconds=time.perf_counter() - start)
    if disk_cache:
        cyxla.store_lowered(key, [program])
        _cache.store_affine(key, pattern)
    executable, compiled = cyxla.compile_program(program, disk_cache)
    statistics = statistics.add(seconds_of(compiled))
    return affine_from(pattern, executable), statistics


def seconds_of(compiled: cyxla.CompileStatistics) -> _cache.BuildStatistics:
    """Returns the time of compiling or loading an executable, without counting the program.

    The counts of a build are those of the programs of terms, so the affine programs of the
    blocks add their time but not to the counts.
    """
    return _cache.BuildStatistics(
        compile_seconds=compiled.compile_seconds, load_seconds=compiled.load_seconds
    )


def affine_from(
    entries: _cache.AffineEntries, program: Executable | None
) -> AffineEvaluator:
    """Returns the evaluator of stored entries and, unless they are fixed, a program."""
    return AffineEvaluator(
        entries.rows, entries.columns, program, entries.constants, entries.values
    )


def affine_functions(
    analysis: Analysis, variables: VariableLayout, params: ParamsLayout
) -> tuple[Callable[..., Array], Callable[..., Any], Array]:
    """Returns the value of the remainder, its compressed Jacobian and the point of zeros.

    The functions take the point and the parameter leaves. The Jacobian uses the coloring that
    ASDEX found for the pattern of the remainder, and returns a sparse matrix.
    """
    remainder = analysis.affine
    assert remainder is not None
    sizes = [math.prod(shape) for shape in analysis.cut_shapes]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    zeros = jnp.zeros(int(offsets[-1]) + variables.size)
    sample = [jnp.zeros(aval.shape, aval.dtype) for aval in params.avals]

    def value(z: Array, leaves: Sequence[Array]) -> Array:
        cuts = [
            z[offsets[index] : offsets[index + 1]].reshape(shape)
            for index, shape in enumerate(analysis.cut_shapes)
        ]
        x = z[offsets[-1] :]
        parts = [
            x[offset : offset + math.prod(shape)].reshape(shape)
            for offset, shape in zip(variables.offsets, variables.shapes, strict=True)
        ]
        return evaluate_term(
            remainder, analysis.shape, analysis.constants, parts, leaves, cuts
        ).ravel()

    import asdex

    with detection_lock, warnings.catch_warnings():
        # A dense remainder, such as a matrix product, is correct and costs what the
        # dense derivative would, so the warning about the missing speedup is no use here.
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        coloring = asdex.jacobian_coloring(value, zeros, sample, argnums=0)
    return value, asdex.jacobian_from_coloring(value, coloring), zeros


def input_parts(
    compiled: CompiledTerm,
    block: Block,
    variables: VariableLayout,
    param_leaves: Sequence[np.ndarray],
    rows: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns the variable indices and the parameter values of a term, with a row axis.

    A term of `rows` reads its own variables in every row and takes the row's slice of the
    inputs that have a row axis. Other terms read the same inputs in every row.
    """
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
    return index_parts, param_parts


def term_rows(
    compiled: CompiledTerm,
    block: Block,
    variables: VariableLayout,
    param_leaves: Sequence[np.ndarray],
) -> tuple[Int[np.ndarray, "rows n"], Float[np.ndarray, "rows m"]]:
    """Returns the input indices and the parameter values of each entry of a cut."""
    rows = compiled.entries
    index_parts, param_parts = input_parts(
        compiled, block, variables, param_leaves, rows
    )
    params = np.concatenate(param_parts, axis=1) if param_parts else np.zeros((rows, 0))
    if params.shape[1] < compiled.signature.num_params:
        params = np.pad(
            params, ((0, 0), (0, compiled.signature.num_params - params.shape[1]))
        )
    inputs = np.concatenate(index_parts, axis=1)
    return np.ascontiguousarray(inputs, np.intc), np.ascontiguousarray(
        params, np.float64
    )


def block_inputs(
    compiled: CompiledTerm,
    block: Block,
    variables: VariableLayout,
    param_leaves: Sequence[np.ndarray],
) -> tuple[Int[np.ndarray, " n"], Float[np.ndarray, " m"]]:
    """Returns the variables and the parameter values of a block term, which all entries share."""
    index_parts, param_parts = input_parts(compiled, block, variables, param_leaves, 1)
    params = np.concatenate(param_parts, axis=1)[0] if param_parts else np.zeros(0)
    params = np.pad(params, (0, compiled.signature.num_params - params.shape[0]))
    inputs = np.concatenate(index_parts, axis=1)[0]
    return np.ascontiguousarray(inputs, np.intc), np.ascontiguousarray(
        params, np.float64
    )


def linear_rows(
    block: Block, variables: VariableLayout, param_leaves: Sequence[np.ndarray]
) -> LinearRows:
    """Returns the constants and the nonzero coefficients of the remainder of a block.

    The coefficients of the variables and those of the cut entries are separate matrices.
    """
    rows = block.rows
    empty_pointers = np.zeros(rows + 1, np.intp)
    if block.affine is None:
        # The only cut is the output, so each row is one entry of it.
        return LinearRows(
            np.zeros(rows),
            empty_pointers,
            np.zeros(0, np.intc),
            np.zeros(0),
            np.arange(rows + 1, dtype=np.intp),
            np.arange(rows, dtype=np.intc),
            np.ones(rows),
        )
    entries = block.affine(param_leaves)
    num_cuts = sum(math.prod(shape) for shape in block.analysis.cut_shapes)
    used = entries.values != 0.0
    split = entries.columns < num_cuts

    def matrix(
        selected: np.ndarray, shift: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entry_rows = entries.rows[selected]
        order = np.argsort(entry_rows, kind="stable")
        pointers = np.zeros(rows + 1, np.intp)
        np.cumsum(np.bincount(entry_rows, minlength=rows), out=pointers[1:])
        return (
            pointers,
            (entries.columns[selected] - shift)[order].astype(np.intc),
            entries.values[selected][order],
        )

    pointers, indices, coefficients = matrix(used & ~split, num_cuts)
    cut_pointers, cut_columns, cut_coefficients = matrix(used & split, 0)
    return LinearRows(
        constants=entries.constants,
        pointers=pointers,
        indices=indices,
        coefficients=coefficients,
        cut_pointers=cut_pointers,
        cut_columns=cut_columns,
        cut_coefficients=cut_coefficients,
    )
