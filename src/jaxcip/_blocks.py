"""Block terms: the programs of a term with many entries that each read a few variables.

A nonlinear term whose value is a vector becomes one SCIP expression per entry. An entry
only gets the variables it depends on as children, which ASDEX finds from the sparsity of the
Jacobian of the term's function. One set of programs evaluates all entries at once. Its
derivatives are compressed with a coloring of the Jacobian, so a program computes one
derivative per color and not one per variable. Every program returns its derivatives as
the entries of the Jacobian in the order of its pattern, entry after entry. SCIP's
expression `e` takes `data[pointers[e]:pointers[e + 1]]` as its derivatives with respect
to its children, which are the variables `columns[pointers[e]:pointers[e + 1]]` of the term.
"""

import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import jaxormick as jmk
import numpy as np
from jaxtyping import Array, Float, Int

from jaxcip._analysis import Term, sum_outputs
from jaxcip._options import RelaxationOptions
from jaxcip._programs import (
    OPTIONAL_PROGRAMS,
    LoweredProgram,
    LoweredPrograms,
    ProgramOptions,
    TermSignature,
    is_wanted,
    lower,
    term_arguments,
    term_signature,
)
from jaxcip._types import Program

BlockFunction = Callable[
    [Float[Array, " n"], Float[Array, " m"]], Float[Array, " rows"]
]
"""A vector function `f(x, params)` of two flat vectors."""

detection_lock = threading.Lock()
"""Serializes the detection of patterns, which compiles and runs numba code."""

DENSE_FRACTION = 0.5
"""Share of the variables that the entries of a dense cut read, on average."""

SMALL_BLOCK = 64
"""Largest number of entries times inputs of a cut that is kept with its sum or product.

Splitting a small cut gives SCIP several expressions and more callbacks for no gain in
sparsity."""

PATTERN_CACHE_LIMIT = 256
"""Number of patterns that are kept, so that the analysis and the lowering of a term share one."""

patterns: OrderedDict[tuple[str, tuple[float, ...], TermSignature], "BlockPattern"] = (
    OrderedDict()
)
"""Patterns of the cuts that were analysed, by jaxpr, signs and signature, oldest first."""

if TYPE_CHECKING:
    # asdex imports numba, which a loaded problem does not need.
    import asdex


class BlockPattern(NamedTuple):
    """Which variables each entry of a block term depends on."""

    pointers: Int[np.ndarray, " rows_plus_one"]
    """Start of each entry in `columns`, then the total number of dependencies."""
    columns: Int[np.ndarray, " nnz"]
    """For each dependency, the index of the variable among the inputs of the term. The
    variables of an entry are in increasing order."""

    @property
    def rows(self) -> Int[np.ndarray, " nnz"]:
        """The entry of each dependency."""
        return np.repeat(np.arange(len(self.pointers) - 1), np.diff(self.pointers))

    @property
    def nnz(self) -> int:
        """Number of dependencies."""
        return len(self.columns)


class LoweredTerm(NamedTuple):
    """The lowered programs of a term and, for a block, its pattern."""

    programs: LoweredPrograms
    pattern: BlockPattern | None


def block_function(term: Term, signature: TermSignature) -> BlockFunction:
    """Returns the vector function of all entries of a term."""

    def evaluate(
        x: Float[Array, " n"], params: Float[Array, " m"]
    ) -> Float[Array, " rows"]:
        arguments = term_arguments(signature, x, params)
        return sum_outputs(term, signature.shape, arguments).ravel()

    return evaluate


def pattern_of(term: Term, signature: TermSignature) -> BlockPattern:
    """Returns the pattern of a block term, which is detected once for each term."""
    key = (str(term.jaxpr), term.signs, signature)
    with detection_lock:
        found = patterns.get(key)
    if found is not None:
        return found
    found = detect_pattern(block_function(term, signature), signature)
    with detection_lock:
        patterns[key] = found
        while len(patterns) > PATTERN_CACHE_LIMIT:
            patterns.popitem(last=False)
    return found


def is_dense(term: Term, shape: tuple[int, ...]) -> bool:
    """Returns whether a cut with several entries is not worth separating from what follows.

    That is the case for a small cut, and for one whose entries read most of its variables.
    Separating the latter would give SCIP many expressions that each read everything,
    instead of one.
    """
    signature = term_signature(term, shape)
    size = signature.num_entries * signature.num_inputs
    if size <= SMALL_BLOCK:
        return True
    return pattern_of(term, signature).nnz > DENSE_FRACTION * size


def vector_avals(*sizes: int) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Returns the shapes of flat vectors of the given lengths."""
    return tuple(jax.ShapeDtypeStruct((size,), jnp.float64) for size in sizes)


def detect_pattern(function: BlockFunction, signature: TermSignature) -> BlockPattern:
    """Returns the pattern of the dependencies of the entries of a block function.

    An entry that depends on no variable gets the first one, so that every expression has a
    child. The derivative of such an entry is zero. If ASDEX cannot analyse the function,
    every entry depends on every variable.
    """
    import asdex

    rows, inputs = signature.num_entries, signature.num_inputs
    try:
        with detection_lock:
            sparsity = asdex.jacobian_sparsity(
                function,
                jnp.zeros(signature.num_inputs),
                jnp.zeros(signature.num_params),
            )
        found = np.asarray(sparsity.rows, dtype=np.int64)
        found_columns = np.asarray(sparsity.cols, dtype=np.int64)
    except NotImplementedError:
        # ASDEX has no rule for some primitives, such as the root of `lax.custom_root`.
        # Every entry may then read every variable.
        found = np.repeat(np.arange(rows, dtype=np.int64), inputs)
        found_columns = np.tile(np.arange(inputs, dtype=np.int64), rows)
    empty = np.setdiff1d(np.arange(rows), found)
    keys = np.unique(np.concatenate([found * inputs + found_columns, empty * inputs]))
    pointers = np.zeros(rows + 1, dtype=np.intc)
    np.cumsum(np.bincount(keys // inputs, minlength=rows), out=pointers[1:])
    return BlockPattern(pointers, np.ascontiguousarray(keys % inputs, dtype=np.intc))


def coloring_of(
    pattern: BlockPattern,
    num_inputs: int,
    avals: Sequence[jax.ShapeDtypeStruct],
    mode: Literal["fwd", "rev"] | None,
) -> "asdex.ColoredPattern":
    """Returns the coloring of the pattern for a function with the given arguments."""
    import asdex

    sparsity = asdex.SparsityPattern.from_coo(
        pattern.rows.astype(np.int32),
        pattern.columns.astype(np.int32),
        (len(pattern.pointers) - 1, num_inputs),
        input_avals=tuple(avals),
        argnums=0,
    )
    with warnings.catch_warnings():
        # A dense row costs what its dense derivative costs, so the warning about the
        # missing speedup is no use here.
        warnings.simplefilter("ignore", asdex.DenseColoringWarning)
        return asdex.jacobian_coloring_from_sparsity(sparsity, mode=mode)


def compressed(
    function: Callable[..., Array],
    pattern: BlockPattern,
    num_inputs: int,
    avals: Sequence[jax.ShapeDtypeStruct],
    mode: Literal["fwd", "rev"] | None,
) -> Callable[..., tuple[Array, Array]]:
    """Returns a function that gives the value of `function` and the entries of its Jacobian.

    The Jacobian is with respect to the first argument, and its entries come in the order of
    the pattern.
    """
    import asdex

    coloring = coloring_of(pattern, num_inputs, avals, mode)
    both = asdex.value_and_compressed_jacobian_from_coloring(function, coloring)

    def evaluate(*arguments: Array) -> tuple[Array, Array]:
        value, matrix = both(*arguments)
        return value, asdex.decompress_data(matrix, coloring)

    return evaluate


def block_programs(
    function: BlockFunction,
    pattern: BlockPattern,
    signature: TermSignature,
    options: RelaxationOptions,
) -> tuple[Callable[..., Any], ...]:
    """Returns the six programs of a block function in the order of `Program`.

    Each program takes the flat vectors of all entries and gives all entries at once, with
    the derivatives in the order of the pattern.
    """
    inputs, params = signature.num_inputs, signature.num_params
    relaxed = jmk.relax(
        function, alpha_bb=options.alpha_bb, tighten_bounds=options.tighten_bounds
    )

    def convex(x: Array, lower: Array, upper: Array, p: Array) -> Array:
        return relaxed(x, lower, upper, p).convex

    def concave(x: Array, lower: Array, upper: Array, p: Array) -> Array:
        return relaxed(x, lower, upper, p).concave

    # Forward mode keeps the programs of the relaxations small, as for scalar terms.
    box = vector_avals(inputs, inputs, inputs, params)
    value_and_jacobian = compressed(
        function, pattern, inputs, vector_avals(inputs, params), None
    )

    def entries(x: Array, p: Array) -> Array:
        return value_and_jacobian(x, p)[1]

    def hessian_vector(x: Array, direction: Array, p: Array) -> Array:
        return jax.jvp(lambda y: entries(y, p), (x,), (direction,))[1]

    return (
        value_and_jacobian,
        hessian_vector,
        compressed(convex, pattern, inputs, box, "fwd"),
        compressed(concave, pattern, inputs, box, "fwd"),
        jmk.interval(function, tighten_bounds=options.tighten_bounds),
        jmk.interval(entries, tighten_bounds=options.tighten_bounds),
    )


def block_program_shapes(
    program: Program, signature: TermSignature
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Returns the shapes of the arguments of a block program."""
    vector, params = vector_avals(signature.num_inputs, signature.num_params)
    match program:
        case Program.VALUE_AND_GRADIENT:
            return (vector, params)
        case Program.HESSIAN_VECTOR:
            return (vector, vector, params)
        case Program.CONVEX | Program.CONCAVE:
            return (vector, vector, vector, params)
        case Program.INTERVAL | Program.GRADIENT_INTERVAL:
            return (vector, vector, params)
        case _:
            raise AssertionError(f"Unknown program {program}.")


def lower_block_programs(
    term: Term, signature: TermSignature, options: ProgramOptions
) -> tuple[LoweredTerm, tuple[str, ...]]:
    """Lowers the programs of a block term and returns them with the reasons for left out ones.

    An optional program that cannot be lowered is left out. A required one raises.
    """
    function = block_function(term, signature)
    pattern = pattern_of(term, signature)
    lowered: list[LoweredProgram | None] = []
    skipped: list[str] = []
    programs = block_programs(function, pattern, signature, options.relaxation)
    for program, program_function in zip(Program, programs, strict=True):
        reason = is_wanted(program, options, term.jaxpr)
        if reason is None:
            try:
                lowered.append(
                    lower(program_function, block_program_shapes(program, signature))
                )
                continue
            except Exception as error:
                if program not in OPTIONAL_PROGRAMS:
                    raise
                reason = f"{type(error).__name__}: {error}"
        lowered.append(None)
        skipped.append(f"{program.name}: {reason}")
    return LoweredTerm(LoweredPrograms._make(lowered), pattern), tuple(skipped)
