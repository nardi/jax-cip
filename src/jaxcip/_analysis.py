"""Analysis of the jaxpr of a function into an affine part and nonlinear terms.

SCIP handles affine functions much better than opaque ones, since it propagates and
separates linear rows exactly. The analysis splits the output of a function at additions
and subtractions into terms, classifies each term as affine or nonlinear in the variables,
and keeps for every term only the inputs it depends on.
"""

import enum
import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import partial_eval
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var, jaxpr_as_fun
from jaxtyping import Array, Float, Int

from jaxcip._problem_types import RowFunction
from jaxcip._rows import description_of, forget_since, last_number

Atom = Var | Literal


class Linearity(enum.IntEnum):
    """How a value depends on the variables, ordered from least to most general."""

    CONSTANT = 0
    """Depends only on parameters and constants."""
    AFFINE = 1
    """Is an affine function of the variables."""
    NONLINEAR = 2
    """May depend on the variables in any way."""


class Origin(enum.Enum):
    """Kinds of inputs of an analysed jaxpr."""

    CONSTANT = 0
    """An array that the function closes over."""
    VARIABLE = 1
    """A leaf of the variables."""
    PARAMETER = 2
    """A leaf of the parameters."""
    GATHER = 3
    """The variables that the rows of a `rows` term read, as one vector per row."""


class InputRef(NamedTuple):
    """One input of a term, as a position among the inputs of its kind."""

    origin: Origin
    index: int


class RowSource(NamedTuple):
    """The variables that each row of a `rows` term reads."""

    leaves: Int[np.ndarray, "rows k"]
    """The leaf of the variables that each input of each row is an entry of."""
    entries: Int[np.ndarray, "rows k"]
    """The flat position of each input of each row in its leaf."""


class Term(NamedTuple):
    """A sum `Σ sign · output` of values of a function, as a jaxpr of its own inputs."""

    jaxpr: Jaxpr
    """The jaxpr computing the summed values from `inputs`, without constants.

    An input with a selection enters as a vector of only the selected entries, so terms
    that differ only in the entries they read have the same jaxpr.
    """
    signs: tuple[float, ...]
    """Sign of each output of `jaxpr` in the sum."""
    inputs: tuple[InputRef, ...]
    """The inputs of `jaxpr` in order."""
    selections: tuple[tuple[int, ...] | None, ...]
    """For each input, the flat indices of the entries `jaxpr` reads, or `None` for all."""
    linearity: Linearity
    gather: RowSource | None = None
    """For a term of `rows`, the variables of each row, in the single input of origin `GATHER`.

    The term is the scalar function of one row, and the block has one row for each row of the
    source. Other terms have the entries of their output as rows.
    """
    row_axes: tuple[bool, ...] = ()
    """For a term of `rows`, whether each input has a leading axis with one entry per row."""


class Analysis(NamedTuple):
    """The affine part and the nonlinear terms of a function."""

    shape: tuple[int, ...]
    """Shape of the output of the function."""
    constants: tuple[np.ndarray, ...]
    """Arrays that the function closes over."""
    affine: Term | None
    """Sum of the affine and constant terms, if there are any."""
    nonlinear: tuple[Term, ...]
    """Nonlinear terms, each a single value unless sums are not split."""


SPLIT_PRIMITIVES = frozenset({"add", "add_any", "sub", "neg"})
"""Primitives whose outputs are sums or differences of their operands."""

LINEAR_PRIMITIVES = frozenset(
    {
        "add",
        "add_any",
        "sub",
        "neg",
        "reshape",
        "broadcast_in_dim",
        "squeeze",
        "expand_dims",
        "transpose",
        "slice",
        "concatenate",
        "reduce_sum",
        "copy",
        "copy_p",
        "rev",
        "pad",
        "cumsum",
        "split",
        "stack",
    }
)
"""Primitives that are linear in all their operands together."""

INDEXED_PRIMITIVES = {
    "dynamic_slice": 1,
    "dynamic_update_slice": 2,
    "gather": 1,
    "scatter-add": 1,
}
"""Primitives linear in their data operands, with the position of the first index operand.

The indices must be constant. For `scatter-add` the updates come after the indices.
"""

BILINEAR_PRIMITIVES = frozenset({"mul", "dot_general"})
"""Primitives that are linear in each operand when the other is constant."""

CALL_PRIMITIVES = frozenset(
    {"jit", "pjit", "closed_call", "core_call", "remat2", "checkpoint"}
)
"""Primitives that apply their body once to their operands.

Loops such as `scan` also hold a body, but run it repeatedly, so a value that is constant on
the first pass may not be on later ones. They are treated as nonlinear.
"""

SUBJAXPR_PARAMETERS = ("jaxpr", "call_jaxpr")
"""Parameters that hold the body of call primitives."""


def analyse(
    function: Callable[..., Array],
    variables: Sequence[jax.ShapeDtypeStruct],
    parameters: Sequence[jax.ShapeDtypeStruct],
    split_sums: bool,
) -> Analysis:
    """Returns the affine part and the nonlinear terms of `function(*variables, *parameters)`.

    With `split_sums`, every nonlinear term of the outer sum is separate. Without it, the
    nonlinear terms form one term.
    """
    first_number = last_number[0] + 1
    try:
        return analyse_function(function, variables, parameters, split_sums)
    finally:
        # The descriptions of marked rows are only needed while this analysis reads them.
        forget_since(first_number)


def analyse_function(
    function: Callable[..., Array],
    variables: Sequence[jax.ShapeDtypeStruct],
    parameters: Sequence[jax.ShapeDtypeStruct],
    split_sums: bool,
) -> Analysis:
    """Returns the analysis of a function, which `analyse` wraps to clean up after it."""
    closed = jax.make_jaxpr(function)(*variables, *parameters)
    if len(closed.jaxpr.outvars) != 1:
        raise ValueError("A function of a problem must return one array.")
    jaxpr = partial_eval.convert_constvars_jaxpr(closed.jaxpr)
    origins = (
        [InputRef(Origin.CONSTANT, index) for index in range(len(closed.consts))]
        + [InputRef(Origin.VARIABLE, index) for index in range(len(variables))]
        + [InputRef(Origin.PARAMETER, index) for index in range(len(parameters))]
    )
    output = jaxpr.outvars[0]
    producers = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    affine: list[tuple[Atom, float]] = []
    nonlinear: list[tuple[Atom, float]] = []
    known = propagate(
        jaxpr,
        [
            Linearity.AFFINE if origin.origin is Origin.VARIABLE else Linearity.CONSTANT
            for origin in origins
        ],
        (),
    )
    for atom, sign in summands(output, 1.0, producers):
        if isinstance(atom, Var) and known[atom] is Linearity.NONLINEAR:
            nonlinear.append((atom, sign))
        else:
            affine.append((atom, sign))
    groups = [(pair,) for pair in nonlinear] if split_sums else [tuple(nonlinear)]
    constants = [np.asarray(constant) for constant in closed.consts]
    shape = shape_of(output)
    terms: list[Term] = []
    for group in groups:
        if not group:
            continue
        built = None
        if split_sums:
            built = rows_term(jaxpr, origins, group[0], shape, producers, constants)
        terms.append(built if built is not None else subterm(jaxpr, origins, group))
    return Analysis(
        shape=shape,
        constants=tuple(constants),
        affine=subterm(jaxpr, origins, tuple(affine)) if affine else None,
        nonlinear=tuple(terms),
    )


def variable_views(
    jaxpr: Jaxpr, origins: Sequence[InputRef]
) -> Callable[[Atom], tuple[np.ndarray, np.ndarray] | None]:
    """Returns a function that finds which variables the entries of a value are.

    For a value built from variables alone, through reshapes, slices, transposes and
    concatenations, it returns two arrays of the shape of the value, holding for each entry
    the leaf of the variables and the flat position in that leaf. Other values give `None`.
    """
    initial = {}
    for var, ref in zip(jaxpr.invars, origins, strict=True):
        if ref.origin is Origin.VARIABLE:
            entries = input_entries(var)
            initial[var] = (np.full(entries.shape, ref.index), entries)
    return trace_views(jaxpr, initial)


def trace_views(
    jaxpr: Jaxpr, initial: dict[Var, tuple[np.ndarray, np.ndarray]]
) -> Callable[[Atom], tuple[np.ndarray, np.ndarray] | None]:
    """Returns the function of `variable_views` for a jaxpr whose inputs have known views."""
    producers = {var: equation for equation in jaxpr.eqns for var in equation.outvars}
    known: dict[Var, tuple[np.ndarray, np.ndarray] | None] = dict(initial)

    def view(atom: Atom) -> tuple[np.ndarray, np.ndarray] | None:
        if isinstance(atom, Literal):
            return None
        if atom not in known:
            known[atom] = compute(atom)
        return known[atom]

    def compute(var: Var) -> tuple[np.ndarray, np.ndarray] | None:
        equation = producers.get(var)
        if equation is None:
            return None
        name = equation.primitive.name
        if name in CALL_PRIMITIVES:
            return through_call(equation, var)
        if name not in VIEW_PRIMITIVES:
            return None
        operands = [view(operand) for operand in equation.invars]
        if any(operand is None for operand in operands):
            return None
        parts = [operand for operand in operands if operand is not None]
        parameters = equation.params
        match name:
            case "concatenate":
                axis = parameters["dimension"]
                return (
                    np.concatenate([part[0] for part in parts], axis=axis),
                    np.concatenate([part[1] for part in parts], axis=axis),
                )
            case "stack":
                axis = parameters["axis"]
                return (
                    np.stack([part[0] for part in parts], axis=axis),
                    np.stack([part[1] for part in parts], axis=axis),
                )
            case "reshape" if parameters["dimensions"] is None:
                shape = tuple(parameters["new_sizes"])
                return parts[0][0].reshape(shape), parts[0][1].reshape(shape)
            case "squeeze":
                axes = tuple(parameters["dimensions"])
                return np.squeeze(parts[0][0], axes), np.squeeze(parts[0][1], axes)
            case "broadcast_in_dim":
                shape = tuple(parameters["shape"])
                expanded = [1] * len(shape)

                def widen(array: np.ndarray) -> np.ndarray:
                    for axis, target in zip(
                        range(array.ndim),
                        parameters["broadcast_dimensions"],
                        strict=True,
                    ):
                        expanded[target] = array.shape[axis]
                    return np.broadcast_to(array.reshape(expanded), shape)

                return widen(parts[0][0]), widen(parts[0][1])
            case "slice":
                return sliced(parts[0][0], equation), sliced(parts[0][1], equation)
            case "transpose":
                axes = tuple(parameters["permutation"])
                return np.transpose(parts[0][0], axes), np.transpose(parts[0][1], axes)
            case "rev":
                axes = tuple(parameters["dimensions"])
                return np.flip(parts[0][0], axes), np.flip(parts[0][1], axes)
            case "convert_element_type" if jnp.issubdtype(
                parameters["new_dtype"], jnp.floating
            ):
                return parts[0]
            case "copy" | "copy_p":
                return parts[0]
        return None

    def through_call(
        equation: JaxprEqn, var: Var
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns the view of an output of a call, from the views of its operands."""
        body = next(
            equation.params[key]
            for key in SUBJAXPR_PARAMETERS
            if key in equation.params
        )
        operand_views = {
            inner: operand_view
            for inner, operand in zip(body.jaxpr.invars, equation.invars, strict=True)
            if (operand_view := view(operand)) is not None
        }
        inner_view = trace_views(body.jaxpr, operand_views)
        return inner_view(body.jaxpr.outvars[equation.outvars.index(var)])

    return view


VIEW_PRIMITIVES = frozenset(
    {
        "concatenate",
        "stack",
        "reshape",
        "squeeze",
        "broadcast_in_dim",
        "slice",
        "transpose",
        "rev",
        "convert_element_type",
        "copy",
        "copy_p",
    }
)
"""Primitives that move the entries of a value around without changing them."""


def rows_term(
    jaxpr: Jaxpr,
    origins: Sequence[InputRef],
    pair: tuple[Atom, float],
    shape: tuple[int, ...],
    producers: dict[Var, JaxprEqn],
    constants: list[np.ndarray],
) -> Term | None:
    """Returns the term of `rows` that a nonlinear value is, or `None` if it is not one.

    The value has to be the result of the marked call itself, for the whole output. The
    vector it reads has to be made of variables alone, and each parameter of the rows an
    input of the function. Otherwise the value is an ordinary term. The constants that the
    row function closes over are appended to `constants`.
    """
    atom, sign = pair
    equation = producers.get(atom) if isinstance(atom, Var) else None
    description = None if equation is None else description_of(equation)
    if equation is None or description is None or shape_of(atom) != shape:
        return None
    vector, *row_parameters = equation.invars
    view = variable_views(jaxpr, origins)(vector)
    references = dict(zip(jaxpr.invars, origins, strict=True))
    parameter_refs = [
        references.get(operand) if isinstance(operand, Var) else None
        for operand in row_parameters
    ]
    if (
        view is None
        or view[0].ndim != 1
        or any(ref is None or ref.origin is Origin.VARIABLE for ref in parameter_refs)
    ):
        return None
    inputs_per_row = description.inputs.shape[1]
    row_avals = [
        jax.ShapeDtypeStruct(shape_of(operand)[1:], dtype_of(operand))
        for operand in row_parameters
    ]

    def row(entries: Array, *parameters: Array) -> Array:
        return description.function(
            entries, jax.tree.unflatten(description.structure, parameters)
        )

    closed = jax.make_jaxpr(row)(
        jax.ShapeDtypeStruct((inputs_per_row,), jnp.float64), *row_avals
    )
    own_constants = [
        InputRef(Origin.CONSTANT, len(constants) + index)
        for index in range(len(closed.consts))
    ]
    constants.extend(np.asarray(constant) for constant in closed.consts)
    refs = [
        *own_constants,
        InputRef(Origin.GATHER, 0),
        *(ref for ref in parameter_refs if ref is not None),
    ]
    leaves, entries = view
    return Term(
        jaxpr=partial_eval.convert_constvars_jaxpr(closed.jaxpr),
        signs=(sign,),
        inputs=tuple(refs),
        selections=(None,) * len(refs),
        linearity=Linearity.NONLINEAR,
        gather=RowSource(
            leaves=leaves[description.inputs], entries=entries[description.inputs]
        ),
        row_axes=(
            *(False,) * len(own_constants),
            True,
            *(True,) * len(row_parameters),
        ),
    )


class RowPiece(NamedTuple):
    """One nonlinear term of the function of `rows`, as a function of the entries it reads."""

    function: RowFunction
    """The function of the entries in `columns` and of the leaves in `leaves`."""
    columns: Int[np.ndarray, " k"]
    """The positions among the entries of a row that the function reads."""
    leaves: tuple[int, ...]
    """The parameters of a row that the function reads, as positions among the leaves."""
    structure: jax.tree_util.PyTreeDef
    """Structure of the parameters of the function, which is a list of leaves."""


class RowSplit(NamedTuple):
    """The function of `rows` as an affine part and nonlinear pieces, which sum to it."""

    affine: Callable[..., Array] | None
    """The affine part, as a function of the entries and the leaves of one row."""
    pieces: tuple[RowPiece, ...]


def split_rows(
    function: RowFunction,
    structure: jax.tree_util.PyTreeDef,
    width: int,
    row_avals: Sequence[jax.ShapeDtypeStruct],
) -> RowSplit:
    """Splits the function of one row like the analysis splits any function.

    Each nonlinear summand reads only its own entries and parameters, so SCIP gets it with
    only the variables that it needs, and the affine part becomes linear rows. A function
    that is a single term over all of its inputs stays as it is.
    """

    def flat(entries: Array, *leaves: Array) -> Array:
        return jnp.asarray(function(entries, jax.tree.unflatten(structure, leaves)))

    analysis = analyse(
        flat, [jax.ShapeDtypeStruct((width,), jnp.float64)], row_avals, True
    )
    if analysis.shape != ():
        raise ValueError(
            f"The function of rows must return a scalar, not shape {analysis.shape}."
        )
    constants = analysis.constants
    affine = None
    if analysis.affine is not None:
        affine_term = analysis.affine

        def affine(entries: Array, *leaves: Array) -> Array:
            return evaluate_term(affine_term, (), constants, [entries], leaves)

    pieces = []
    for term in analysis.nonlinear:
        used = sorted(
            {ref.index for ref in term.inputs if ref.origin is Origin.PARAMETER}
        )
        selection = next(
            selection
            for ref, selection in zip(term.inputs, term.selections, strict=True)
            if ref.origin is Origin.VARIABLE
        )
        columns = np.arange(width) if selection is None else np.asarray(selection)
        pieces.append(
            RowPiece(
                function=piece_function(term, constants, used),
                columns=columns,
                leaves=tuple(used),
                structure=jax.tree.structure(list(used)),
            )
        )
    if (
        affine is None
        and len(pieces) == 1
        and len(pieces[0].columns) == width
        and len(pieces[0].leaves) == len(row_avals)
    ):
        return RowSplit(
            None, (pieces[0]._replace(function=function, structure=structure),)
        )
    return RowSplit(affine, tuple(pieces))


def piece_function(
    term: Term, constants: Sequence[np.ndarray], used: Sequence[int]
) -> RowFunction:
    """Returns the function of the entries and the used leaves of a row that is a term."""

    def evaluate(entries: Array, leaves: Sequence[Array]) -> Array:
        arguments = []
        for ref, selection in zip(term.inputs, term.selections, strict=True):
            if ref.origin is Origin.VARIABLE:
                arguments.append(entries)
                continue
            value = (
                constants[ref.index]
                if ref.origin is Origin.CONSTANT
                else leaves[used.index(ref.index)]
            )
            arguments.append(
                value if selection is None else jnp.ravel(value)[np.asarray(selection)]
            )
        return sum_outputs(term, (), arguments)

    return evaluate


def shape_of(atom: Atom) -> tuple[int, ...]:
    """Returns the shape of a value of a jaxpr."""
    return tuple(cast(ShapedArray, atom.aval).shape)


def dtype_of(atom: Atom) -> np.dtype:
    """Returns the type of the entries of a value of a jaxpr."""
    return cast(ShapedArray, atom.aval).dtype


def summands(
    atom: Atom, sign: float, producers: dict[Var, JaxprEqn]
) -> list[tuple[Atom, float]]:
    """Returns the terms of a value that is a sum, with their signs."""
    pending = [(atom, sign)]
    terms: list[tuple[Atom, float]] = []
    # A long chain of additions is deep, so a stack replaces recursion, with operands
    # pushed in reverse to keep the terms in the order of the sum.
    while pending:
        atom, sign = pending.pop()
        equation = producers.get(atom) if isinstance(atom, Var) else None
        if equation is None or equation.primitive.name not in SPLIT_PRIMITIVES:
            terms.append((atom, sign))
            continue
        match equation.primitive.name, equation.invars:
            case "neg", [operand]:
                pending.append((operand, -sign))
            case "sub", [first, second]:
                pending += [(second, -sign), (first, sign)]
            case _:
                pending += [(operand, sign) for operand in reversed(equation.invars)]
    return terms


def subterm(
    jaxpr: Jaxpr, origins: Sequence[InputRef], pairs: tuple[tuple[Atom, float], ...]
) -> Term:
    """Returns the term that sums the given values of a jaxpr, over only its own inputs."""
    outputs = jaxpr.replace(outvars=[atom for atom, _ in pairs])
    pruned, used = partial_eval.dce_jaxpr(
        outputs, [True] * len(pairs), instantiate=False
    )
    inputs = tuple(
        origin for origin, is_used in zip(origins, used, strict=True) if is_used
    )
    classes = [
        Linearity.AFFINE if origin.origin is Origin.VARIABLE else Linearity.CONSTANT
        for origin in inputs
    ]
    linearity = max(
        linearity_of(pruned, classes, ()),
        default=Linearity.CONSTANT,
    )
    selections = tuple(read_entries(pruned, var) for var in pruned.invars)
    return Term(
        jaxpr=compact(pruned, selections),
        signs=tuple(sign for _, sign in pairs),
        inputs=inputs,
        selections=selections,
        linearity=linearity,
    )


def compact(jaxpr: Jaxpr, selections: Sequence[tuple[int, ...] | None]) -> Jaxpr:
    """Returns the jaxpr with each selected input replaced by a vector of its entries.

    `read_entries` selects only inputs read through chains of `slice`, so the end of each
    chain becomes a slice of the compact vector, or a concatenation where the entries are
    not evenly spaced in it.
    """
    if all(selection is None for selection in selections):
        return jaxpr
    sources = {
        var: {entry: index for index, entry in enumerate(selection)}
        for var, selection in zip(jaxpr.invars, selections, strict=True)
        if selection is not None
    }

    def evaluate(*arguments: Array) -> list[Array]:
        values: dict[Var, Array] = dict(zip(jaxpr.invars, arguments, strict=True))
        views = {var: (var, input_entries(var)) for var in sources}

        def read(atom: Atom) -> Any:
            if isinstance(atom, Literal):
                return atom.val
            if atom in views:
                source, entries = views[atom]
                return slice_entries(values[source], sources[source], entries)
            return values[atom]

        for equation in jaxpr.eqns:
            match equation.primitive.name, equation.invars:
                case "slice", [Var() as var] if var in views:
                    source, entries = views[var]
                    views[equation.outvars[0]] = (source, sliced(entries, equation))
                    continue
            outputs = equation.primitive.bind(
                *map(read, equation.invars), **equation.params
            )
            multiple = equation.primitive.multiple_results
            result = outputs if multiple else [outputs]
            values.update(zip(equation.outvars, result, strict=True))
        return [read(atom) for atom in jaxpr.outvars]

    avals = [
        jax.ShapeDtypeStruct((len(sources[var]),), dtype_of(var))
        if var in sources
        else var.aval
        for var in jaxpr.invars
    ]
    closed = jax.make_jaxpr(evaluate)(*avals)
    assert not closed.consts, "Compacting a jaxpr added constants."
    return closed.jaxpr


def input_entries(var: Var) -> np.ndarray:
    """Returns the flat index of each entry of an input, in the shape of the input."""
    shape = shape_of(var)
    return np.arange(math.prod(shape)).reshape(shape)


def sliced(entries: np.ndarray, equation: JaxprEqn) -> np.ndarray:
    """Returns the entries that a `slice` equation takes from entries of an input."""
    region = tuple(
        slice(start, limit, stride)
        for start, limit, stride in zip(
            equation.params["start_indices"],
            equation.params["limit_indices"],
            equation.params["strides"] or (1,) * entries.ndim,
            strict=True,
        )
    )
    return entries[region]


def slice_entries(
    vector: Array, position: dict[int, int], entries: np.ndarray
) -> Array:
    """Returns the given entries of an input from the vector of its read entries."""
    indices = [position[int(entry)] for entry in entries.ravel()]
    steps = set(np.diff(indices).tolist()) or {1}
    if len(steps) == 1:
        part = jax.lax.slice(vector, (indices[0],), (indices[-1] + 1,), (steps.pop(),))
    else:
        part = jnp.concatenate(
            [jax.lax.slice(vector, (index,), (index + 1,)) for index in indices]
        )
    return part.reshape(entries.shape)


def read_entries(jaxpr: Jaxpr, var: Var) -> tuple[int, ...] | None:
    """Returns the flat indices of the entries of an input that a jaxpr reads.

    Only inputs that are read through chains of `slice` get a selection, which covers
    static indexing such as `x[3]`, `x[2:5]` and `x[1:][2]`. Any other use reads the whole
    input.
    """
    if var in jaxpr.outvars:
        return None
    views = {var: input_entries(var)}
    is_read = np.zeros(views[var].size, dtype=bool)
    for equation in jaxpr.eqns:
        match equation.primitive.name, equation.invars:
            case "slice", [Var() as operand] if operand in views:
                views[equation.outvars[0]] = sliced(views[operand], equation)
                continue
        for operand in equation.invars:
            if operand is var:
                return None
            if isinstance(operand, Var) and operand in views:
                is_read[views[operand].ravel()] = True
    for atom in jaxpr.outvars:
        if isinstance(atom, Var) and atom in views:
            is_read[views[atom].ravel()] = True
    if is_read.all():
        return None
    return tuple(int(index) for index in np.flatnonzero(is_read))


def linearity_of(
    jaxpr: Jaxpr, inputs: Sequence[Linearity], constants: Sequence[Linearity]
) -> list[Linearity]:
    """Returns the linearity of each output of a jaxpr from the linearity of its inputs."""
    known = propagate(jaxpr, inputs, constants)
    return [
        known[atom] if isinstance(atom, Var) else Linearity.CONSTANT
        for atom in jaxpr.outvars
    ]


def propagate(
    jaxpr: Jaxpr, inputs: Sequence[Linearity], constants: Sequence[Linearity]
) -> dict[Var, Linearity]:
    """Returns the linearity of every value of a jaxpr from the linearity of its inputs."""
    known: dict[Var, Linearity] = dict(zip(jaxpr.constvars, constants, strict=True))
    known.update(zip(jaxpr.invars, inputs, strict=True))

    def read(atom: Atom) -> Linearity:
        return known[atom] if isinstance(atom, Var) else Linearity.CONSTANT

    for equation in jaxpr.eqns:
        result = equation_linearity(equation, [read(atom) for atom in equation.invars])
        known.update(zip(equation.outvars, result, strict=True))
    return known


def equation_linearity(equation: JaxprEqn, inputs: list[Linearity]) -> list[Linearity]:
    """Returns the linearity of each output of one equation."""
    count = len(equation.outvars)
    most = max(inputs, default=Linearity.CONSTANT)
    if most is Linearity.CONSTANT:
        return [Linearity.CONSTANT] * count
    name = equation.primitive.name
    body = next(
        (equation.params[key] for key in SUBJAXPR_PARAMETERS if key in equation.params),
        None,
    )
    if name not in CALL_PRIMITIVES:
        body = None
    match body:
        case ClosedJaxpr():
            return linearity_of(
                body.jaxpr, inputs, [Linearity.CONSTANT] * len(body.consts)
            )
    if most is Linearity.NONLINEAR:
        return [Linearity.NONLINEAR] * count
    if name in LINEAR_PRIMITIVES:
        return [Linearity.AFFINE] * count
    if name == "convert_element_type" and jnp.issubdtype(
        equation.params["new_dtype"], jnp.floating
    ):
        return [Linearity.AFFINE] * count
    if name in INDEXED_PRIMITIVES:
        position = INDEXED_PRIMITIVES[name]
        indices = inputs[position:] if name != "scatter-add" else inputs[1:2]
        is_constant = all(value is Linearity.CONSTANT for value in indices)
        return [Linearity.AFFINE if is_constant else Linearity.NONLINEAR] * count
    if name == "select_n" and inputs[0] is Linearity.CONSTANT:
        return [Linearity.AFFINE] * count
    if name in BILINEAR_PRIMITIVES and Linearity.CONSTANT in inputs:
        return [Linearity.AFFINE] * count
    if name == "div" and inputs[1] is Linearity.CONSTANT:
        return [Linearity.AFFINE] * count
    return [Linearity.NONLINEAR] * count


def evaluate_term(
    term: Term,
    shape: tuple[int, ...],
    constants: Sequence[Any],
    variables: Sequence[Any],
    parameters: Sequence[Any],
) -> Float[Array, "*shape"]:
    """Returns the value of a term from all inputs of the analysed function."""
    sources = {
        Origin.CONSTANT: constants,
        Origin.VARIABLE: variables,
        Origin.PARAMETER: parameters,
    }
    arguments = [
        sources[ref.origin][ref.index]
        if selection is None
        else jnp.ravel(sources[ref.origin][ref.index])[np.asarray(selection)]
        for ref, selection in zip(term.inputs, term.selections, strict=True)
    ]
    return sum_outputs(term, shape, arguments)


def sum_outputs(term: Term, shape: tuple[int, ...], arguments: Sequence[Any]) -> Array:
    """Returns `Σ sign · output` of a term's jaxpr applied to its inputs."""
    return combine_outputs(
        term, shape, jaxpr_as_fun(ClosedJaxpr(term.jaxpr, []))(*arguments)
    )


def combine_outputs(
    term: Term, shape: tuple[int, ...], outputs: Sequence[Any]
) -> Array:
    """Returns `Σ sign · output` for the outputs of a term's jaxpr."""
    total = jnp.zeros(shape, jnp.float64)
    for sign, output in zip(term.signs, outputs, strict=True):
        total = total + sign * jnp.broadcast_to(jnp.asarray(output, jnp.float64), shape)
    return total
