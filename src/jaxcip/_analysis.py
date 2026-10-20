"""Analysis of the jaxpr of a function into an affine part and nonlinear terms.

SCIP handles affine functions much better than opaque ones, since it propagates and
separates linear rows exactly. The analysis splits the output of a function at additions
and subtractions into terms, classifies each term as affine or nonlinear in the variables,
and keeps for every term only the inputs it depends on.
"""

import enum
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import partial_eval
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var, jaxpr_as_fun
from jaxtyping import Array, Float

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


class InputRef(NamedTuple):
    """One input of a term, as a position among the inputs of its kind."""

    origin: Origin
    index: int


class Term(NamedTuple):
    """A sum `Σ sign · output` of values of a function, as a jaxpr of its own inputs."""

    jaxpr: Jaxpr
    """The jaxpr computing the summed values from `inputs`, without constants."""
    signs: tuple[float, ...]
    """Sign of each output of `jaxpr` in the sum."""
    inputs: tuple[InputRef, ...]
    """The inputs of `jaxpr` in order."""
    selections: tuple[tuple[int, ...] | None, ...]
    """For each input, the flat indices of the entries `jaxpr` reads, or `None` for all."""
    linearity: Linearity


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
    for atom, sign in summands(output, 1.0, producers):
        term = subterm(jaxpr, origins, ((atom, sign),))
        match term.linearity:
            case Linearity.NONLINEAR:
                nonlinear.append((atom, sign))
            case _:
                affine.append((atom, sign))
    groups = [(pair,) for pair in nonlinear] if split_sums else [tuple(nonlinear)]
    return Analysis(
        shape=shape_of(output),
        constants=tuple(np.asarray(constant) for constant in closed.consts),
        affine=subterm(jaxpr, origins, tuple(affine)) if affine else None,
        nonlinear=tuple(subterm(jaxpr, origins, group) for group in groups if group),
    )


def shape_of(atom: Atom) -> tuple[int, ...]:
    """Returns the shape of a value of a jaxpr."""
    return tuple(cast(ShapedArray, atom.aval).shape)


def summands(
    atom: Atom, sign: float, producers: dict[Var, JaxprEqn]
) -> list[tuple[Atom, float]]:
    """Returns the terms of a value that is a sum, with their signs."""
    equation = producers.get(atom) if isinstance(atom, Var) else None
    if equation is None or equation.primitive.name not in SPLIT_PRIMITIVES:
        return [(atom, sign)]
    match equation.primitive.name, equation.invars:
        case "neg", [operand]:
            return summands(operand, -sign, producers)
        case "sub", [first, second]:
            return summands(first, sign, producers) + summands(second, -sign, producers)
        case _:
            return [
                pair
                for operand in equation.invars
                for pair in summands(operand, sign, producers)
            ]


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
    return Term(
        jaxpr=pruned,
        signs=tuple(sign for _, sign in pairs),
        inputs=inputs,
        selections=tuple(
            read_entries(pruned, var) if ref.origin is Origin.VARIABLE else None
            for ref, var in zip(inputs, pruned.invars, strict=True)
        ),
        linearity=linearity,
    )


def read_entries(jaxpr: Jaxpr, var: Var) -> tuple[int, ...] | None:
    """Returns the flat indices of the entries of an input that a jaxpr reads.

    Only inputs that are read through `slice` alone get a selection, which covers static
    indexing such as `x[3]` and `x[2:5]`. Any other use reads the whole input.
    """
    if var in jaxpr.outvars:
        return None
    shape = shape_of(var)
    is_read = np.zeros(shape, dtype=bool)
    for equation in jaxpr.eqns:
        if var not in equation.invars:
            continue
        if equation.primitive.name != "slice":
            return None
        strides = equation.params["strides"] or (1,) * len(shape)
        region = tuple(
            slice(start, limit, stride)
            for start, limit, stride in zip(
                equation.params["start_indices"],
                equation.params["limit_indices"],
                strides,
                strict=True,
            )
        )
        is_read[region] = True
    if is_read.all():
        return None
    return tuple(int(index) for index in np.flatnonzero(is_read))


def linearity_of(
    jaxpr: Jaxpr, inputs: Sequence[Linearity], constants: Sequence[Linearity]
) -> list[Linearity]:
    """Returns the linearity of each output of a jaxpr from the linearity of its inputs."""
    known: dict[Var, Linearity] = dict(zip(jaxpr.constvars, constants, strict=True))
    known.update(zip(jaxpr.invars, inputs, strict=True))

    def read(atom: Atom) -> Linearity:
        return known[atom] if isinstance(atom, Var) else Linearity.CONSTANT

    for equation in jaxpr.eqns:
        result = equation_linearity(equation, [read(atom) for atom in equation.invars])
        known.update(zip(equation.outvars, result, strict=True))
    return [read(atom) for atom in jaxpr.outvars]


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
        case Jaxpr():
            return linearity_of(body, inputs, ())
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
    arguments = [sources[ref.origin][ref.index] for ref in term.inputs]
    return sum_outputs(term, shape, arguments)


def sum_outputs(term: Term, shape: tuple[int, ...], arguments: Sequence[Any]) -> Array:
    """Returns `Σ sign · output` of a term's jaxpr applied to its inputs."""
    outputs = jaxpr_as_fun(ClosedJaxpr(term.jaxpr, []))(*arguments)
    total = jnp.zeros(shape, jnp.float64)
    for sign, output in zip(term.signs, outputs, strict=True):
        total = total + sign * jnp.broadcast_to(jnp.asarray(output, jnp.float64), shape)
    return total
