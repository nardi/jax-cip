"""Rows that each read a few variables of a vector, marked for the analysis.

A function of a problem often has many rows of one form, such as `w[r] · x[a[r]] · x[b[r]]`
for many `r`. Written as a vector expression, each row would appear to read every variable
that the vector expression reads, and every SCIP expression would get all of them as
children. `rows` marks the vector expression instead: row `r` is the function `g` of the
entries `vector[inputs[r]]` and of its own slice of the parameters. The analysis finds the
mark in the jaxpr and gives each row only its own variables, with one set of compiled
programs for all rows.

The mark is the name of a `jax.jit` call, which carries the number of a registered
description. The description lives for as long as the analysis of the function that calls
`rows`.
"""

import functools
import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.core import JaxprEqn
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from jaxcip._problem_types import RowFunction

if TYPE_CHECKING:
    from jaxcip._analysis import RowSplit

NAME_PREFIX = "jaxcip_rows_"
"""Start of the name of the `jax.jit` call that marks rows in a jaxpr."""


class RowsDescription(NamedTuple):
    """What the analysis needs to know about the rows of one call of `rows`."""

    function: RowFunction
    inputs: Int[np.ndarray, "rows k"]
    """For each row, the positions in the vector of the entries that the row reads."""
    structure: jax.tree_util.PyTreeDef
    """Structure of the parameters of one row."""


registered: dict[int, RowsDescription] = {}
"""Descriptions of the marked calls that are in use, by number."""

last_number = [0]
"""The number of the latest marked call, which counts up from 1."""

REGISTRY_LIMIT = 4096
"""Largest number of descriptions that are kept, so that traces outside of an analysis do
not accumulate them."""


def mark(vector: Array, description: RowsDescription, leaves: Sequence[Array]) -> Array:
    """Returns the rows of a description over the vector, as a call that the analysis finds."""
    last_number[0] += 1
    number = last_number[0]
    registered[number] = description
    if len(registered) > REGISTRY_LIMIT:
        registered.pop(next(iter(registered)))

    def marked(vector: Array, *row_leaves: Array) -> Array:
        def one(entries: Array, *parameters: Array) -> Array:
            return description.function(
                entries, jax.tree.unflatten(description.structure, parameters)
            )

        return jax.vmap(one)(vector[description.inputs], *row_leaves)

    marked.__name__ = f"{NAME_PREFIX}{number}"
    return jax.jit(marked)(vector, *leaves)


def rows(
    function: RowFunction,
    inputs: Int[ArrayLike, "rows k"],
    params: PyTree[ArrayLike] = None,
) -> Callable[[Float[ArrayLike, " n"]], Float[Array, " rows"]]:
    """Returns a function of a vector that evaluates rows of one form.

    Row `r` is `function(vector[inputs[r]], params_r)`, where `params_r` is the `r`-th slice
    of every leaf of `params`, a pytree of arrays whose leading axis has one entry per row,
    or `None`. The returned function takes the vector and returns the vector of the rows.
    A `Constraint` over it gives each row only its own variables as children, when the
    vector is made of variables alone, and `params` may depend on the parameters of the
    problem. The function of a row is split like any function of a problem: its affine part
    becomes linear terms of the row, and each nonlinear summand reads only its own entries.
    """
    table = np.asarray(inputs, dtype=np.intc)
    if table.ndim != 2:
        raise ValueError(f"The inputs of rows need two axes, not {table.ndim}.")
    if np.any(table < 0):
        raise IndexError("The inputs of rows index missing entries of the vector.")
    count = table.shape[0]
    leaves, structure = jax.tree.flatten(params)
    arrays = [jnp.asarray(leaf) for leaf in leaves]
    if any(array.ndim == 0 or array.shape[0] != count for array in arrays):
        raise ValueError(f"Every parameter of rows needs {count} rows.")

    row_avals = [jax.ShapeDtypeStruct(array.shape[1:], array.dtype) for array in arrays]
    split: list["RowSplit"] = []

    def apply(vector: Float[ArrayLike, " n"]) -> Array:
        vector = jnp.asarray(vector)
        if vector.ndim != 1:
            raise ValueError(f"The vector of rows needs one axis, not {vector.ndim}.")
        if table.size and np.max(table) >= vector.shape[0]:
            raise IndexError("The inputs of rows index missing entries of the vector.")
        if not split:
            # The analysis imports this module for the marks, so it is imported late.
            from jaxcip._analysis import split_rows

            split.append(split_rows(function, structure, table.shape[1], row_avals))
        results = []
        if split[0].affine is not None:
            results.append(jax.vmap(split[0].affine)(vector[table], *arrays))
        for piece in split[0].pieces:
            results.append(
                mark(
                    vector,
                    RowsDescription(
                        piece.function, table[:, piece.columns], piece.structure
                    ),
                    [arrays[index] for index in piece.leaves],
                )
            )
        return functools.reduce(operator.add, results)

    return apply


def description_of(equation: JaxprEqn) -> RowsDescription | None:
    """Returns the description of the rows that a jaxpr equation marks, or `None`."""
    if equation.primitive.name not in ("jit", "pjit"):
        return None
    name: Any = equation.params.get("name", "")
    if not str(name).startswith(NAME_PREFIX):
        return None
    return registered.get(int(str(name)[len(NAME_PREFIX) :]))


def forget_since(number: int) -> None:
    """Drops the descriptions that were registered from `number` on."""
    for key in [key for key in registered if key >= number]:
        del registered[key]
