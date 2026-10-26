"""Tests of the jaxpr analysis into an affine part and nonlinear terms."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcip._analysis import Linearity, analyse, evaluate_term, read_entries

WEIGHTS = jnp.array([1.0, -2.0, 0.5, 3.0])
"""An array that functions close over, which becomes a constant of the jaxpr."""


class Case(NamedTuple):
    """A function of a vector of four variables and two parameters."""

    function: Callable[[jax.Array, jax.Array], jax.Array]
    num_nonlinear: int
    """Number of cuts when sums are split."""


CASES = {
    "linear": Case(lambda x, p: 2 * x[0] + x[1] - p[0], 0),
    "constant": Case(lambda x, p: 2 * p[0], 0),
    "sum": Case(lambda x, p: x[0] ** 2 + jnp.sin(x[1]) - x[2], 2),
    "negated": Case(lambda x, p: -(x[0] * x[1]), 1),
    "closed_over": Case(lambda x, p: jnp.dot(WEIGHTS, x) + jnp.exp(x[3]), 1),
    "jit_affine": Case(lambda x, p: jax.jit(lambda y: 2 * y)(x).sum(), 0),
    "jit_nonlinear": Case(lambda x, p: jax.jit(lambda y: y**2)(x).sum(), 1),
    "loop": Case(
        lambda x, p: jax.lax.fori_loop(0, 3, lambda _, guess: guess * x[0], 1.0) + x[1],
        1,
    ),
    "checkpoint": Case(lambda x, p: jax.checkpoint(lambda y: y * p[0])(x).sum(), 0),
    "to_float32": Case(lambda x, p: x[0].astype(jnp.float32) * 1.0, 0),
    "to_integer": Case(lambda x, p: x[0].astype(jnp.int32) * 1.0, 1),
    "stack": Case(lambda x, p: jnp.stack([x[0], p[0], x[3]])[jnp.array([2])].sum(), 0),
    "gather": Case(lambda x, p: x[jnp.array([0, 2])].sum(), 0),
    "variable_index": Case(
        lambda x, p: jax.lax.dynamic_index_in_dim(
            x, jnp.clip(x[0], 0, 3).astype(jnp.int32), keepdims=False
        ),
        1,
    ),
    "update_slice": Case(
        lambda x, p: jax.lax.dynamic_update_slice(x, p[:1], (1,)).sum(), 0
    ),
    "scatter_add": Case(lambda x, p: x.at[jnp.array([1])].add(p[:1]).sum(), 0),
    "constant_select": Case(lambda x, p: jnp.where(p[0] > 0, x[0], x[1]), 0),
    "variable_select": Case(lambda x, p: jnp.where(x[0] > 0, x[0], x[1]), 1),
    "divide_by_constant": Case(lambda x, p: x[0] / p[1], 0),
    "divide_by_variable": Case(lambda x, p: p[1] / x[0], 1),
    "vector": Case(lambda x, p: x * p[0] + x**2, 1),
    "identity": Case(lambda x, p: x, 0),
}
"""Functions with the expected outcome of the analysis."""

VARIABLES = (jax.ShapeDtypeStruct((4,), jnp.float64),)
PARAMETERS = (jax.ShapeDtypeStruct((2,), jnp.float64),)


@pytest.mark.parametrize("split_sums", [True, False])
@pytest.mark.parametrize("name", sorted(CASES))
def test_terms_sum_to_the_function(name: str, split_sums: bool) -> None:
    """Checks the classification of each case and that its terms sum to the function."""
    case = CASES[name]
    analysis = analyse(case.function, VARIABLES, PARAMETERS, split_sums)
    if analysis.affine is not None:
        assert analysis.affine.linearity <= Linearity.AFFINE
    expected_count = case.num_nonlinear if split_sums else min(case.num_nonlinear, 1)
    assert len(analysis.nonlinear) == expected_count
    assert all(term.linearity is Linearity.NONLINEAR for term in analysis.nonlinear)
    rng = np.random.default_rng(len(name))
    for _ in range(3):
        x, params = rng.uniform(-2, 2, 4), rng.uniform(0.5, 2, 2)
        cuts = [
            evaluate_term(term, shape, analysis.constants, [x], [params])
            for term, shape in zip(analysis.nonlinear, analysis.cut_shapes, strict=True)
        ]
        if analysis.affine is None:
            assert len(cuts) == 1
            total = cuts[0]
        else:
            total = evaluate_term(
                analysis.affine, analysis.shape, analysis.constants, [x], [params], cuts
            )
        expected = case.function(jnp.asarray(x), jnp.asarray(params))
        np.testing.assert_allclose(total, expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("function", "selection"),
    [
        (lambda x, p: x[0] * x[2], (0, 2)),
        (lambda x, p: jnp.prod(x[::2] ** 2), (0, 2)),
        (lambda x, p: x[0] * x[1] * x[2] * x[3], None),
        (lambda x, p: jnp.prod(x), None),
        (lambda x, p: x[0] * jnp.prod(x), None),
    ],
)
def test_selections(
    function: Callable[[jax.Array, jax.Array], jax.Array],
    selection: tuple[int, ...] | None,
) -> None:
    """Checks the entries of the variables that a nonlinear term reads."""
    analysis = analyse(function, VARIABLES, PARAMETERS, split_sums=True)
    (term,) = analysis.nonlinear
    assert term.selections == (selection,)


def test_multiple_outputs() -> None:
    """Checks that a function with several outputs is refused."""

    def both(x: jax.Array, p: jax.Array) -> Any:
        return x, p

    with pytest.raises(ValueError, match="one array"):
        analyse(both, VARIABLES, PARAMETERS, split_sums=True)


def test_long_sum() -> None:
    """Checks that a sum longer than Python's recursion limit splits into its terms."""

    def long_sum(x: jax.Array, p: jax.Array) -> jax.Array:
        total = x[0] ** 2
        for index in range(1, 1500):
            total = total + p[0] * x[index % 4]
        return total

    analysis = analyse(long_sum, VARIABLES, PARAMETERS, split_sums=True)
    assert analysis.affine is not None
    assert len(analysis.nonlinear) == 1


def test_terms_reading_different_entries_share_a_jaxpr() -> None:
    """Checks that terms differing only in the entries they read have equal jaxprs."""
    analysis = analyse(
        lambda x, p: x[0] * x[2] + x[1] * x[3], VARIABLES, PARAMETERS, split_sums=True
    )
    first, second = analysis.nonlinear
    assert (first.selections, second.selections) == (((0, 2),), ((1, 3),))
    assert str(first.jaxpr) == str(second.jaxpr)


def test_unevenly_spaced_selection() -> None:
    """Checks a slice whose entries are not evenly spaced among the read entries."""

    def function(x: jax.Array, p: jax.Array) -> jax.Array:
        return jnp.sum(x[:, :2] ** 2) * x[0, 2]

    variables = (jax.ShapeDtypeStruct((3, 3), jnp.float64),)
    analysis = analyse(function, variables, PARAMETERS, split_sums=True)
    (term,) = analysis.nonlinear
    assert term.selections == ((0, 1, 2, 3, 4, 6, 7),)
    x = np.random.default_rng(0).uniform(-2, 2, (3, 3))
    params = np.ones(2)
    value = evaluate_term(term, analysis.shape, analysis.constants, [x], [params])
    np.testing.assert_allclose(value, function(jnp.asarray(x), jnp.asarray(params)))


def test_chained_slices_select_entries() -> None:
    """Checks that slices of slices select entries and leave no offsets in the jaxpr."""

    def function(x: jax.Array, p: jax.Array) -> jax.Array:
        rest = p[1:]
        return jnp.sin(x[0] * rest[0]) * x[1] + jnp.sin(x[2] * rest[1:][1]) * x[3]

    variables = (jax.ShapeDtypeStruct((4,), jnp.float64),)
    parameters = (jax.ShapeDtypeStruct((4,), jnp.float64),)
    analysis = analyse(function, variables, parameters, split_sums=True)
    first, second = analysis.nonlinear
    assert first.selections == ((0, 1), (1,))
    assert second.selections == ((2, 3), (3,))
    assert str(first.jaxpr) == str(second.jaxpr)
    x = np.random.default_rng(0).uniform(-2, 2, 4)
    params = np.array([5.0, 2.0, 7.0, 3.0])
    cuts = [
        evaluate_term(term, (), analysis.constants, [x], [params])
        for term in analysis.nonlinear
    ]
    assert analysis.affine is not None
    value = evaluate_term(analysis.affine, (), analysis.constants, [x], [params], cuts)
    np.testing.assert_allclose(value, function(jnp.asarray(x), jnp.asarray(params)))


def test_output_slice_reads_its_entries() -> None:
    """Checks that a slice returned by a jaxpr counts as read."""
    closed = jax.make_jaxpr(lambda x: (x[1:][1:2], x[0] ** 2))(jnp.zeros(4))
    assert read_entries(closed.jaxpr, closed.jaxpr.invars[0]) == (0, 2)
