"""Tests of the jaxpr analysis into an affine part and nonlinear terms."""

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcip._analysis import Linearity, analyse, evaluate_term

WEIGHTS = jnp.array([1.0, -2.0, 0.5, 3.0])
"""An array that functions close over, which becomes a constant of the jaxpr."""


class Case(NamedTuple):
    """A function of a vector of four variables and two parameters."""

    function: Callable[[jax.Array, jax.Array], jax.Array]
    has_affine: bool
    """Whether the function has an affine or constant part."""
    num_nonlinear: int
    """Number of nonlinear terms when sums are split."""


CASES = {
    "linear": Case(lambda x, p: 2 * x[0] + x[1] - p[0], True, 0),
    "constant": Case(lambda x, p: 2 * p[0], True, 0),
    "sum": Case(lambda x, p: x[0] ** 2 + jnp.sin(x[1]) - x[2], True, 2),
    "negated": Case(lambda x, p: -(x[0] * x[1]), False, 1),
    "closed_over": Case(lambda x, p: jnp.dot(WEIGHTS, x) + jnp.exp(x[3]), True, 1),
    "jit_affine": Case(lambda x, p: jax.jit(lambda y: 2 * y)(x).sum(), True, 0),
    "jit_nonlinear": Case(lambda x, p: jax.jit(lambda y: y**2)(x).sum(), False, 1),
    "loop": Case(
        lambda x, p: jax.lax.fori_loop(0, 3, lambda _, guess: guess * x[0], 1.0) + x[1],
        True,
        1,
    ),
    "checkpoint": Case(
        lambda x, p: jax.checkpoint(lambda y: y * p[0])(x).sum(), True, 0
    ),
    "to_float32": Case(lambda x, p: x[0].astype(jnp.float32) * 1.0, True, 0),
    "to_integer": Case(lambda x, p: x[0].astype(jnp.int32) * 1.0, False, 1),
    "gather": Case(lambda x, p: x[jnp.array([0, 2])].sum(), True, 0),
    "variable_index": Case(
        lambda x, p: jax.lax.dynamic_index_in_dim(
            x, jnp.clip(x[0], 0, 3).astype(jnp.int32), keepdims=False
        ),
        False,
        1,
    ),
    "update_slice": Case(
        lambda x, p: jax.lax.dynamic_update_slice(x, p[:1], (1,)).sum(), True, 0
    ),
    "scatter_add": Case(lambda x, p: x.at[jnp.array([1])].add(p[:1]).sum(), True, 0),
    "constant_select": Case(lambda x, p: jnp.where(p[0] > 0, x[0], x[1]), True, 0),
    "variable_select": Case(lambda x, p: jnp.where(x[0] > 0, x[0], x[1]), False, 1),
    "divide_by_constant": Case(lambda x, p: x[0] / p[1], True, 0),
    "divide_by_variable": Case(lambda x, p: p[1] / x[0], False, 1),
    "vector": Case(lambda x, p: x * p[0] + x**2, True, 1),
    "identity": Case(lambda x, p: x, True, 0),
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
    assert (analysis.affine is not None) == case.has_affine
    if analysis.affine is not None:
        assert analysis.affine.linearity <= Linearity.AFFINE
    expected_count = case.num_nonlinear if split_sums else min(case.num_nonlinear, 1)
    assert len(analysis.nonlinear) == expected_count
    assert all(term.linearity is Linearity.NONLINEAR for term in analysis.nonlinear)
    rng = np.random.default_rng(len(name))
    for _ in range(3):
        x, params = rng.uniform(-2, 2, 4), rng.uniform(0.5, 2, 2)
        terms = [analysis.affine, *analysis.nonlinear]
        total = sum(
            evaluate_term(term, analysis.shape, analysis.constants, [x], [params])
            for term in terms
            if term is not None
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
