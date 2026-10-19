"""Tests of the expression handler's callbacks inside SCIP, including their failure paths."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import PARAMS, TEST_FUNCTIONS, build_handle

import jaxcip as jcp
from jaxcip._handler import FunctionHandle
from jaxcip._problem import Scip
from jaxcip._types import Domain, JaxTerm

INPUTS = np.array([0, 1], np.intc)
"""Inputs of a term over the first two variables."""


def box_problem(lower: float, upper: float, count: int = 2) -> Scip:
    """Returns a problem with `count` continuous variables in `[lower, upper]`."""
    scip = Scip("box")
    for _ in range(count):
        scip.add_variable(lower, upper, 0.0, Domain.CONTINUOUS)
    return scip


def hessian_direction(
    function: Callable[[jax.Array, jax.Array], jax.Array],
    x: np.ndarray,
    direction: np.ndarray,
    options: jcp.HandlerOptions = jcp.HandlerOptions(),
    handle: FunctionHandle | None = None,
) -> np.ndarray | None:
    """Returns SCIP's Hessian-vector product of a function of two variables."""
    handle = build_handle(function) if handle is None else handle
    scip = box_problem(-10.0, 10.0)
    term = JaxTerm(handle, PARAMS, INPUTS, 1.0)
    return scip.hessian_direction(term, options, x, direction)


@pytest.mark.parametrize("name", ["product", "six_hump_camel", "newton_sqrt"])
@pytest.mark.parametrize("eval_gradient", [True, False])
def test_hessian_direction(name: str, eval_gradient: bool) -> None:
    """Checks the Hessian-vector product of the `BWFWDIFF` callback against JAX."""
    function = TEST_FUNCTIONS[name]
    x = np.array([0.7, 1.3])
    direction = np.array([0.4, -1.1])
    options = jcp.HandlerOptions(eval_gradient=eval_gradient)
    expected = jax.hessian(function)(jnp.asarray(x), jnp.asarray(PARAMS)) @ direction
    product = hessian_direction(function, x, direction, options)
    assert product is not None
    np.testing.assert_allclose(product, expected, rtol=1e-10)


def test_hessian_direction_errors() -> None:
    """Checks the errors of points and directions that do not cover every variable."""
    with pytest.raises(ValueError, match="2 variables, got 3"):
        hessian_direction(TEST_FUNCTIONS["product"], np.zeros(3), np.zeros(2))


def test_hessian_unavailable() -> None:
    """Checks that no Hessian comes without the program or with Hessians switched off."""
    function = TEST_FUNCTIONS["product"]
    x, direction = np.array([0.7, 1.3]), np.array([1.0, 0.0])
    without = build_handle(function, optional=False)
    for options, handle in (
        (jcp.HandlerOptions(hessian=False), None),
        (jcp.HandlerOptions(), without),
    ):
        assert hessian_direction(function, x, direction, options, handle) is None


@pytest.mark.parametrize("eval_gradient", [True, False])
@pytest.mark.parametrize(
    "function",
    [
        lambda x, p: jnp.log(x[0]) * x[1],
        lambda x, p: jnp.sqrt(x[0]) * x[1],
        lambda x, p: x[0] ** 1.5 * x[1],
    ],
    ids=["infinite_value", "infinite_gradient", "infinite_hessian"],
)
def test_non_finite_derivatives(
    function: Callable[[jax.Array, jax.Array], jax.Array], eval_gradient: bool
) -> None:
    """Checks that non-finite values and derivatives give no Hessian."""
    options = jcp.HandlerOptions(eval_gradient=eval_gradient)
    point, direction = np.array([0.0, 1.0]), np.array([1.0, 1.0])
    assert hessian_direction(function, point, direction, options) is None


def test_unbounded_variables() -> None:
    """Checks a problem whose variables are bounded only by linear constraints.

    Until propagation bounds them, intervals, estimators, monotonicity and reverse
    propagation see infinite boxes.
    """
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-math.inf, upper=math.inf, shape=(2,)),
        objective=lambda x, p: (x[0] - 1) ** 2 * jnp.exp(x[1]) - x[1],
        constraints=(jcp.Constraint(lambda x, p: x, lower=-3.0, upper=3.0),),
    )
    options = jcp.SolveOptions(
        node_limit=10000,
        settings=(jcp.ScipSetting(jcp.ScipParameter.PRESOLVING_MAXROUNDS, 0),),
    )
    solution = jcp.solve(problem, options=options)
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(-3.0, abs=1e-5)


def test_overflowing_relaxations() -> None:
    """Checks a function whose intervals and estimators overflow on its box."""
    problem = jcp.Problem(
        variables=jcp.Variable(
            lower=jnp.array([0.0, 1.0]), upper=jnp.array([800.0, 2.0])
        ),
        objective=lambda x, p: jnp.exp(x[0]) * jnp.exp(-x[0]) * x[1],
    )
    solution = jcp.solve(problem, options=jcp.SolveOptions(node_limit=1000))
    assert solution.objective == pytest.approx(1.0, abs=1e-5)


def test_free_variables() -> None:
    """Checks a problem whose variables have no bounds at all.

    Initial estimators, estimators and reverse propagation see infinite boxes throughout.
    """
    problem = jcp.Problem(
        variables=jcp.Variable(shape=(2,)),
        objective=lambda x, p: jnp.sqrt(1 + x[0] ** 2) * jnp.exp(x[1] ** 2),
    )
    solution = jcp.solve(problem, options=jcp.SolveOptions(node_limit=1000))
    assert solution.objective == pytest.approx(1.0, abs=1e-5)


def test_steep_estimators() -> None:
    """Checks a function whose estimators have coefficients too large for SCIP's LP.

    The minimum of `(x0 - 15)⁸ x1 - x0` is at `x0 = 15 + 8^(-1/7)` and `x1 = 1`.
    """
    problem = jcp.Problem(
        variables=jcp.Variable(
            lower=jnp.array([0.0, 1.0]), upper=jnp.array([30.0, 2.0])
        ),
        objective=lambda x, p: (x[0] - 15) ** 8 * x[1] - x[0],
    )
    solution = jcp.solve(problem, options=jcp.SolveOptions(node_limit=1000))
    root = 8 ** (-1 / 7)
    assert solution.objective == pytest.approx(root**8 - 15 - root, 1e-6)


def test_compared_expressions() -> None:
    """Checks that SCIP orders and merges expressions of one function when simplifying."""
    handle = build_handle(TEST_FUNCTIONS["six_hump_camel"])
    scip = box_problem(-2.0, 2.0)
    scip.set_parameter("display/verblevel", 0)
    other = np.array([0.5, 0.5])
    swapped = INPUTS[::-1].copy()
    terms = [
        JaxTerm(handle, PARAMS, INPUTS, 1.0),
        JaxTerm(handle, other, INPUTS, 1.0),
        JaxTerm(handle, PARAMS, swapped, 1.0),
        JaxTerm(handle, PARAMS, INPUTS, 1.0),
        JaxTerm(build_handle(TEST_FUNCTIONS["product"]), PARAMS, INPUTS, 1.0),
    ]
    scip.add_jax_constraint(
        terms,
        np.zeros(0, np.intc),
        np.zeros(0),
        0.0,
        -math.inf,
        1.0,
        jcp.HandlerOptions(),
    )
    scip.solve()
    assert scip.status is jcp.Status.OPTIMAL
