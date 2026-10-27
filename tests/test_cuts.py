"""Tests of splitting a function through linear operations into cuts."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxcip as jcp
from jaxcip import _blocks
from jaxcip._block import BlockHandle
from jaxcip._compile import linear_rows

COUNT = 20
"""Number of variables of the Styblinski-Tang problem."""

TANG_MINIMUM = -78.33233140754284
"""Smallest value of `x⁴ - 16x² + 5x` over the reals, per variable."""


def tang(count: int = COUNT) -> jcp.Problem:
    """Returns the problem of the README: `min Σ x⁴ - 16 x² + 5 x` over a box."""
    return jcp.Problem(
        variables=jcp.Variable(lower=-5.0, upper=5.0, shape=(count,)),
        objective=lambda x, p: jnp.sum(x**4 - 16 * x**2 + 5 * x),
    )


def test_a_sum_over_variables_splits_into_two_cuts_and_a_linear_term() -> None:
    """Checks the cuts, their coefficients and the linear term of the README objective."""
    compiled = jcp.compile(tang(), None, jcp.CompileOptions(disk_cache=False))
    objective = compiled.objective
    assert objective is not None
    first, second = objective.terms
    for term in (first, second):
        assert isinstance(term.handle, BlockHandle)
        assert term.entries == COUNT
        np.testing.assert_array_equal(term.handle.pointers, np.arange(COUNT + 1))
    linear = linear_rows(objective, compiled.variables, [])
    np.testing.assert_array_equal(linear.cut_coefficients[:COUNT], np.ones(COUNT))
    np.testing.assert_array_equal(
        linear.cut_coefficients[COUNT:], np.full(COUNT, -16.0)
    )
    np.testing.assert_array_equal(linear.indices, np.arange(COUNT))
    np.testing.assert_array_equal(linear.coefficients, np.full(COUNT, 5.0))


def test_the_split_objective_has_the_known_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that the problem of the cuts solves to the known optimum."""
    # Three variables are too few to be kept apart from their sum, so the limit is lifted.
    monkeypatch.setattr(_blocks, "SMALL_BLOCK", 0)
    problem = tang(3)
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    assert compiled.objective is not None
    assert len(compiled.objective.terms) == 2
    solution = jcp.solve(
        compiled,
        options=jcp.SolveOptions(time_limit=120.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(3 * TANG_MINIMUM, rel=1e-5)
    assert solution.x is not None
    assert np.allclose(np.asarray(solution.x), -2.9035, atol=1e-3)


def weighted(width: int) -> jcp.Problem:
    """Returns a problem with the rows `W exp(x) ≤ 4` for weights `W` in the parameters."""
    return jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=1.5, shape=(width,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(
            jcp.Constraint(lambda x, p: p["weights"] @ jnp.exp(x), upper=4.0),
        ),
    )


WEIGHTS = np.array([[1.0, 0.0, 2.0], [0.5, 1.5, 0.0]])
"""The weights of two rows over three variables."""

WIDE_WEIGHTS = np.random.default_rng(0).uniform(0.0, 1.0, (2, 24))
"""The weights of two rows over 24 variables, enough to keep the cut apart from its sum."""


def test_a_matrix_of_parameters_gives_the_coefficients_at_solve_time() -> None:
    """Checks that `W @ f(x)` gets the entries of `W` as coefficients of the cut."""
    params = {"weights": jnp.asarray(WIDE_WEIGHTS)}
    compiled = jcp.compile(weighted(24), params, jcp.CompileOptions(disk_cache=False))
    block = compiled.constraints[0]
    [term] = block.terms
    assert isinstance(term.handle, BlockHandle)
    assert term.entries == 24
    for scale in (1.0, 2.5):
        linear = linear_rows(block, compiled.variables, [scale * WIDE_WEIGHTS])
        dense = np.zeros((2, 24))
        for row in range(2):
            columns, coefficients = linear.cut_row(row)
            dense[row, columns] = coefficients
        np.testing.assert_allclose(dense, scale * WIDE_WEIGHTS)


def test_the_weights_change_the_solution_without_compiling_again() -> None:
    """Checks the optimum for two weights against the same rows written one by one."""
    compiled = jcp.compile(
        weighted(3),
        {"weights": jnp.asarray(WEIGHTS)},
        jcp.CompileOptions(disk_cache=False),
    )
    options = jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET)
    objectives = []
    for scale in (1.0, 0.6):
        weights = scale * WEIGHTS
        solution = jcp.solve(compiled, {"weights": jnp.asarray(weights)}, options)
        reference = jcp.solve(
            jcp.Problem(
                variables=jcp.Variable(lower=0.0, upper=1.5, shape=(3,)),
                objective=lambda x, p: -jnp.sum(x),
                constraints=tuple(
                    jcp.Constraint(
                        lambda x, p, row=row: jnp.sum(row * jnp.exp(x)), upper=4.0
                    )
                    for row in weights
                ),
            ),
            options=options,
        )
        assert solution.status is reference.status is jcp.Status.OPTIMAL
        assert solution.objective == pytest.approx(reference.objective, abs=1e-6)
        objectives.append(solution.objective)
    assert objectives[0] != pytest.approx(objectives[1])


def test_dense_cuts_stay_with_the_operation_that_follows() -> None:
    """Checks that a cut whose entries all read every variable joins the sum over it."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=0.2, shape=(24,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(
            jcp.Constraint(lambda x, p: jnp.sum(jnp.exp(jnp.cumsum(x))), upper=100.0),
        ),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    [term] = compiled.constraints[0].terms
    assert term.entries == 1
    assert jcp.solve(compiled).status is jcp.Status.OPTIMAL


def test_small_cuts_stay_with_the_operation_that_follows() -> None:
    """Checks that a cut with few entries and inputs is not separated from its sum."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-5.0, upper=5.0, shape=(3,)),
        objective=lambda x, p: jnp.sum(x**4 - 16 * x**2 + 5 * x),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    assert compiled.objective is not None
    [term] = compiled.objective.terms
    assert term.entries == 1


def test_without_splitting_sums_the_output_is_the_only_cut() -> None:
    """Checks that `split_sums=False` keeps one cut for the whole objective."""
    compiled = jcp.compile(
        tang(), None, jcp.CompileOptions(disk_cache=False, split_sums=False)
    )
    assert compiled.objective is not None
    [term] = compiled.objective.terms
    assert term.entries == 1
    assert compiled.objective.affine is None
