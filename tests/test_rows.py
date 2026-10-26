"""Tests of `jcp.rows`, from the analysis of its mark to the solved problem."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcip as jcp
from jaxcip import _rows
from jaxcip._analysis import InputRef, Origin, analyse, variable_views
from jaxcip._compile import linear_rows, term_rows

COUNT = 6
"""Number of rows in the test problems."""

WIDTH = 8
"""Number of variables in the test problems."""

TABLE = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [1, 4], [3, 6]], dtype=np.intc)
"""The two entries that each row of the test problems reads."""


def product_row(x: Any, weight: Any) -> jax.Array:
    """Returns a row with a product, a power and a weight."""
    return weight * x[0] * x[1] + jnp.exp(0.3 * x[0]) * x[1] ** 2


def compiled_rows(
    constraint: Any, params: Any = None, **options: Any
) -> jcp.CompiledProblem:
    """Compiles a problem with one constraint over eight variables."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.0, shape=(WIDTH,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(constraint,),
    )
    return jcp.compile(problem, params, jcp.CompileOptions(disk_cache=False, **options))


def weights() -> np.ndarray:
    """Returns one weight for each row."""
    return np.linspace(0.5, 1.5, COUNT)


def test_rows_become_a_term_per_summand_with_their_own_inputs() -> None:
    """Checks that each nonlinear summand of a row is one term, with each row's inputs."""
    row = jcp.rows(product_row, TABLE, weights())
    compiled = compiled_rows(jcp.Constraint(lambda x, p: row(x), upper=40.0))
    block = compiled.constraints[0]
    first, second = block.terms
    assert block.rows == COUNT
    for term in (first, second):
        assert term.term.gather is not None
        assert term.signature.shape == ()
        inputs, _ = term_rows(term, block, compiled.variables, [])
        np.testing.assert_array_equal(inputs, TABLE)
    _, params = term_rows(first, block, compiled.variables, [])
    np.testing.assert_allclose(params[:, -1], weights())
    # The programs are lowered once for all rows, or came from an earlier build.
    assert compiled.statistics.lowered + compiled.statistics.reused == 2
    assert _rows.registered == {}


def test_a_term_reads_only_the_entries_it_needs() -> None:
    """Checks that a summand that reads one entry of a row has only that variable."""
    row = jcp.rows(lambda x, p: jnp.exp(x[0]) + p * x[1] - x[0] * 0.5, TABLE, weights())
    compiled = compiled_rows(jcp.Constraint(lambda x, p: row(x), upper=40.0))
    block = compiled.constraints[0]
    [term] = block.terms
    inputs, _ = term_rows(term, block, compiled.variables, [])
    np.testing.assert_array_equal(inputs, TABLE[:, :1])
    assert block.analysis.affine is not None
    linear = linear_rows(block, compiled.variables, [])
    assert linear.indices.shape == (2 * COUNT,)


def test_row_values_match_the_function() -> None:
    """Checks the value of the compiled terms of each row against the row function."""
    row = jcp.rows(product_row, TABLE, weights())
    compiled = compiled_rows(jcp.Constraint(lambda x, p: row(x), upper=40.0))
    block = compiled.constraints[0]
    point = np.linspace(0.6, 1.9, WIDTH)
    expected = jax.vmap(product_row)(point[TABLE], weights())
    total = np.zeros(COUNT)
    for term in block.terms:
        inputs, params = term_rows(term, block, compiled.variables, [])
        for index in range(COUNT):
            total[index] += term.term.signs[0] * term.handle.evaluate(
                point[inputs[index]], params[index]
            )
    np.testing.assert_allclose(total, np.asarray(expected), rtol=1e-12)


def test_rows_in_a_sum_with_a_sign_and_an_affine_part() -> None:
    """Checks the term, its sign and the affine part of a function that adds them."""
    row = jcp.rows(product_row, TABLE, weights())
    constraint = jcp.Constraint(
        lambda x, p: 3.0 * x[:COUNT] - row(x) + 1.0, lower=-50.0, upper=0.0
    )
    compiled = compiled_rows(constraint)
    block = compiled.constraints[0]
    assert [term.term.signs for term in block.terms] == [(-1.0,), (-1.0,)]
    assert block.analysis.affine is not None
    solution = jcp.solve(
        jcp.Problem(
            variables=jcp.Variable(lower=0.5, upper=2.0, shape=(WIDTH,)),
            objective=lambda x, p: jnp.sum(x),
            constraints=(constraint,),
        ),
        options=jcp.SolveOptions(time_limit=60.0),
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.x is not None
    point = np.asarray(solution.x)
    values = (
        3.0 * point[:COUNT]
        - np.asarray(jax.vmap(product_row)(point[TABLE], weights()))
        + 1.0
    )
    assert np.all(values <= 1e-6) and np.all(values >= -50.0 - 1e-6)


def test_rows_with_parameters_of_the_problem() -> None:
    """Checks that rows can take parameters that change between solves."""

    def constraint(x: jax.Array, params: Any) -> jax.Array:
        return jcp.rows(product_row, TABLE, params["weight"])(x)

    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.0, shape=(WIDTH,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(jcp.Constraint(constraint, upper=9.0),),
    )
    params = {"weight": jnp.asarray(weights())}
    compiled = jcp.compile(problem, params, jcp.CompileOptions(disk_cache=False))
    terms = compiled.constraints[0].terms
    assert all(term.term.gather is not None for term in terms)
    assert any(
        ref.origin is Origin.PARAMETER for term in terms for ref in term.term.inputs
    )
    first = jcp.solve(compiled, params)
    heavier = jcp.solve(compiled, {"weight": 2.0 * params["weight"]})
    assert first.status is heavier.status is jcp.Status.OPTIMAL
    assert heavier.objective > first.objective
    assert compiled.statistics.lowered + compiled.statistics.reused == 2


def test_variables_in_several_leaves() -> None:
    """Checks that a vector stacked from scalar variables is still made of variables."""
    row = jcp.rows(lambda x, p: x[0] * x[1] + jnp.sin(x[1]), np.array([[0, 2], [3, 1]]))
    problem = jcp.Problem(
        variables=(
            jcp.Variable(lower=0.5, upper=2.0),
            jcp.Variable(lower=0.5, upper=2.0, shape=(2,)),
            jcp.Variable(lower=0.5, upper=2.0),
        ),
        objective=lambda x, p: x[0] + x[2],
        constraints=(
            jcp.Constraint(
                lambda x, p: row(jnp.concatenate([x[0][None], x[1], x[2][None]])),
                lower=2.0,
            ),
        ),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    block = compiled.constraints[0]
    product, sine = block.terms
    inputs, _ = term_rows(product, block, compiled.variables, [])
    np.testing.assert_array_equal(inputs, [[0, 2], [3, 1]])
    inputs, _ = term_rows(sine, block, compiled.variables, [])
    np.testing.assert_array_equal(inputs, [[2], [1]])
    assert jcp.solve(compiled).status is jcp.Status.OPTIMAL


def test_rows_of_stacked_scalar_variables_read_their_own_variables() -> None:
    """Checks that a vector stacked from many scalar variables gives each row two inputs."""
    row = jcp.rows(lambda x, p: x[0] * x[1], np.array([[0, 3], [2, 1], [3, 2]]))
    problem = jcp.Problem(
        variables=tuple(jcp.Variable(lower=0.5, upper=2.0) for _ in range(4)),
        objective=lambda x, p: x[0],
        constraints=(jcp.Constraint(lambda x, p: row(jnp.stack(x)), upper=3.0),),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    block = compiled.constraints[0]
    [term] = block.terms
    inputs, _ = term_rows(term, block, compiled.variables, [])
    np.testing.assert_array_equal(inputs, [[0, 3], [2, 1], [3, 2]])


def test_a_vector_that_is_not_made_of_variables_is_an_ordinary_term() -> None:
    """Checks that rows of a computed vector are still solved, as one opaque term."""
    row = jcp.rows(lambda x, p: x[0] * x[1], np.array([[0, 1], [1, 2]]))
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.0, shape=(3,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: row(jnp.exp(x)), lower=10.0),),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    [term] = compiled.constraints[0].terms
    assert term.term.gather is None
    solution = jcp.solve(compiled, options=jcp.SolveOptions(time_limit=60.0))
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.x is not None
    point = np.exp(np.asarray(solution.x))
    assert point[0] * point[1] >= 10.0 - 1e-5 and point[1] * point[2] >= 10.0 - 1e-5


def test_rows_that_are_not_the_whole_output_are_ordinary_terms() -> None:
    """Checks that a function of the rows, such as their sum, is an ordinary term."""
    row = jcp.rows(product_row, TABLE, weights())
    compiled = compiled_rows(jcp.Constraint(lambda x, p: jnp.sum(row(x)), upper=100.0))
    [term] = compiled.constraints[0].terms
    assert term.term.gather is None


def test_rows_without_splitting_sums_are_an_ordinary_term() -> None:
    """Checks that `split_sums=False` keeps one term for the whole function."""
    row = jcp.rows(product_row, TABLE, weights())
    compiled = compiled_rows(
        jcp.Constraint(lambda x, p: row(x), upper=100.0), split_sums=False
    )
    [term] = compiled.constraints[0].terms
    assert term.term.gather is None


def test_affine_rows_have_no_term() -> None:
    """Checks that rows of a linear function belong to the affine part."""
    row = jcp.rows(lambda x, w: w * (x[0] - 2 * x[1]), TABLE, weights())
    compiled = compiled_rows(jcp.Constraint(lambda x, p: row(x), upper=3.0))
    assert compiled.constraints[0].terms == ()
    assert compiled.constraints[0].analysis.affine is not None


def test_rows_with_constants_in_the_function() -> None:
    """Checks that arrays the row function closes over become shared constants."""
    scale = np.array([2.0, 3.0])
    row = jcp.rows(lambda x, p: jnp.sum(scale * jnp.exp(x)), TABLE)
    compiled = compiled_rows(jcp.Constraint(lambda x, p: row(x), upper=60.0))
    block = compiled.constraints[0]
    [term] = block.terms
    assert any(ref.origin is Origin.CONSTANT for ref in term.term.inputs)
    inputs, params = term_rows(term, block, compiled.variables, [])
    point = np.linspace(0.6, 1.9, WIDTH)
    value = term.handle.evaluate(point[inputs[0]], params[0])
    assert value == pytest.approx(float(np.sum(scale * np.exp(point[TABLE[0]]))))


def test_rows_outside_a_problem() -> None:
    """Checks that rows evaluate like any function when they are called directly."""
    row = jcp.rows(product_row, TABLE, weights())
    point = jnp.linspace(0.6, 1.9, WIDTH)
    np.testing.assert_allclose(
        row(point), jax.vmap(product_row)(point[TABLE], jnp.asarray(weights()))
    )


def test_the_registry_stays_small(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that descriptions of calls outside an analysis cannot pile up."""
    monkeypatch.setattr(_rows, "REGISTRY_LIMIT", 3)
    row = jcp.rows(product_row, TABLE, weights())
    point = jnp.linspace(0.6, 1.9, WIDTH)
    for _ in range(8):
        row(point)
    assert len(_rows.registered) == 3
    _rows.registered.clear()


def test_analysis_without_a_view_of_variables_in_one_dimension() -> None:
    """Checks that a marked call on a matrix of variables is read through its reshape."""
    row = jcp.rows(lambda x, p: x[0] * x[1], np.array([[0, 3], [1, 2]]))
    analysis = analyse(
        lambda x: row(x.reshape(-1)),
        [jax.ShapeDtypeStruct((2, 2), jnp.float64)],
        [],
        True,
    )
    [term] = analysis.nonlinear
    assert term.gather is not None
    np.testing.assert_array_equal(term.gather.entries, [[0, 3], [1, 2]])
    np.testing.assert_array_equal(term.gather.leaves, np.zeros((2, 2), dtype=int))


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (lambda: jcp.rows(lambda x, p: x[0], np.array([0, 1])), ValueError, "two axes"),
        (
            lambda: jcp.rows(lambda x, p: x[0], np.array([[0, -1]])),
            IndexError,
            "missing entries",
        ),
        (
            lambda: jcp.rows(lambda x, p: x[0], np.array([[0, 1]]), jnp.ones(2)),
            ValueError,
            "needs 1 rows",
        ),
        (
            lambda: jcp.rows(lambda x, p: x[0], np.array([[0, 1]]), jnp.ones(())),
            ValueError,
            "needs 1 rows",
        ),
        (
            lambda: jcp.rows(lambda x, p: x[0], np.array([[0, 5]]))(jnp.ones(3)),
            IndexError,
            "missing entries",
        ),
        (
            lambda: jcp.rows(lambda x, p: x[0], np.array([[0, 1]]))(jnp.ones((2, 2))),
            ValueError,
            "one axis",
        ),
        (
            lambda: jcp.rows(lambda x, p: x, np.array([[0, 1]]))(jnp.ones(3)),
            ValueError,
            "return a scalar",
        ),
    ],
)
def test_rows_check_their_arguments(
    make: Any, error: type[Exception], message: str
) -> None:
    """Checks the errors for inputs, parameters and vectors that do not fit."""
    with pytest.raises(error, match=message):
        make()


VIEWS: dict[str, Any] = {
    "reshape": lambda x, y: x.reshape(-1),
    "squeeze": lambda x, y: jnp.squeeze(x[None], 0).ravel(),
    "broadcast": lambda x, y: jnp.broadcast_to(y, (2, 3)).ravel(),
    "transpose": lambda x, y: x.T.ravel(),
    "flip": lambda x, y: jnp.flip(y),
    "slice": lambda x, y: x[:, 1:].ravel(),
    "strided": lambda x, y: y[::2],
    "concatenate": lambda x, y: jnp.concatenate([y, x.ravel(), y[:1]]),
    "stack": lambda x, y: jnp.stack([y, x[0], x[1]]).ravel(),
    "stack of scalars": lambda x, y: jnp.stack([y[2], x[1, 0], y[0]]),
    "copy": lambda x, y: jnp.array(y, copy=True),
    "conversion": lambda x, y: y.astype(jnp.float32),
}
"""Ways to move variables into a vector, from a matrix `x` and a vector `y` of variables."""


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_variable_views(name: str) -> None:
    """Checks that the leaf and entry of every element of a moved vector are found."""
    operation = VIEWS[name]
    avals = [
        jax.ShapeDtypeStruct((2, 3), jnp.float64),
        jax.ShapeDtypeStruct((3,), jnp.float64),
    ]
    closed = jax.make_jaxpr(operation)(*avals)
    origins = [InputRef(Origin.VARIABLE, 0), InputRef(Origin.VARIABLE, 1)]
    found = variable_views(closed.jaxpr, origins)(closed.jaxpr.outvars[0])
    assert found is not None
    leaves, entries = found
    sources = [np.zeros((2, 3), dtype=int), np.ones(3, dtype=int)]
    expected_leaves = np.asarray(operation(*map(jnp.asarray, sources)))
    ids = [np.arange(6).reshape(2, 3), np.arange(3)]
    expected_entries = np.asarray(operation(*map(jnp.asarray, ids)))
    np.testing.assert_array_equal(leaves, expected_leaves)
    np.testing.assert_array_equal(entries, expected_entries)


def test_computed_values_are_no_views() -> None:
    """Checks that arithmetic, a parameter and a literal are not views of variables."""
    closed = jax.make_jaxpr(
        lambda x, p: (
            x * 2.0,
            p,
            x + p,
            jnp.concatenate([x, jnp.exp(x)]),
            x.astype(jnp.int32),
            jax.jit(jnp.exp)(x),
        )
    )(jax.ShapeDtypeStruct((3,), jnp.float64), jax.ShapeDtypeStruct((3,), jnp.float64))
    origins = [InputRef(Origin.VARIABLE, 0), InputRef(Origin.PARAMETER, 0)]
    view = variable_views(closed.jaxpr, origins)
    assert all(view(atom) is None for atom in closed.jaxpr.outvars)
    literal = jax.make_jaxpr(lambda x: x + 1.0)(jax.ShapeDtypeStruct((), jnp.float64))
    assert (
        variable_views(literal.jaxpr, origins[:1])(literal.jaxpr.eqns[0].invars[1])
        is None
    )
