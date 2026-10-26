"""Tests of block terms: patterns, compressed programs, and the entries inside SCIP."""

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import jaxormick as jmk
import numpy as np
import pytest

import jaxcip as jcp
from jaxcip._block import BlockHandle, BlockStateRef
from jaxcip._blocks import block_function, detect_pattern
from jaxcip._compile import CompiledTerm, block_inputs, compiled_terms
from jaxcip._problem import Scip
from jaxcip._types import BlockTerm, Domain

WIDTH = 6
"""Number of variables of the test problems."""

TOLERANCE = 1e-9
"""Allowed difference between PJRT and JAX results, which run the same XLA code."""

FUNCTIONS: dict[str, Callable[[jax.Array, Any], jax.Array]] = {
    "neighbours": lambda x, p: x[:-1] * x[1:],
    "mixed": lambda x, p: jnp.stack(
        [x[0] * x[1], jnp.exp(x[2]) - x[0] ** 2, jnp.sum(x**2)]
    ),
    "scaled": lambda x, p: jnp.sin(p["scale"][0] * x[:-1]) * x[1:],
    "constant_entry": lambda x, p: jnp.stack([x[0] * x[1], jnp.asarray(3.0)]),
    "elementwise": lambda x, p: jnp.exp(x) * x,
}
"""Vector functions of `WIDTH` variables whose entries read few of them, and a dense one."""

PARAMS = {"scale": jnp.array([0.7])}
"""The parameters of the test problems."""


def compiled_block(name: str, upper: float = 100.0) -> jcp.CompiledProblem:
    """Compiles a problem with the named function as its only constraint."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(WIDTH,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(jcp.Constraint(FUNCTIONS[name], upper=upper),),
    )
    return jcp.compile(problem, PARAMS, jcp.CompileOptions(disk_cache=False))


def block_of(name: str) -> tuple[CompiledTerm, jcp.CompiledProblem]:
    """Returns the one block term of a test problem."""
    compiled = compiled_block(name)
    [term] = compiled.constraints[0].terms
    assert isinstance(term.handle, BlockHandle)
    return term, compiled


def flat_inputs(
    term: CompiledTerm, compiled: jcp.CompiledProblem
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the variables and the parameter values of a block term."""
    leaves = [np.asarray(PARAMS["scale"])]
    return block_inputs(term, compiled.constraints[0], compiled.variables, leaves)


def random_box(seed: int, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a box inside the domain of the test problems and a point in it."""
    rng = np.random.default_rng(seed)
    first, second = rng.uniform(0.5, 2.5, (2, size))
    lower, upper = np.minimum(first, second), np.maximum(first, second)
    return rng.uniform(lower, upper), lower, upper


def test_the_pattern_of_neighbours() -> None:
    """Checks that each entry of `x[:-1] * x[1:]` reads two variables."""
    term, _ = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    np.testing.assert_array_equal(handle.pointers, np.arange(0, 11, 2))
    np.testing.assert_array_equal(
        handle.columns, np.repeat(np.arange(5), 2) + [0, 1] * 5
    )
    assert (handle.num_inputs, handle.num_entries, handle.nnz) == (WIDTH, 5, 10)


def test_an_entry_without_variables_gets_one() -> None:
    """Checks that an entry that is constant in the variables still has a child."""
    term, _ = block_of("constant_entry")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    np.testing.assert_array_equal(handle.pointers, [0, 2, 3])
    np.testing.assert_array_equal(handle.columns, [0, 1, 0])
    values, jacobian = handle.value_and_jacobian(np.array([1.5, 2.0]), np.zeros(1))
    np.testing.assert_allclose(values, [3.0, 3.0])
    assert jacobian[2] == 0.0


def test_the_pattern_of_a_dense_row() -> None:
    """Checks that a row that sums over all variables reads all of them."""
    term, _ = block_of("mixed")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    assert list(np.diff(handle.pointers)) == [2, 2, WIDTH]


@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_programs_match_dense_jax(name: str) -> None:
    """Checks every program of a block against dense JAX on random points and boxes."""
    term, compiled = block_of(name)
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    function = block_function(term.term, term.signature)
    variables, params = flat_inputs(term, compiled)
    size = len(variables)
    rows, columns = (
        np.repeat(np.arange(handle.num_entries), np.diff(handle.pointers)),
        handle.columns,
    )
    jacobian = jax.jit(jax.jacfwd(function))
    hessian = jax.jit(jax.jacfwd(jax.jacfwd(function)))
    convex = jax.jit(jmk.convex_side(function))
    concave = jax.jit(jmk.concave_side(function))
    bounds = jax.jit(jmk.interval(function))
    gradient_bounds = jax.jit(jmk.interval(jax.jacfwd(function)))
    for seed in range(5):
        point, lower, upper = random_box(seed, size)
        direction = np.linspace(-1.0, 1.0, size) + seed
        expected = np.asarray(function(jnp.asarray(point), jnp.asarray(params)))
        if size == WIDTH:
            by_name = FUNCTIONS[name](jnp.asarray(point), PARAMS)
            np.testing.assert_allclose(expected, np.asarray(by_name), TOLERANCE)
        values, entries = handle.value_and_jacobian(point, params)
        np.testing.assert_allclose(values, expected, TOLERANCE, TOLERANCE)
        dense = np.asarray(jacobian(point, params))
        np.testing.assert_allclose(entries, dense[rows, columns], TOLERANCE, TOLERANCE)
        outside = np.ones_like(dense, dtype=bool)
        outside[rows, columns] = False
        assert np.all(dense[outside] == 0.0)
        product = np.einsum("ijk,k->ij", np.asarray(hessian(point, params)), direction)
        np.testing.assert_allclose(
            handle.hessian_vector(point, direction, params),
            product[rows, columns],
            1e-8,
            1e-8,
        )
        interval = handle.interval(lower, upper, params)
        low, high = bounds(lower, upper, params)
        np.testing.assert_allclose(interval.lower, low, TOLERANCE, TOLERANCE)
        np.testing.assert_allclose(interval.upper, high, TOLERANCE, TOLERANCE)
        gradient_interval = handle.gradient_interval(lower, upper, params)
        low, high = gradient_bounds(lower, upper, params)
        np.testing.assert_allclose(
            gradient_interval.lower,
            np.asarray(low)[rows, columns],
            TOLERANCE,
            TOLERANCE,
        )
        np.testing.assert_allclose(
            gradient_interval.upper,
            np.asarray(high)[rows, columns],
            TOLERANCE,
            TOLERANCE,
        )
        for overestimate, side in ((False, convex), (True, concave)):
            side_values, side_gradient = side(point, lower, upper, params)
            relaxed, entries = handle.relaxation(
                point, lower, upper, params, overestimate
            )
            np.testing.assert_allclose(relaxed, side_values, TOLERANCE, TOLERANCE)
            np.testing.assert_allclose(
                entries, np.asarray(side_gradient)[rows, columns], 1e-8, 1e-8
            )


@pytest.mark.parametrize("name", ["neighbours", "mixed", "constant_entry"])
def test_entries_match_the_programs_of_single_entries(name: str) -> None:
    """Checks the relaxations and bounds of every entry against those of the entry alone."""
    term, compiled = block_of(name)
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    function = block_function(term.term, term.signature)
    variables, params = flat_inputs(term, compiled)
    size = len(variables)
    point, lower, upper = random_box(7, size)
    values, entries = handle.relaxation(point, lower, upper, params)
    interval = handle.interval(lower, upper, params)
    for entry in range(handle.num_entries):
        alone = jmk.convex_side(lambda y, q, e=entry: function(y, q)[e])
        alone_value, alone_gradient = alone(point, lower, upper, params)
        start, stop = handle.pointers[entry], handle.pointers[entry + 1]
        assert values[entry] == pytest.approx(float(alone_value), 1e-9, abs=1e-9)
        dense = np.zeros(size)
        dense[handle.columns[start:stop]] = entries[start:stop]
        np.testing.assert_allclose(dense, alone_gradient, 1e-8, 1e-8)
        low, high = jmk.interval(lambda y, q, e=entry: function(y, q)[e])(
            lower, upper, params
        )
        assert np.asarray(interval.lower)[entry] == pytest.approx(
            float(low), 1e-9, abs=1e-9
        )
        assert np.asarray(interval.upper)[entry] == pytest.approx(
            float(high), 1e-9, abs=1e-9
        )


def test_block_validation() -> None:
    """Checks the errors of a handle with a malformed pattern or programs."""
    term, _ = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    pointers, columns = handle.pointers, handle.columns
    programs = handle.programs
    with pytest.raises(ValueError, match="at least one input"):
        BlockHandle(0, 1, pointers, columns, programs)
    with pytest.raises(ValueError, match="do not match"):
        BlockHandle(WIDTH, 1, pointers[:-1], columns, programs)
    with pytest.raises(ValueError, match="needs a variable"):
        BlockHandle(WIDTH, 1, np.zeros(3, np.intc), columns[:0], programs)
    with pytest.raises(ValueError, match="needs a variable"):
        BlockHandle(WIDTH, 1, np.array([0, 0, 2], np.intc), columns[:2], programs)
    with pytest.raises(IndexError, match="missing inputs"):
        BlockHandle(WIDTH, 1, pointers, columns + WIDTH, programs)
    with pytest.raises(ValueError, match="program is required"):
        BlockHandle(
            WIDTH,
            1,
            pointers,
            columns,
            programs._replace(convex=None),  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError, match="outputs, expected"):
        wrong = programs._replace(hessian_vector=programs.value_and_gradient)
        BlockHandle(WIDTH, 1, pointers, columns, wrong)
    with pytest.raises(ValueError, match="has 3 entries"):
        handle.value_and_jacobian(np.zeros(3), np.zeros(1))
    with pytest.raises(ValueError, match="params has"):
        handle.value_and_jacobian(np.zeros(WIDTH), np.zeros(4))


def test_missing_optional_programs() -> None:
    """Checks that a block without optional programs still works and reports them missing."""
    term, _ = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    reduced = BlockHandle(
        WIDTH,
        1,
        handle.pointers,
        handle.columns,
        handle.programs._replace(hessian_vector=None, gradient_interval=None),
    )
    with pytest.raises(ValueError, match="no hessian_vector program"):
        reduced.hessian_vector(np.ones(WIDTH), np.ones(WIDTH), np.zeros(1))
    values, _ = reduced.value_and_jacobian(np.full(WIDTH, 2.0), np.zeros(1))
    np.testing.assert_allclose(values, np.full(5, 4.0))


def solve_block(name: str, **kwargs: Any) -> jcp.Solution:
    """Solves the problem of a test function with a time limit."""
    return jcp.solve(
        compiled_block(name, **kwargs),
        PARAMS,
        jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )


def scalar_rows(name: str, upper: float) -> jcp.Problem:
    """Returns the problem of a test function with one constraint for each entry."""
    function = FUNCTIONS[name]
    count = int(function(jnp.ones(WIDTH), PARAMS).shape[0])
    return jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(WIDTH,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=tuple(
            jcp.Constraint(lambda x, p, e=entry: function(x, p)[e], upper=upper)
            for entry in range(count)
        ),
    )


@pytest.mark.parametrize("name", ["neighbours", "mixed", "scaled", "elementwise"])
def test_a_block_solves_like_its_scalar_rows(name: str) -> None:
    """Checks that a vector constraint has the optimum of the same rows written one by one."""
    upper = 4.0 if name != "mixed" else 12.0
    block = solve_block(name, upper=upper)
    scalar = jcp.solve(
        scalar_rows(name, upper),
        PARAMS,
        jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert block.status is scalar.status is jcp.Status.OPTIMAL
    assert block.objective == pytest.approx(scalar.objective, rel=1e-5, abs=1e-6)
    assert block.statistics.eval.calls > 0


def test_a_block_with_integer_variables() -> None:
    """Checks that a block over integer variables solves, with the right answer."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0, upper=3, shape=(WIDTH,), domain=Domain.INTEGER),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: x[:-1] * x[1:], upper=5.0),),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.x is not None
    point = np.asarray(solution.x)
    assert np.all(point[:-1] * point[1:] <= 5.0 + 1e-6)
    assert solution.objective == pytest.approx(-12.0)


def build_scip(name: str, compiled: jcp.CompiledProblem) -> Scip:
    """Returns the SCIP problem of a compiled problem."""
    from jaxcip._solve import apply_options, build_scip

    scip = build_scip(compiled, PARAMS)
    apply_options(
        scip, jcp.SolveOptions(verbosity=jcp.Verbosity.QUIET, time_limit=60.0)
    )
    return scip


@pytest.mark.parametrize("name", ["neighbours", "mixed"])
def test_a_copy_of_scip_gets_its_own_state(name: str) -> None:
    """Checks that a copy made by SCIP solves like the original and keeps the states apart."""
    compiled = compiled_block(name, upper=6.0)
    scip = build_scip(name, compiled)
    copied_bound, _ = scip.solve_copy()
    scip.solve()
    assert scip.status is jcp.Status.OPTIMAL
    assert copied_bound == pytest.approx(scip.primal_bound, rel=1e-5, abs=1e-6)


def test_solves_in_threads_agree_with_sequential_solves() -> None:
    """Checks that solves of one compiled problem in different threads share no inputs."""
    compiled = compiled_block("scaled", upper=1.5)
    scales = [0.4, 0.9, 1.3, 1.8]
    options = jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET)

    def solve(scale: float) -> float:
        return jcp.solve(compiled, {"scale": jnp.array([scale])}, options).objective

    sequential = [solve(scale) for scale in scales]
    with ThreadPoolExecutor(len(scales)) as pool:
        threaded = list(pool.map(solve, scales * 2))
    assert len(set(np.round(sequential, 6))) > 1
    np.testing.assert_allclose(threaded, sequential * 2, rtol=1e-9)


def test_hessian_of_an_entry() -> None:
    """Checks SCIP's Hessian-vector product of an entry against JAX."""
    term, compiled = block_of("mixed")
    scip = build_scip("mixed", compiled)
    variables, params = flat_inputs(term, compiled)
    function = block_function(term.term, term.signature)
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    state = BlockStateRef(handle, params, variables, jcp.HandlerOptions())
    point = np.linspace(0.7, 1.9, WIDTH)
    direction = np.linspace(-0.5, 1.0, WIDTH)
    hessian = np.asarray(jax.jacfwd(jax.jacfwd(function))(point, params))
    for entry in range(handle.num_entries):
        product = scip.hessian_direction(
            BlockTerm(state, entry, 1.0), jcp.HandlerOptions(), point, direction
        )
        start, stop = handle.pointers[entry], handle.pointers[entry + 1]
        expected = (hessian[entry] @ direction)[handle.columns[start:stop]]
        assert product is not None
        np.testing.assert_allclose(product, expected, rtol=1e-9, atol=1e-9)
    with pytest.raises(IndexError, match="has 3 entries"):
        scip.hessian_direction(
            BlockTerm(state, 7, 1.0), jcp.HandlerOptions(), point, direction
        )


def test_the_same_entry_twice_is_merged() -> None:
    """Checks that SCIP orders and merges equal entries of a block when it simplifies."""
    term, compiled = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    variables, params = flat_inputs(term, compiled)
    scip = Scip("merge")
    for _ in range(WIDTH):
        scip.add_variable(0.5, 2.5, 0.0, Domain.CONTINUOUS)
    scip.set_parameter("display/verblevel", 0)
    state = BlockStateRef(handle, params, variables, jcp.HandlerOptions())
    other = BlockStateRef(handle, params, variables, jcp.HandlerOptions())
    terms = [
        BlockTerm(state, 0, 1.0),
        BlockTerm(state, 1, 1.0),
        BlockTerm(state, 0, 1.0),
        BlockTerm(other, 0, 1.0),
    ]
    scip.add_jax_constraint(
        terms,
        np.zeros(0, np.intc),
        np.zeros(0),
        0.0,
        -math.inf,
        20.0,
        jcp.HandlerOptions(),
    )
    scip.solve()
    assert scip.status is jcp.Status.OPTIMAL


def test_a_missing_entry_is_rejected() -> None:
    """Checks that an expression cannot be made for an entry that the block lacks."""
    term, compiled = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    variables, params = flat_inputs(term, compiled)
    scip = Scip("missing")
    for _ in range(WIDTH):
        scip.add_variable(0.5, 2.5, 0.0, Domain.CONTINUOUS)
    state = BlockStateRef(handle, params, variables, jcp.HandlerOptions())
    with pytest.raises(IndexError, match="has 5 entries"):
        scip.add_jax_constraint(
            [BlockTerm(state, 5, 1.0)],
            np.zeros(0, np.intc),
            np.zeros(0),
            0.0,
            -math.inf,
            1.0,
            jcp.HandlerOptions(),
        )
    with pytest.raises(ValueError, match="params has"):
        BlockStateRef(handle, np.zeros(3), variables, jcp.HandlerOptions())


def test_detect_pattern_of_a_function() -> None:
    """Checks the pattern of a function that is not a compiled term."""
    term, _ = block_of("neighbours")
    pattern = detect_pattern(block_function(term.term, term.signature), term.signature)
    assert pattern.nnz == 10
    np.testing.assert_array_equal(pattern.rows, np.repeat(np.arange(5), 2))


def block_scip(
    function: Callable[[jax.Array, Any], jax.Array],
    options: jcp.HandlerOptions = jcp.HandlerOptions(),
) -> tuple[Scip, BlockStateRef]:
    """Returns a SCIP problem of two variables and the state of the block of a function."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-10.0, upper=10.0, shape=(2,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(jcp.Constraint(function, upper=1.0),),
    )
    compiled = jcp.compile(problem, None, jcp.CompileOptions(disk_cache=False))
    [term] = compiled.constraints[0].terms
    assert isinstance(term.handle, BlockHandle)
    variables, params = block_inputs(
        term, compiled.constraints[0], compiled.variables, []
    )
    scip = Scip("block")
    for _ in range(2):
        scip.add_variable(-10.0, 10.0, 0.0, Domain.CONTINUOUS)
    return scip, BlockStateRef(term.handle, params, variables, options)


@pytest.mark.parametrize(
    "function",
    [
        lambda x, p: jnp.stack([jnp.log(x[0]) * x[1], x[0] * x[1]]),
        lambda x, p: jnp.stack([jnp.sqrt(x[0]) * x[1], x[0] * x[1]]),
        lambda x, p: jnp.stack([x[0] ** 1.5 * x[1], x[0] * x[1]]),
    ],
    ids=["infinite_value", "infinite_gradient", "infinite_hessian"],
)
def test_non_finite_derivatives_give_no_hessian(
    function: Callable[[jax.Array, Any], jax.Array],
) -> None:
    """Checks that non-finite values and derivatives of an entry give no Hessian."""
    scip, state = block_scip(function)
    product = scip.hessian_direction(
        BlockTerm(state, 0, 1.0),
        jcp.HandlerOptions(),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
    )
    assert product is None


def test_hessian_unavailable_for_entries() -> None:
    """Checks that no Hessian comes without the program or with Hessians switched off."""
    function = lambda x, p: jnp.stack([x[0] * x[1], x[0] ** 2 * x[1]])  # noqa: E731
    point, direction = np.array([0.7, 1.3]), np.array([1.0, 0.0])
    scip, state = block_scip(function)
    assert (
        scip.hessian_direction(
            BlockTerm(state, 0, 1.0), jcp.HandlerOptions(), point, direction
        )
        is not None
    )
    scip, off = block_scip(function, jcp.HandlerOptions(hessian=False))
    assert (
        scip.hessian_direction(
            BlockTerm(off, 0, 1.0), jcp.HandlerOptions(), point, direction
        )
        is None
    )
    handle = state.handle
    without = BlockStateRef(
        BlockHandle(
            handle.num_inputs,
            handle.num_params,
            handle.pointers,
            handle.columns,
            handle.programs._replace(hessian_vector=None),
        ),
        np.zeros(1),
        state.inputs,
        jcp.HandlerOptions(),
    )
    assert (
        scip.hessian_direction(
            BlockTerm(without, 0, 1.0), jcp.HandlerOptions(), point, direction
        )
        is None
    )


def two_entries(
    function: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array, Any], jax.Array]:
    """Returns a function of the variables with a second entry next to the one given."""
    return lambda x, p: jnp.stack([function(x), jnp.exp(0.1 * x[1])])


def test_unbounded_variables_in_a_block() -> None:
    """Checks a block over variables that are bounded only by linear constraints.

    Until propagation bounds them, intervals, estimators, monotonicity and reverse
    propagation see infinite boxes.
    """
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-math.inf, upper=math.inf, shape=(2,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(
            jcp.Constraint(lambda x, p: x, lower=-3.0, upper=3.0),
            jcp.Constraint(
                lambda x, p: jnp.stack(
                    [(x[0] - 1) ** 2 * jnp.exp(x[1]), jnp.exp(0.1 * x[1])]
                ),
                upper=jnp.array([2.0, 100.0]),
            ),
        ),
    )
    options = jcp.SolveOptions(
        node_limit=10000,
        verbosity=jcp.Verbosity.QUIET,
        settings=(jcp.ScipSetting(jcp.ScipParameter.PRESOLVING_MAXROUNDS, 0),),
    )
    solution = jcp.solve(problem, options=options)
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.x is not None
    point = np.asarray(solution.x)
    assert (point[0] - 1) ** 2 * math.exp(point[1]) <= 2.0 + 1e-6


def test_free_variables_in_a_block() -> None:
    """Checks a block whose variables have no bounds at all."""
    problem = jcp.Problem(
        variables=jcp.Variable(shape=(2,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack(
                    [jnp.sqrt(1 + x[0] ** 2) * jnp.exp(x[1] ** 2), x[0] * x[1]]
                ),
                lower=jnp.array([-math.inf, 0.5]),
                upper=jnp.array([2.0, 4.0]),
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(node_limit=1000, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.nodes > 0


def test_overflowing_relaxations_in_a_block() -> None:
    """Checks a block whose intervals and estimators overflow on its box."""
    problem = jcp.Problem(
        variables=(
            jcp.Variable(lower=0.0, upper=800.0),
            jcp.Variable(lower=1.0, upper=2.0),
            jcp.Variable(),
        ),
        objective=lambda x, p: x[2],
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack(
                    [jnp.exp(x[0]) * jnp.exp(-x[0]) * x[1] - x[2], x[1] * x[1]]
                ),
                upper=jnp.array([0.0, 5.0]),
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(node_limit=1000, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.objective == pytest.approx(1.0, abs=1e-5)


def test_steep_estimators_in_a_block() -> None:
    """Checks a block whose estimators have coefficients too large for SCIP's LP.

    The minimum of `(x0 - 15)⁸ x1 - x0` is at `x0 = 15 + 8^(-1/7)` and `x1 = 1`.
    """
    problem = jcp.Problem(
        variables=(
            jcp.Variable(lower=0.0, upper=30.0),
            jcp.Variable(lower=1.0, upper=2.0),
            jcp.Variable(),
        ),
        objective=lambda x, p: x[2] - x[0],
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack([(x[0] - 15) ** 8 * x[1] - x[2], x[1] * x[1]]),
                upper=jnp.array([0.0, 5.0]),
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(node_limit=1000, verbosity=jcp.Verbosity.QUIET),
    )
    root = 8 ** (-1 / 7)
    assert solution.objective == pytest.approx(root**8 - 15 - root, 1e-6)


@pytest.mark.parametrize(
    "options",
    [
        jcp.HandlerOptions(caches=False),
        jcp.HandlerOptions(init_estimates=False),
        jcp.HandlerOptions(monotonicity=False),
        jcp.HandlerOptions(reverse_propagation=False),
        jcp.HandlerOptions(hessian=False),
        jcp.HandlerOptions(global_estimators=False),
        jcp.HandlerOptions(zero_copy=False),
    ],
    ids=lambda options: next(
        name
        for name, value in options._asdict().items()
        if value != getattr(jcp.HandlerOptions(), name)
    ),
)
def test_switches_of_the_handler_in_a_block(options: jcp.HandlerOptions) -> None:
    """Checks that a block still solves with each switch of the handler changed."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(WIDTH,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: x[:-1] * x[1:], upper=4.0),),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
        compile_options=jcp.CompileOptions(handler=options, disk_cache=False),
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(-12.3, abs=1e-5)


def test_a_constant_entry_is_constant_in_every_direction() -> None:
    """Checks that SCIP learns from the gradient intervals that a constant entry has no slope."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(2,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack([x[0] * x[1], jnp.asarray(3.0)]), upper=4.0
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(-(2.5 + 4.0 / 2.5), abs=1e-5)


def test_statistics_count_direct_calls() -> None:
    """Checks that direct calls of a handle count as runs and that the counters reset."""
    term, compiled = block_of("neighbours")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    _, params = flat_inputs(term, compiled)
    handle.reset_statistics()
    handle.value_and_jacobian(np.ones(WIDTH), params)
    assert handle.statistics().eval.runs == 1
    handle.reset_statistics()
    assert handle.statistics().eval.runs == 0


@pytest.mark.parametrize("overestimate", [False, True])
def test_estimators_of_entries(overestimate: bool) -> None:
    """Checks that the estimator of an entry bounds it on the box, sampled densely."""
    term, compiled = block_of("mixed")
    handle = term.handle
    assert isinstance(handle, BlockHandle)
    function = jax.jit(jax.vmap(block_function(term.term, term.signature), (0, None)))
    variables, params = flat_inputs(term, compiled)
    rng = np.random.default_rng(3)
    point, lower, upper = random_box(11, len(variables))
    samples = rng.uniform(lower, upper, (300, len(variables)))
    values = np.asarray(function(samples, params))
    for entry in range(handle.num_entries):
        estimator = handle.estimator(entry, point, lower, upper, params, overestimate)
        assert estimator.success
        start, stop = handle.pointers[entry], handle.pointers[entry + 1]
        plane = estimator.constant + samples[:, handle.columns[start:stop]] @ (
            estimator.coefficients
        )
        slack = 1e-8 * (1 + np.abs(values[:, entry]))
        if overestimate:
            assert np.all(plane >= values[:, entry] - slack)
        else:
            assert np.all(plane <= values[:, entry] + slack)
    with pytest.raises(IndexError, match="has 3 entries"):
        handle.estimator(3, point, lower, upper, params, overestimate)


def test_unusable_estimators() -> None:
    """Checks that estimators with infinite or too steep coefficients are reported as failures."""
    scip, state = block_scip(
        lambda x, p: jnp.stack(
            [jnp.exp(x[0]) * jnp.exp(-x[0]) * x[1], (x[0] - 15) ** 8 * x[1]]
        )
    )
    del scip
    handle = state.handle
    params = np.zeros(1)
    lower, upper = np.array([0.0, 1.0]), np.array([800.0, 2.0])
    huge = handle.estimator(0, np.array([800.0, 1.0]), lower, upper, params, False)
    assert not huge.success
    steep = handle.estimator(
        1,
        np.array([30.0, 2.0]),
        np.array([0.0, 1.0]),
        np.array([30.0, 2.0]),
        params,
        False,
    )
    assert not steep.success
    fine = handle.estimator(
        1,
        np.array([15.0, 1.5]),
        np.array([14.0, 1.0]),
        np.array([16.0, 2.0]),
        params,
        False,
    )
    assert fine.success


def test_reverse_propagation_proves_infeasibility() -> None:
    """Checks a block whose entry is out of reach for its relaxation but not for its intervals."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.4, upper=0.6),
        objective=lambda x, p: x,
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack([x - x**2, jnp.exp(x)]),
                upper=jnp.array([0.1, 100.0]),
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.INFEASIBLE


def test_skipped_programs_of_a_block() -> None:
    """Checks that optional programs of a block that are switched off or fail are skipped."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-1.0, upper=1.0, shape=(2,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack([jnp.abs(x[0]) * x[1], x[1] ** 2]), upper=4.0
            ),
        ),
    )
    off = jcp.compile(
        problem,
        options=jcp.CompileOptions(
            handler=jcp.HandlerOptions(hessian=False, monotonicity=False),
            disk_cache=False,
        ),
    )
    assert any("switched off" in reason for reason in off.statistics.skipped)
    failing = jcp.compile(problem, options=jcp.CompileOptions(disk_cache=False))
    assert any("Error" in reason for reason in failing.statistics.skipped)
    assert jcp.solve(failing).status is jcp.Status.OPTIMAL


def test_a_block_loads_from_the_disk_cache() -> None:
    """Checks that a block compiled twice with the disk cache has the same pattern the second time."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(WIDTH,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: x[:-1] * x[1:] * 1.5, upper=4.0),),
    )
    compiled_terms.clear()
    first = jcp.compile(problem, options=jcp.CompileOptions(disk_cache=True))
    compiled_terms.clear()
    second = jcp.compile(problem, options=jcp.CompileOptions(disk_cache=True))
    assert second.statistics.loaded > 0
    [one] = first.constraints[0].terms
    [two] = second.constraints[0].terms
    assert isinstance(one.handle, BlockHandle) and isinstance(two.handle, BlockHandle)
    np.testing.assert_array_equal(one.handle.pointers, two.handle.pointers)
    np.testing.assert_array_equal(one.handle.columns, two.handle.columns)


def test_a_required_program_of_a_block_may_fail() -> None:
    """Checks that a block without relaxation rules cannot be compiled."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-1.0, upper=1.0, shape=(2,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.stack([jnp.where(x[0] > 0, x[0], x[1] ** 2), x[1]]),
                upper=4.0,
            ),
        ),
    )
    with pytest.raises(Exception, match="relaxation rule"):
        jcp.compile(problem, options=jcp.CompileOptions(disk_cache=False))


def chain_problem(count: int, domain: Domain = Domain.CONTINUOUS) -> jcp.Problem:
    """Returns a chain of products and exponentials over `count` variables."""
    return jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=3.0, shape=(count,), domain=domain),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(
            jcp.Constraint(
                lambda x, p: x[:-1] * x[1:] + jnp.exp(0.3 * x[1:]), upper=7.0
            ),
        ),
    )


def solve_chain(
    count: int, prefetch: bool, domain: Domain = Domain.CONTINUOUS
) -> jcp.Solution:
    """Solves the chain problem with prefetch switched on or off."""
    return jcp.solve(
        chain_problem(count, domain),
        options=jcp.SolveOptions(time_limit=120.0, verbosity=jcp.Verbosity.QUIET),
        compile_options=jcp.CompileOptions(
            handler=jcp.HandlerOptions(prefetch=prefetch), disk_cache=False
        ),
    )


def test_prefetch_answers_the_other_entries_of_a_block() -> None:
    """Checks that one run answers the callbacks of all entries at the same point or box."""
    with_prefetch = solve_chain(16, True)
    without = solve_chain(16, False)
    assert with_prefetch.status is without.status is jcp.Status.OPTIMAL
    assert with_prefetch.objective == pytest.approx(without.objective, abs=1e-6)
    for callback, factor in (("eval", 2), ("inteval", 3)):
        runs = getattr(with_prefetch.statistics, callback).runs
        assert runs * factor < getattr(without.statistics, callback).runs


def test_prefetch_keeps_estimators_correct_off_the_lp_solution() -> None:
    """Checks that integer rounding, which separates at other points, gives the same answer."""
    with_prefetch = solve_chain(8, True, Domain.INTEGER)
    without = solve_chain(8, False, Domain.INTEGER)
    assert with_prefetch.status is without.status is jcp.Status.OPTIMAL
    assert with_prefetch.objective == pytest.approx(without.objective, abs=1e-6)


def test_prefetch_with_a_fixed_variable() -> None:
    """Checks a block over a variable that presolve fixes, which leaves a child that is no variable."""
    problem = jcp.Problem(
        variables=(
            jcp.Variable(lower=1.0, upper=1.0),
            jcp.Variable(lower=0.5, upper=3.0, shape=(4,)),
        ),
        objective=lambda x, p: -jnp.sum(x[1]),
        constraints=(
            jcp.Constraint(
                lambda x, p: jnp.concatenate([x[0] * x[1][:1], x[1][:-1] * x[1][1:]]),
                upper=5.0,
            ),
        ),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.OPTIMAL
