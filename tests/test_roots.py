"""Tests of the warm start of roots, from the rebuilt term to the expression handler."""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jaxormick._implicit import inverse

import jaxcip as jcp
from jaxcip._analysis import analyse
from jaxcip._compile import CompiledTerm, term_rows
from jaxcip._handler import FunctionHandle, output_sizes, program_slots
from jaxcip._problem import Scip
from jaxcip._programs import (
    program_functions,
    program_shapes,
    stateful_term_function,
    term_function,
    term_signature,
)
from jaxcip._roots import state_equations, state_size
from jaxcip._types import Domain, JaxTerm, Program, Programs

SIZE = 2
"""Number of entries of the roots in the test terms."""

COUPLING = np.array([[0.3, -0.2], [0.1, 0.25]])
"""Coupling of the entries of the root, which makes its equations a system."""


def solve_newton(
    function: Callable[[jax.Array], jax.Array], guess: jax.Array
) -> jax.Array:
    """Returns a root of `function` by Newton's method, which stops at a tolerance."""
    jacobian = jax.jacfwd(function)

    def keeps_going(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        steps, point = state
        return (steps < 60) & (jnp.max(jnp.abs(function(point))) > 1e-10)

    def step(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        steps, point = state
        return steps + 1, point - inverse(jacobian(point)) @ function(point)

    return lax.while_loop(keeps_going, step, (jnp.array(0), guess))[1]


def root_of(x: jax.Array, has_aux: bool = False) -> Any:
    """Returns the root `y` of `y³ + y + C y = x` found by Newton's method."""

    def residual(y: jax.Array) -> jax.Array:
        return y**3 + y + COUPLING @ y - x[:SIZE]

    def solve(function: Callable[..., Any], guess: jax.Array) -> Any:
        root = solve_newton(function, guess)
        return (root, jnp.zeros(())) if has_aux else root

    def tangent_solve(function: Callable[..., Any], value: jax.Array) -> jax.Array:
        return inverse(jax.jacfwd(function)(jnp.zeros(SIZE))) @ value

    return lax.custom_root(residual, jnp.zeros(SIZE), solve, tangent_solve, has_aux)


def objective(x: jax.Array, params: None) -> jax.Array:
    """Returns the squared distance of the root to a target, which is smooth in `x`."""
    return jnp.sum((root_of(x) - jnp.array([0.4, -0.7])) ** 2)


def compiled_root_term() -> tuple[jcp.CompiledProblem, CompiledTerm, np.ndarray]:
    """Returns the compiled root problem, its only term and the parameters of the term."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(SIZE,)),
        objective=objective,
    )
    compiled = jcp.compile(problem, options=jcp.CompileOptions(disk_cache=False))
    block = compiled.objective
    assert block is not None
    [term] = block.terms
    _, params = term_rows(term, block, compiled.variables, [])
    return compiled, term, params[0]


def reference_root(x: np.ndarray) -> np.ndarray:
    """Returns the root of the test equations by Newton's method in NumPy."""
    point = np.zeros(SIZE)
    for _ in range(50):
        residual = point**3 + point + COUPLING @ point - x
        point = point - np.linalg.solve(
            3 * np.diag(point**2) + np.eye(SIZE) + COUPLING, residual
        )
    return point


def test_state_size_counts_the_roots_at_the_top_level() -> None:
    """Checks which roots of a function take a warm start."""

    def analysed(function: Callable[[jax.Array], jax.Array]) -> Any:
        [term] = analyse(
            function, [jax.ShapeDtypeStruct((SIZE,), jnp.float64)], [], True
        ).nonlinear
        return term.jaxpr

    assert state_size(analysed(lambda x: jnp.sum(root_of(x) ** 2))) == SIZE
    assert (
        state_size(analysed(lambda x: jnp.sum(root_of(x) ** 2 + root_of(2 * x) ** 2)))
        == 2 * SIZE
    )
    assert state_size(analysed(lambda x: jnp.sum(jnp.sin(x) ** 2))) == 0
    nested = analysed(lambda x: jnp.sum(jax.jit(lambda y: root_of(y) ** 2)(x)))
    assert state_size(nested) == 0


def test_roots_with_auxiliary_data_take_no_warm_start() -> None:
    """Checks that a root whose solver returns auxiliary data keeps its own guess."""
    jaxpr = jax.make_jaxpr(lambda x: root_of(x, has_aux=True)[0])(jnp.zeros(SIZE)).jaxpr
    assert state_equations(jaxpr) == []


def test_stateful_function_matches_the_function() -> None:
    """Checks the value, gradient and roots for warm starts that differ in distance."""
    compiled, term, params = compiled_root_term()
    block = compiled.objective
    assert block is not None
    function = term_function(term.term, term.signature)
    stateful = stateful_term_function(term.term, term.signature)
    x = jnp.array([0.6, -0.3])
    value, gradient = jax.value_and_grad(function)(x, params)
    for state in (jnp.full(SIZE, jnp.nan), jnp.zeros(SIZE), jnp.array([4.0, -3.0])):
        (warm_value, roots), warm_gradient = jax.value_and_grad(stateful, has_aux=True)(
            x, params, state
        )
        np.testing.assert_allclose(warm_value, value, rtol=1e-12)
        np.testing.assert_allclose(warm_gradient, gradient, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(roots, reference_root(np.asarray(x)), atol=1e-12)


def test_nan_entries_fall_back_to_the_written_guess() -> None:
    """Checks that a warm start with a NaN entry mixes the guess of the term with its own."""
    _, term, params = compiled_root_term()
    stateful = stateful_term_function(term.term, term.signature)
    x = jnp.array([0.6, -0.3])
    mixed = jnp.array([jnp.nan, 0.5])
    value, roots = stateful(x, jnp.asarray(params), mixed)
    assert bool(jnp.all(jnp.isfinite(roots)))
    np.testing.assert_allclose(roots, reference_root(np.asarray(x)), atol=1e-12)
    assert math.isfinite(float(value))


def test_handler_carries_the_roots_between_runs() -> None:
    """Checks that each run starts from the roots of the run before."""
    _, term, params = compiled_root_term()
    handle = term.handle
    assert handle.state_size == SIZE
    x = np.array([0.6, -0.3])
    runs = handle.warm_runs(x, params, 3)
    expected = reference_root(x)
    for value, state in runs:
        np.testing.assert_allclose(state, expected, atol=1e-12)
        assert value == runs[0][0]
    other = np.array([0.55, -0.25])
    [(_, moved)] = handle.warm_runs(other, params, 1)
    np.testing.assert_allclose(moved, reference_root(other), atol=1e-12)


def test_the_result_does_not_depend_on_the_start() -> None:
    """Checks that the value and the roots are the same for any warm start."""
    _, term, params = compiled_root_term()
    handle = term.handle
    x = np.array([-0.8, 0.9])
    [(cold_value, cold_roots)] = handle.warm_runs(x, params, 1)
    for start in ([5.0, -5.0], [0.1, 0.1], [np.nan, 2.0]):
        [(value, roots)] = handle.warm_runs(x, params, 1, start=np.array(start))
        assert value == pytest.approx(cold_value, rel=1e-13)
        np.testing.assert_allclose(roots, cold_roots, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="start"):
        handle.warm_runs(x, params, 1, start=np.zeros(5))


def test_direct_calls_of_a_handle_with_roots() -> None:
    """Checks the direct value, gradient and Hessian-vector calls of a handle with roots."""
    _, term, params = compiled_root_term()
    function = term_function(term.term, term.signature)
    x, direction = np.array([0.6, -0.3]), np.array([0.5, 1.0])
    value, gradient = jax.value_and_grad(function)(jnp.asarray(x), jnp.asarray(params))
    hessian = jax.hessian(function)(jnp.asarray(x), jnp.asarray(params))
    assert term.handle.evaluate(x, params) == pytest.approx(float(value), rel=1e-10)
    got_value, got_gradient = term.handle.value_and_gradient(x, params)
    np.testing.assert_allclose(got_gradient, gradient, rtol=1e-8, atol=1e-10)
    product = term.handle.hessian_vector(x, direction, params)
    np.testing.assert_allclose(product, hessian @ direction, rtol=1e-6, atol=1e-8)
    assert got_value == pytest.approx(float(value), rel=1e-10)


def test_scip_hessian_with_roots() -> None:
    """Checks the Hessian-vector product of a root term inside SCIP's callbacks."""
    _, term, params = compiled_root_term()
    function = term_function(term.term, term.signature)
    x, direction = np.array([0.6, -0.3]), np.array([0.5, 1.0])
    scip = Scip("roots")
    for _ in range(SIZE):
        scip.add_variable(-2.0, 2.0, 0.0, Domain.CONTINUOUS)
    jax_term = JaxTerm(term.handle, params, np.arange(SIZE, dtype=np.intc), 1.0)
    product = scip.hessian_direction(jax_term, jcp.HandlerOptions(), x, direction)
    expected = jax.hessian(function)(jnp.asarray(x), jnp.asarray(params)) @ direction
    assert product is not None
    np.testing.assert_allclose(product, expected, rtol=1e-6, atol=1e-8)


def test_solving_with_a_warm_started_root() -> None:
    """Checks that a problem with a root reaches the `x` whose root is the target."""
    target = np.array([0.4, -0.7])
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(SIZE,)),
        objective=objective,
    )
    solution = jcp.solve(problem, options=jcp.SolveOptions(time_limit=60))
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.x is not None
    expected = target**3 + target + COUPLING @ target
    np.testing.assert_allclose(np.asarray(solution.x), expected, atol=1e-5)
    assert solution.statistics.eval.calls > 0


def test_handle_validation_with_roots() -> None:
    """Checks the errors of a warm start that the programs do not match."""
    _, term, _ = compiled_root_term()
    programs: Programs = term.handle.programs
    with pytest.raises(ValueError, match="negative"):
        FunctionHandle(SIZE, 1, programs, -1)
    with pytest.raises(ValueError, match="value_and_gradient program has 3 outputs"):
        FunctionHandle(SIZE, term.handle.num_params, programs, 0)
    assert output_sizes(Program.VALUE_AND_GRADIENT, SIZE, 4) == (1, SIZE, 4)
    assert program_slots(Program.CONVEX, 4) == program_slots(Program.CONVEX)
    assert len(program_slots(Program.HESSIAN_VECTOR, 4)) == 4


def test_programs_without_roots_are_unchanged() -> None:
    """Checks that terms without a root keep the programs and shapes they had."""
    x = jax.ShapeDtypeStruct((SIZE,), jnp.float64)
    [plain] = analyse(lambda v: jnp.sum(jnp.sin(v) ** 2), [x], [], True).nonlinear
    signature = term_signature(plain, ())
    assert signature.state_size == 0
    assert len(program_shapes(Program.VALUE_AND_GRADIENT, SIZE, 1)) == 2
    assert len(program_shapes(Program.VALUE_AND_GRADIENT, SIZE, 1, 3)) == 3
    functions = program_functions(
        term_function(plain, signature), jcp.RelaxationOptions()
    )
    assert len(functions) == len(Program)
