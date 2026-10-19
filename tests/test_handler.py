"""Tests of the handle programs, which must match the same JAX computations."""

from typing import Any

import jax
import jax.numpy as jnp
import jaxormick as jmk
import numpy as np
import pytest
from conftest import PARAMS, TEST_FUNCTIONS, build_handle, random_boxes

from jaxcip._handler import FunctionHandle, check_length, output_sizes
from jaxcip._types import Program, Programs

TOLERANCE = 1e-9
"""Allowed difference between PJRT and JAX results, which run the same XLA code."""


@pytest.fixture(scope="module")
def handles() -> dict[str, FunctionHandle]:
    """Returns a handle for every test function."""
    return {name: build_handle(function) for name, function in TEST_FUNCTIONS.items()}


@pytest.mark.parametrize("name", sorted(TEST_FUNCTIONS))
def test_programs_match_jax(name: str, handles: dict[str, FunctionHandle]) -> None:
    """Checks every program of a handle against the same JAX computation."""
    function = TEST_FUNCTIONS[name]
    handle = handles[name]
    gradient = jax.jit(jax.value_and_grad(function))
    hessian = jax.jit(jax.hessian(function))
    bounds = jax.jit(jmk.interval(function))
    gradient_bounds = jax.jit(jmk.interval(jax.grad(function)))
    convex = jax.jit(jmk.convex_side(function))
    concave = jax.jit(jmk.concave_side(function))
    for point, lower, upper in random_boxes(len(name), 10):
        expected = float(function(jnp.asarray(point), jnp.asarray(PARAMS)))
        assert handle.evaluate(point, PARAMS) == pytest.approx(expected, TOLERANCE)
        value, grad = handle.value_and_gradient(point, PARAMS)
        expected_value, expected_grad = gradient(point, PARAMS)
        assert value == pytest.approx(float(expected_value), TOLERANCE)
        np.testing.assert_allclose(grad, expected_grad, TOLERANCE, TOLERANCE)
        direction = point[::-1] - 0.25
        np.testing.assert_allclose(
            handle.hessian_vector(point, direction, PARAMS),
            np.asarray(hessian(point, PARAMS)) @ direction,
            1e-8,
            1e-8,
        )
        interval = handle.interval(lower, upper, params=PARAMS)
        expected_low, expected_high = bounds(lower, upper, PARAMS)
        assert interval.lower == pytest.approx(float(expected_low), TOLERANCE)
        assert interval.upper == pytest.approx(float(expected_high), TOLERANCE)
        gradient_interval = handle.gradient_interval(lower, upper, PARAMS)
        expected_low, expected_high = gradient_bounds(lower, upper, PARAMS)
        np.testing.assert_allclose(gradient_interval.lower, expected_low, TOLERANCE)
        np.testing.assert_allclose(gradient_interval.upper, expected_high, TOLERANCE)
        for overestimate, side in ((False, convex), (True, concave)):
            estimator = handle.estimator(point, lower, upper, PARAMS, overestimate)
            side_value, side_gradient = side(point, lower, upper, PARAMS)
            assert estimator.success
            np.testing.assert_allclose(
                estimator.coefficients, side_gradient, TOLERANCE, TOLERANCE
            )
            assert estimator.constant + estimator.coefficients @ point == pytest.approx(
                float(side_value), 1e-8, abs=1e-8
            )


@pytest.mark.parametrize("name", sorted(TEST_FUNCTIONS))
def test_estimators_are_valid(name: str, handles: dict[str, FunctionHandle]) -> None:
    """Checks that each estimator bounds the function on its box, sampled densely."""
    function = jax.jit(jax.vmap(TEST_FUNCTIONS[name], in_axes=(0, None)))
    handle = handles[name]
    rng = np.random.default_rng(1)
    for point, lower, upper in random_boxes(7, 10):
        samples = rng.uniform(lower, upper, (200, 2))
        values = np.asarray(function(samples, PARAMS))
        slack = 1e-8 * (1 + np.abs(values))
        for overestimate in (False, True):
            estimator = handle.estimator(point, lower, upper, PARAMS, overestimate)
            plane = estimator.constant + samples @ estimator.coefficients
            assert estimator.success
            if overestimate:
                assert np.all(plane >= values - slack)
            else:
                assert np.all(plane <= values + slack)


def test_estimator_clamps_the_reference_point(
    handles: dict[str, FunctionHandle],
) -> None:
    """Checks that a reference point outside the box gives the estimator at the nearest point."""
    handle = handles["product"]
    lower, upper = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    outside = handle.estimator(np.array([3.0, -3.0]), lower, upper, PARAMS, False)
    corner = handle.estimator(np.array([1.0, 0.0]), lower, upper, PARAMS, False)
    assert outside.constant == corner.constant
    np.testing.assert_array_equal(outside.coefficients, corner.coefficients)


def test_statistics_count_direct_calls(handles: dict[str, FunctionHandle]) -> None:
    """Checks that direct calls count as evaluations and that resetting clears them."""
    handle = handles["sum_exp"]
    handle.reset_statistics()
    handle.evaluate(np.zeros(2), PARAMS)
    handle.value_and_gradient(np.zeros(2), PARAMS)
    statistics = handle.statistics()
    assert statistics.eval.runs == 2
    assert statistics.eval.seconds > 0
    handle.reset_statistics()
    assert handle.statistics().eval.runs == 0


def test_time_program(handles: dict[str, FunctionHandle]) -> None:
    """Checks that timing a program returns a positive time per execution."""
    handle = handles["six_hump_camel"]
    points = np.random.default_rng(0).uniform(-1, 1, (16, 2))
    lower, upper = -np.ones(2), np.ones(2)
    for program in Program:
        seconds = handle.time_program(program, points, lower, upper, PARAMS, repeat=2)
        assert 0 < seconds < 1


def test_time_program_errors(handles: dict[str, FunctionHandle]) -> None:
    """Checks the errors of timing without points or repetitions, and of a missing program."""
    handle = handles["product"]
    box = (-np.ones(2), np.ones(2), PARAMS)
    with pytest.raises(ValueError, match="at least one point"):
        handle.time_program(Program.VALUE, np.zeros((0, 2)), *box)
    with pytest.raises(ValueError, match="at least one point"):
        handle.time_program(Program.VALUE, np.zeros((1, 2)), *box, repeat=0)
    with pytest.raises(ValueError, match="points"):
        handle.time_program(Program.VALUE, np.zeros((1, 3)), *box)
    without = build_handle(TEST_FUNCTIONS["product"], optional=False)
    with pytest.raises(ValueError, match="no hessian_vector program"):
        without.time_program(Program.HESSIAN_VECTOR, np.zeros((1, 2)), *box)


def test_missing_optional_programs() -> None:
    """Checks that a handle works without its optional programs and refuses to run them."""
    handle = build_handle(TEST_FUNCTIONS["product"], optional=False)
    assert handle.evaluate(np.array([2.0, 3.0]), PARAMS) == 6.0
    with pytest.raises(ValueError, match="no hessian_vector program"):
        handle.hessian_vector(np.zeros(2), np.ones(2), PARAMS)
    with pytest.raises(ValueError, match="no gradient_interval program"):
        handle.gradient_interval(-np.ones(2), np.ones(2), PARAMS)


def test_input_lengths_are_checked(handles: dict[str, FunctionHandle]) -> None:
    """Checks that inputs and parameters of the wrong length raise errors."""
    handle = handles["product"]
    with pytest.raises(ValueError, match="input"):
        handle.evaluate(np.zeros(3), PARAMS)
    with pytest.raises(ValueError, match="params"):
        handle.evaluate(np.zeros(2), np.zeros(5))
    with pytest.raises(ValueError, match="reference"):
        handle.estimator(np.zeros(1), -np.ones(2), np.ones(2), PARAMS, False)
    with pytest.raises(ValueError, match="values"):
        check_length("values", np.zeros(2), 3)


def test_handle_validation(handles: dict[str, FunctionHandle]) -> None:
    """Checks the errors of handles without inputs or with mismatching programs."""
    programs = handles["product"].programs
    with pytest.raises(ValueError, match="at least one input"):
        FunctionHandle(0, 1, programs)
    missing: Any = None
    with pytest.raises(ValueError, match="value program is required"):
        FunctionHandle(2, 2, programs._replace(value=missing))
    swapped = programs._replace(value=programs.value_and_gradient)
    with pytest.raises(ValueError, match="value program has 2 outputs"):
        FunctionHandle(2, 2, swapped)


def test_handle_properties(handles: dict[str, FunctionHandle]) -> None:
    """Checks the sizes and programs a handle reports."""
    handle = handles["product"]
    assert (handle.num_inputs, handle.num_params) == (2, 2)
    assert isinstance(handle.programs, Programs)
    assert output_sizes(Program.VALUE_AND_GRADIENT, 3) == (1, 3)
