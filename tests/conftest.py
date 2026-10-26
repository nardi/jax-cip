"""Shared fixtures: the PJRT plugin, a fresh disk cache and test functions."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcip as jcp
from jaxcip._cache import compile_program
from jaxcip._compile import CompiledTerm
from jaxcip._handler import FunctionHandle
from jaxcip._programs import OPTIONAL_PROGRAMS, lower, program_functions, program_shapes
from jaxcip._types import Program, Programs

DOMAIN_LOWER = -2.0
"""Lower bound of every coordinate of the test domain."""

DOMAIN_UPPER = 2.0
"""Upper bound of every coordinate of the test domain."""


def pytest_addoption(parser: pytest.Parser) -> None:
    """Adds the options that select and limit the MINLPLib instances to solve."""
    parser.addoption(
        "--minlplib",
        choices=("none", "quick", "full"),
        default="none",
        help="MINLPLib instances to solve: none, the quick list or every instance",
    )
    parser.addoption(
        "--minlplib-time-limit",
        type=float,
        default=60.0,
        help="time limit in seconds of each MINLPLib solve",
    )


@pytest.fixture(scope="session", autouse=True)
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[jcp.RuntimeInfo]:
    """Loads the PJRT plugin named by `JAXCIP_PJRT_PLUGIN`, with a fresh disk cache."""
    if not os.environ.get("JAXCIP_PJRT_PLUGIN"):
        pytest.fail("Set JAXCIP_PJRT_PLUGIN to the path of a PJRT CPU plugin.")
    previous = os.environ.get("JAXCIP_CACHE_DIR")
    os.environ["JAXCIP_CACHE_DIR"] = str(tmp_path_factory.mktemp("cache"))
    jax.config.update("jax_enable_x64", True)
    yield jcp.initialize()
    if previous is None:
        del os.environ["JAXCIP_CACHE_DIR"]
    else:
        os.environ["JAXCIP_CACHE_DIR"] = previous


@pytest.fixture
def cache_directory() -> Path:
    """Returns the disk cache directory of the session."""
    return Path(os.environ["JAXCIP_CACHE_DIR"])


def random_boxes(
    seed: int, count: int, size: int = 2
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns sub-boxes of the test domain with a point in each, some of them flat."""
    rng = np.random.default_rng(seed)
    boxes = []
    for index in range(count):
        first, second = rng.uniform(DOMAIN_LOWER, DOMAIN_UPPER, (2, size))
        lower, upper = np.minimum(first, second), np.maximum(first, second)
        if index % 5 == 0:
            upper[0] = lower[0]
        boxes.append((rng.uniform(lower, upper), lower, upper))
    return boxes


def six_hump_camel(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns the six-hump camel function, which has a global minimum of about −1.0316."""
    first, second = x[0], x[1]
    return (
        (4 - 2.1 * first**2 + first**4 / 3) * first**2
        + first * second
        + (-4 + 4 * second**2) * second**2
    )


def newton_sqrt(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns the square root of `x[0] + 3` after five Newton steps from 2."""
    target = x[0] + 3
    return jax.lax.fori_loop(
        0, 5, lambda _, guess: (guess + target / guess) / 2, jnp.asarray(2.0, x.dtype)
    )


TEST_FUNCTIONS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "product": lambda x, p: x[0] * x[1],
    "exp_minus_square": lambda x, p: jnp.exp(x[0]) - x[1] ** 2,
    "log_times": lambda x, p: jnp.log(x[0] + 3) * x[1],
    "six_hump_camel": six_hump_camel,
    "odd_power": lambda x, p: x[0] ** 3 - 2 * x[0],
    "reciprocal_and_sqrt": lambda x, p: 1 / (x[0] + 3) + jnp.sqrt(x[1] + 3),
    "dot_times": lambda x, p: jnp.dot(jnp.array([1.0, -2.0]), x) * x[0],
    "sum_exp": lambda x, p: jnp.sum(jnp.exp(x)),
    "quotient": lambda x, p: x[0] / (x[1] + 4),
    "negative_power": lambda x, p: (x[0] + 3) ** -2,
    "newton_sqrt": newton_sqrt,
    "jitted_inner": lambda x, p: jax.jit(lambda y: y[0] * jnp.exp(y[1]))(x),
    "fractional_power": lambda x, p: (x[1] + 2.5) ** 0.3 * x[0],
    "hypotenuse": lambda x, p: jnp.sqrt(x[0] ** 2 + 1) * x[1],
    "parametric": lambda x, p: p[0] * x[0] ** 2 + jnp.exp(p[1] * x[1]),
}
"""Functions of two inputs and two parameters on the test domain."""

PARAMS = np.array([1.5, -0.5])
"""Parameters of the test functions."""


def build_handle(
    function: Callable[[jax.Array, jax.Array], jax.Array],
    num_inputs: int = 2,
    num_params: int = 2,
    relaxation: jcp.RelaxationOptions = jcp.RelaxationOptions(),
    optional: bool = True,
) -> FunctionHandle:
    """Compiles the programs of a function into a handle, without the jaxpr analysis."""
    executables = []
    for program, program_function in zip(
        Program, program_functions(function, relaxation), strict=True
    ):
        if program in OPTIONAL_PROGRAMS and not optional:
            executables.append(None)
            continue
        lowered = lower(
            program_function, program_shapes(program, num_inputs, num_params)
        )
        executables.append(compile_program(lowered, disk_cache=True)[0])
    return FunctionHandle(num_inputs, num_params, Programs._make(executables))


def scalar_handle(term: CompiledTerm) -> FunctionHandle:
    """Returns the handle of a term that has scalar programs, which is every term but a block."""
    assert isinstance(term.handle, FunctionHandle)
    return term.handle
