"""Run time of the programs, build time of problems and solve time of small problems.

Absolute times depend on the machine, so every benchmark records a `kind` and a
`reference` in `extra_info`. `compare.py` divides each time by the time of its
reference in the same run: a program by the value and gradient program of the
same function, and a build or a solve by a fixed NumPy computation that measures
the speed of the machine.
"""

from collections.abc import Callable, Iterator
from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import jaxcip as jcp
from jaxcip._cache import compile_program, enable_x64
from jaxcip._handler import FunctionHandle
from jaxcip._programs import lower, program_functions, program_shapes
from jaxcip._types import Program, Programs

NUM_POINTS = 256
"""Number of points that one timing of a program runs over."""

MACHINE_REFERENCE = "machine"
"""Reference of the benchmarks that are compared with the NumPy computation."""


@pytest.fixture(scope="session", autouse=True)
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[jcp.RuntimeInfo]:
    """Loads the PJRT plugin named by `JAXCIP_PJRT_PLUGIN`, with a fresh disk cache."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("JAXCIP_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        enable_x64()
        yield jcp.initialize()


def six_hump_camel(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns the six-hump camel function in two variables."""
    first, second = x[0], x[1]
    return (
        (4 - 2.1 * first**2 + first**4 / 3) * first**2
        + first * second
        + (-4 + 4 * second**2) * second**2
    )


def exponential_sum(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns a weighted sum of exponentials of products of neighbouring variables."""
    weights = jnp.linspace(-1.0, 1.0, x.size)
    return jnp.sum(weights * jnp.exp(x * jnp.roll(x, 1)))


def newton_iteration(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns the square root of `x[0] + 3` after 20 Newton steps in a `fori_loop`."""
    target = x[0] + 3
    return jax.lax.fori_loop(
        0, 20, lambda step, guess: (guess + target / guess) / 2, jnp.asarray(2.0)
    )


def rosenbrock(x: jax.Array, params: jax.Array) -> jax.Array:
    """Returns the Rosenbrock function."""
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


FUNCTIONS: dict[str, tuple[Callable[[jax.Array, jax.Array], jax.Array], int]] = {
    "six_hump_camel": (six_hump_camel, 2),
    "exponential_sum": (exponential_sum, 20),
    "newton_iteration": (newton_iteration, 1),
    "rosenbrock": (rosenbrock, 10),
}
"""Functions of the program benchmarks, with their number of inputs."""


@cache
def handle(name: str) -> FunctionHandle:
    """Compiles all programs of a benchmark function into a handle."""
    function, num_inputs = FUNCTIONS[name]
    executables = [
        compile_program(
            lower(program_function, program_shapes(program, num_inputs, 1)),
            disk_cache=True,
        )[0]
        for program, program_function in zip(
            Program, program_functions(function, jcp.RelaxationOptions()), strict=True
        )
    ]
    return FunctionHandle(num_inputs, 1, Programs._make(executables))


@pytest.mark.parametrize("program", list(Program), ids=lambda item: item.name.lower())
@pytest.mark.parametrize("name", sorted(FUNCTIONS))
def test_program_time(benchmark: BenchmarkFixture, name: str, program: Program) -> None:
    """Measures one program of a function over many points in a box."""
    function_handle = handle(name)
    num_inputs = function_handle.num_inputs
    points = np.random.default_rng(0).uniform(-1, 1, (NUM_POINTS, num_inputs))
    lower_bounds, upper_bounds = -np.ones(num_inputs), np.ones(num_inputs)
    benchmark.extra_info.update(
        kind="program", name=f"{name}.{program.name.lower()}", reference=name
    )
    benchmark(
        function_handle.time_program,
        program,
        points,
        lower_bounds,
        upper_bounds,
        np.zeros(function_handle.num_params),
    )


def test_machine_reference(benchmark: BenchmarkFixture) -> None:
    """Measures a fixed NumPy computation, which the build and solve times are divided by."""
    matrix = np.random.default_rng(0).normal(size=(200, 200))
    benchmark.extra_info.update(
        kind="reference", name=MACHINE_REFERENCE, reference=MACHINE_REFERENCE
    )
    benchmark(lambda: np.linalg.solve(matrix @ matrix.T + np.eye(200), matrix))


def camel() -> jcp.Problem:
    """Returns the six-hump camel problem on [-3, 3] × [-2, 2]."""
    return jcp.Problem(
        variables=jcp.Variable(
            lower=jnp.array([-3.0, -2.0]), upper=jnp.array([3.0, 2.0])
        ),
        objective=six_hump_camel,
    )


def constrained() -> jcp.Problem:
    """Returns the example of the README, a sum of quartics with a bilinear constraint."""
    return jcp.Problem(
        variables=jcp.Variable(lower=-5.0, upper=5.0, shape=(3,)),
        objective=lambda x, params: jnp.sum(x**4 - 16 * x**2 + 5 * x) / 2,
        constraints=(jcp.Constraint(lambda x, params: x[0] * x[1] + x[2], lower=1.0),),
    )


def mixed_integer() -> jcp.Problem:
    """Returns a small problem with integer variables and a nonconvex constraint."""
    return jcp.Problem(
        variables=jcp.Variable(
            lower=0.0, upper=4.0, domain=jcp.Domain.INTEGER, shape=(3,)
        ),
        objective=lambda x, params: -jnp.sum(jnp.log1p(x) * jnp.array([1.0, 2.0, 3.0])),
        constraints=(
            jcp.Constraint(lambda x, params: jnp.sum(x**2) - x[0] * x[2], upper=12.0),
        ),
    )


def batched() -> jcp.Problem:
    """Returns the example of the guide with rows, 100 rows over 20 variables."""
    rng = np.random.default_rng(0)
    count = 100
    row = jcp.rows(
        lambda x, weight: weight[0] * x[0] ** 2 + jnp.exp(weight[1] * x[1]),
        rng.integers(0, 20, size=(count, 2)),
        rng.uniform(0.5, 2.0, size=(count, 2)),
    )
    return jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=3.0, shape=(20,)),
        objective=lambda x, params: -jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, params: row(x), upper=10.0),),
    )


PROBLEMS: dict[str, Callable[[], jcp.Problem]] = {
    "camel": camel,
    "constrained": constrained,
    "mixed_integer": mixed_integer,
    "batched": batched,
}
"""Problems of the build and solve benchmarks."""


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_cold_build_time(benchmark: BenchmarkFixture, name: str) -> None:
    """Measures tracing, lowering and compiling a problem without any cache."""
    problem = PROBLEMS[name]()
    options = jcp.CompileOptions(disk_cache=False)
    benchmark.extra_info.update(
        kind="cold_build", name=name, reference=MACHINE_REFERENCE
    )
    benchmark.pedantic(
        jcp.compile,
        args=(problem, None, options),
        setup=jcp.clear_caches,
        rounds=3,
    )


def distinct_terms() -> jcp.Problem:
    """Returns a problem of 24 constraints, each with a nonlinear term of its own structure."""
    functions = [jnp.exp, jnp.sin, jnp.cos, jnp.tanh, jnp.log1p, jnp.sqrt]
    constraints = tuple(
        jcp.Constraint(
            lambda x, params, index=index, inner=functions[index % 6]: (
                x[index % 8] ** (2 + index // 6) * inner(x[(index + 1) % 8])
            ),
            upper=50.0,
        )
        for index in range(24)
    )
    return jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.0, shape=(8,)),
        objective=lambda x, params: jnp.sum(x),
        constraints=constraints,
    )


@pytest.mark.parametrize("threads", [1, None], ids=["one_thread", "all_threads"])
def test_cold_build_threads(benchmark: BenchmarkFixture, threads: int | None) -> None:
    """Measures building many different terms on one thread and on one thread per core."""
    problem = distinct_terms()
    options = jcp.CompileOptions(disk_cache=False, threads=threads)
    benchmark.extra_info.update(
        kind="cold_build",
        name=f"distinct_terms.{'one_thread' if threads == 1 else 'all_threads'}",
        reference=MACHINE_REFERENCE,
    )
    benchmark.pedantic(
        jcp.compile,
        args=(problem, None, options),
        setup=jcp.clear_caches,
        rounds=2,
    )


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_warm_build_time(benchmark: BenchmarkFixture, name: str) -> None:
    """Measures building a problem whose programs are in the disk cache."""
    problem = PROBLEMS[name]()
    jcp.compile(problem)
    benchmark.extra_info.update(
        kind="warm_build", name=name, reference=MACHINE_REFERENCE
    )
    benchmark.pedantic(jcp.compile, args=(problem,), setup=jcp.clear_caches, rounds=5)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_solve_time(benchmark: BenchmarkFixture, name: str) -> None:
    """Measures solving a compiled problem to optimality."""
    compiled = jcp.compile(PROBLEMS[name]())
    benchmark.extra_info.update(kind="solve", name=name, reference=MACHINE_REFERENCE)
    solution = benchmark.pedantic(jcp.solve, args=(compiled,), rounds=3)
    assert solution.status is jcp.Status.OPTIMAL
