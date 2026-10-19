"""Tests of solving problems, compared with SCIP's native solves of the same problems."""

import math
from pathlib import Path
from typing import Any, Generic, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxcip as jcp

DATA = Path(__file__).parent / "data"
"""Directory of the native counterparts of the test problems."""

RELATIVE_TOLERANCE = 1e-5
"""Allowed relative difference to the native optimum, a few times SCIP's feasibility tolerance."""


def camel(x: jax.Array) -> jax.Array:
    """Returns the six-hump camel function of `x[0]` and `x[1]`."""
    first, second = x[0], x[1]
    return (
        (4 - 2.1 * first**2 + first**4 / 3) * first**2
        + first * second
        + (-4 + 4 * second**2) * second**2
    )


Leaf = TypeVar("Leaf")
"""Leaf type of a variable structure, a `Variable` in a problem and an array in a function."""


class MixedVariables(NamedTuple, Generic[Leaf]):
    """Variables of the synthetic MINLP."""

    x: Leaf
    y: Leaf
    z: Leaf


def minlp_objective(v: MixedVariables[jax.Array], params: None) -> jax.Array:
    """Returns the objective of the synthetic MINLP."""
    x0, x1 = v.x[0], v.x[1]
    return (
        (x0 - 2) ** 2
        + jnp.exp(0.5 * x1)
        - 3 * jnp.log(x0 + x1)
        + v.y[0] * x1
        - 1.5 * v.y[1]
        + 2 * v.z
    )


def newton_root(x: jax.Array) -> jax.Array:
    """Returns the square root of `x` after five Newton steps from 2."""
    return jax.lax.fori_loop(
        0, 5, lambda _, guess: (guess + x / guess) / 2, jnp.asarray(2.0, x.dtype)
    )


PROBLEMS: dict[str, jcp.Problem] = {
    "camel": jcp.Problem(
        variables=jcp.Variable(
            lower=jnp.array([-3.0, -2.0]), upper=jnp.array([3.0, 2.0])
        ),
        objective=lambda x, params: camel(x),
    ),
    "minlp": jcp.Problem(
        variables=MixedVariables(
            x=jcp.Variable(lower=0.1, upper=4.0, shape=(2,)),
            y=jcp.Variable(lower=0, upper=3, domain=jcp.Domain.INTEGER, shape=(2,)),
            z=jcp.Variable(domain=jcp.Domain.BINARY),
        ),
        objective=minlp_objective,
        constraints=(
            jcp.Constraint(lambda v, p: v.x[0] * v.x[1] + v.y[0] - v.y[1], lower=1.0),
            jcp.Constraint(lambda v, p: jnp.exp(v.x[0] / 2) + v.z * v.x[1], upper=4.0),
            jcp.Constraint(lambda v, p: jnp.sum(v.y) + v.z, upper=4.0),
            jcp.Constraint(lambda v, p: jnp.sum(v.x), upper=5.0),
        ),
    ),
    "newton": jcp.Problem(
        variables=jcp.Variable(
            lower=jnp.array([0.5, 0.0]), upper=jnp.array([4.0, 3.0])
        ),
        objective=lambda x, params: x[0] - 3 * x[1],
        constraints=(jcp.Constraint(lambda x, p: newton_root(x[0]) - x[1], lower=0.0),),
    ),
}
"""Test problems with a native counterpart in `DATA`."""

CONFIGURATIONS: dict[str, jcp.CompileOptions] = {
    "default": jcp.CompileOptions(),
    "single_term": jcp.CompileOptions(split_sums=False),
    "reverse": jcp.CompileOptions(
        relaxation=jcp.RelaxationOptions(mode=jcp.RelaxationMode.REVERSE)
    ),
    "alpha_bb": jcp.CompileOptions(relaxation=jcp.RelaxationOptions(alpha_bb=True)),
    "tighten_bounds": jcp.CompileOptions(
        relaxation=jcp.RelaxationOptions(tighten_bounds=True)
    ),
    "no_disk_cache": jcp.CompileOptions(disk_cache=False),
    **{
        f"no_{switch}": jcp.CompileOptions(
            handler=jcp.HandlerOptions()._replace(**{switch: False})
        )
        for switch in jcp.HandlerOptions._fields
    },
}
"""Compile options that each switch one feature off or on."""


@pytest.fixture(scope="module")
def native() -> dict[str, jcp.FileSolution]:
    """Returns SCIP's native solution of every test problem."""
    solutions = {name: jcp.solve_file(DATA / f"{name}.cip") for name in PROBLEMS}
    assert all(solution.status is jcp.Status.OPTIMAL for solution in solutions.values())
    return solutions


@pytest.mark.parametrize("configuration", sorted(CONFIGURATIONS))
@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_native_optimum(
    name: str, configuration: str, native: dict[str, jcp.FileSolution]
) -> None:
    """Checks that each configuration proves the native optimum."""
    solution = jcp.solve(
        PROBLEMS[name],
        options=jcp.SolveOptions(time_limit=120.0),
        compile_options=CONFIGURATIONS[configuration],
    )
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(
        native[name].objective, rel=RELATIVE_TOLERANCE, abs=RELATIVE_TOLERANCE
    )
    assert solution.dual_bound <= solution.objective + RELATIVE_TOLERANCE
    assert solution.gap <= 1e-4
    assert solution.nodes >= 0 and solution.solving_seconds >= 0


def test_solution_structure() -> None:
    """Checks that a solution has the structure of the variables, with rounded integers."""
    solution = jcp.solve(PROBLEMS["minlp"])
    assert isinstance(solution.x, MixedVariables)
    assert solution.x.x.shape == (2,) and solution.x.z.shape == ()
    np.testing.assert_array_equal(solution.x.y, np.round(solution.x.y))
    assert minlp_objective(solution.x, None) == pytest.approx(solution.objective, 1e-6)
    assert solution.statistics.eval.calls > 0


class Parameters(NamedTuple):
    """Parameters of the parametric test problem."""

    weight: jax.Array
    centre: jax.Array


def parametric_problem() -> jcp.Problem:
    """Returns `min weight · camel(x) + |x − centre|²` over an integer count and a box."""

    class Variables(NamedTuple, Generic[Leaf]):
        x: Leaf
        count: Leaf

    def objective(v: Variables[jax.Array], params: Parameters) -> jax.Array:
        return (
            params.weight * camel(v.x) + jnp.sum((v.x - params.centre) ** 2) + v.count
        )

    return jcp.Problem(
        variables=Variables(
            x=jcp.Variable(lower=-2.0, upper=2.0, shape=(2,)),
            count=jcp.Variable(lower=0, upper=3, domain=jcp.Domain.INTEGER),
        ),
        objective=objective,
        constraints=(
            jcp.Constraint(lambda v, p: jnp.sum(v.x**2), lower=0.04, upper=4.0),
            jcp.BatchedConstraint(
                lambda x, p: jnp.exp(x[0]) - p * x[1],
                inputs=np.array([[0, 1], [1, 0]]),
                params=jnp.array([1.0, 2.0]),
                upper=10.0,
            ),
        ),
    )


def test_compile_once_solve_many() -> None:
    """Checks that one compiled problem solves for several parameter values."""
    problem = parametric_problem()
    params = Parameters(weight=jnp.array(1.0), centre=jnp.zeros(2))
    compiled = jcp.compile(problem, params)
    first = jcp.solve(compiled, params)
    assert first.build == jcp.BuildStatistics()
    shifted = jcp.solve(compiled, params._replace(centre=jnp.array([1.0, -1.0])))
    assert first.status is shifted.status is jcp.Status.OPTIMAL
    assert first.objective != pytest.approx(shifted.objective)
    for solution in (first, shifted):
        assert solution.x is not None
        assert solution.x.count == 0
        assert 0.04 - 1e-6 <= float(jnp.sum(solution.x.x**2)) <= 4.0 + 1e-6


def test_terms_are_reused() -> None:
    """Checks that a second compile reuses the compiled terms of the first."""
    problem = parametric_problem()
    params = Parameters(weight=jnp.array(1.0), centre=jnp.zeros(2))
    jcp.compile(problem, params)
    again = jcp.compile(problem, params)
    assert again.statistics.reused >= 1
    assert again.statistics.lowered == again.statistics.compiled == 0


def test_disk_cache_after_clearing(cache_directory: Path) -> None:
    """Checks that a build after clearing the caches in memory loads from disk."""
    problem = PROBLEMS["newton"]
    jcp.compile(problem)
    jcp.clear_caches()
    statistics = jcp.compile(problem).statistics
    assert statistics.reused == statistics.lowered == statistics.compiled == 0
    assert statistics.loaded > 0
    assert any(cache_directory.glob("*.lowered"))


def test_maximize() -> None:
    """Checks maximizing the negated objective after minimizing the objective.

    The negated objective has the same terms with opposite signs, which must not share
    compiled programs.
    """
    problem = PROBLEMS["camel"]._replace(
        objective=lambda x, params: -camel(x), sense=jcp.Sense.MAXIMIZE
    )
    jcp.solve(PROBLEMS["camel"])
    solution = jcp.solve(problem)
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(1.0316284535, 1e-6)


def test_feasibility_problem() -> None:
    """Checks a problem without an objective and a vector constraint."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(3,)),
        constraints=(
            jcp.Constraint(
                lambda x, p: x**2, lower=jnp.array([1.0, 0.25, 0.0]), upper=2.0
            ),
            jcp.Constraint(lambda x, p: x[0] * x[1], lower=-math.inf, upper=math.inf),
        ),
    )
    solution = jcp.solve(problem)
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == 0.0
    assert solution.x is not None
    assert np.all(solution.x**2 >= np.array([1.0, 0.25, 0.0]) - 1e-6)


def test_linear_problem() -> None:
    """Checks a problem without nonlinear terms, which has no handles."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=10.0, shape=(2,)),
        objective=lambda x, params: jnp.dot(params, x) + 1.0,
        sense=jcp.Sense.MAXIMIZE,
        constraints=(jcp.Constraint(lambda x, p: x[0] + 2 * x[1], upper=8.0),),
    )
    solution = jcp.solve(problem, jnp.array([1.0, 1.0]))
    assert solution.objective == pytest.approx(9.0)
    assert solution.x is not None
    np.testing.assert_allclose(solution.x, [8.0, 0.0], atol=1e-9)
    assert solution.statistics.eval.calls == 0


def test_infeasible_problem() -> None:
    """Checks that an infeasible problem has no solution."""
    problem = PROBLEMS["camel"]._replace(
        constraints=(jcp.Constraint(lambda x, p: jnp.sum(x**2), upper=-1.0),)
    )
    solution = jcp.solve(problem)
    assert solution.status is jcp.Status.INFEASIBLE
    assert solution.x is None


def test_solve_options() -> None:
    """Checks limits, emphasis, verbosity and typed settings."""
    options = jcp.SolveOptions(
        time_limit=60.0,
        gap_limit=0.5,
        node_limit=1,
        emphasis=jcp.Emphasis.FEASIBILITY,
        verbosity=jcp.Verbosity.QUIET,
        settings=(
            jcp.ScipSetting(jcp.ScipParameter.NUMERICS_FEASTOL, 1e-7),
            jcp.ScipSetting(jcp.ScipParameter.LP_PRESOLVING, False),
            jcp.ScipSetting(jcp.ScipParameter.LIMITS_TOTALNODES, 10),
            jcp.ScipSetting(jcp.ScipParameter.LIMITS_MEMORY, 100000),
        ),
    )
    solution = jcp.solve(PROBLEMS["camel"], options=options)
    assert solution.nodes <= 1
    file_solution = jcp.solve_file(DATA / "camel.cip", options)
    assert file_solution.nodes <= 1


@pytest.mark.parametrize(
    "setting",
    [
        jcp.ScipSetting(jcp.ScipParameter.NUMERICS_FEASTOL, "small"),
        jcp.ScipSetting(jcp.ScipParameter.DISPLAY_VERBLEVEL, 1.5),
        jcp.ScipSetting(jcp.ScipParameter.DISPLAY_VERBLEVEL, True),
        jcp.ScipSetting(jcp.ScipParameter.LP_PRESOLVING, 1),
    ],
)
def test_setting_types(setting: jcp.ScipSetting) -> None:
    """Checks that settings of the wrong type are refused."""
    with pytest.raises(TypeError, match=setting.parameter.name):
        jcp.solve(PROBLEMS["newton"], options=jcp.SolveOptions(settings=(setting,)))


def test_file_without_solution(tmp_path: Path) -> None:
    """Checks a native solve of an infeasible file."""
    path = tmp_path / "infeasible.cip"
    path.write_text(
        "STATISTICS\n  Problem name     : infeasible\nOBJECTIVE\n"
        "  Sense            : minimize\nVARIABLES\n"
        "  [continuous] <x>: obj=1, original bounds=[0,1]\nCONSTRAINTS\n"
        "  [linear] <c>: <x>[C] >= 2;\nEND\n"
    )
    solution = jcp.solve_file(path)
    assert solution.status is jcp.Status.INFEASIBLE
    assert solution.x is None


def test_errors() -> None:
    """Checks the errors of invalid problems and parameters."""
    box = jcp.Variable(lower=0.0, upper=1.0, shape=(2,))
    with pytest.raises(TypeError, match="must be a Variable"):
        jcp.compile(jcp.Problem(variables=(box, 1.0)))
    with pytest.raises(ValueError, match="lower bound above"):
        jcp.compile(jcp.Problem(variables=jcp.Variable(lower=1.0, upper=0.0)))
    with pytest.raises(ValueError, match="must return a scalar"):
        jcp.compile(jcp.Problem(variables=box, objective=lambda x, p: x**2))
    batched = jcp.BatchedConstraint(lambda x, p: x[0] * x[1], inputs=np.array([[0, 1]]))
    for constraint, error, message in (
        (batched._replace(inputs=np.array([0, 1])), ValueError, "two axes"),
        (batched._replace(inputs=np.array([[0, 2]])), IndexError, "missing variables"),
        (batched._replace(params=jnp.ones(2)), ValueError, "needs 1 rows"),
        (batched._replace(params=jnp.ones(())), ValueError, "needs 1 rows"),
        (batched._replace(function=lambda x, p: x**2), ValueError, "return a scalar"),
    ):
        with pytest.raises(error, match=message):
            jcp.compile(jcp.Problem(variables=box, constraints=(constraint,)))
    compiled = jcp.compile(
        jcp.Problem(variables=box, objective=lambda x, p: jnp.exp(x[0] * p[0])),
        jnp.ones(1),
    )
    with pytest.raises(ValueError, match="structure"):
        jcp.solve(compiled, (jnp.ones(1),))
    with pytest.raises(ValueError, match="shape"):
        jcp.solve(compiled, jnp.ones(2))


def test_abstract_parameters() -> None:
    """Checks that a problem compiles for parameter shapes without values."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-1.0, upper=1.0),
        objective=lambda x, p: jnp.exp(p * x) - x,
    )
    compiled = jcp.compile(problem, jax.ShapeDtypeStruct((), jnp.float64))
    solution = jcp.solve(compiled, jnp.array(2.0))
    assert solution.objective == pytest.approx(
        math.exp(2 * -math.log(2) / 2) + math.log(2) / 2, 1e-5
    )


def test_skipped_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that optional programs that are switched off, too large or fail are skipped."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-1.0, upper=1.0, shape=(2,)),
        objective=lambda x, p: jnp.abs(x[0] - 0.25) + x[1] ** 2,
    )
    limited = jcp.compile(
        problem, options=jcp.CompileOptions(gradient_interval_limit=0)
    )
    reasons = " ".join(limited.statistics.skipped)
    assert "GRADIENT_INTERVAL" in reasons
    solution = jcp.solve(limited)
    assert solution.objective == pytest.approx(0.0, abs=1e-6)
    off = jcp.compile(
        problem,
        options=jcp.CompileOptions(
            handler=jcp.HandlerOptions(hessian=False, monotonicity=False)
        ),
    )
    assert any("switched off" in reason for reason in off.statistics.skipped)
    failing = jcp.compile(problem._replace(objective=lambda x, p: jnp.abs(x[0]) * x[1]))
    assert any("Error" in reason for reason in failing.statistics.skipped)


def test_required_program_failure() -> None:
    """Checks that a function without relaxation rules cannot be compiled."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-1.0, upper=1.0),
        objective=lambda x, p: jnp.where(x > 0, x, x**2),
    )
    with pytest.raises(Exception, match="relaxation rule"):
        jcp.compile(problem)


WEIGHTS = np.array([1.0, 2.0, 3.0])
"""Weights that a constraint closes over, which become parameters of its term."""


def test_uncached_terms() -> None:
    """Checks fresh terms built without the disk cache.

    The batched constraint has no affine part and the other constraint closes over an
    array constant.
    """
    jcp.clear_caches()
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.0, shape=(3,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(
            jcp.Constraint(lambda x, p: jnp.sum(WEIGHTS * jnp.log(x)), lower=0.0),
            jcp.BatchedConstraint(
                lambda x, p: x[0] * x[1], inputs=np.array([[0, 1], [1, 2]]), lower=1.0
            ),
        ),
    )
    compiled = jcp.compile(problem, options=jcp.CompileOptions(disk_cache=False))
    assert compiled.statistics.lowered > 0
    solution = jcp.solve(compiled)
    assert solution.status is jcp.Status.OPTIMAL
    x = solution.x
    assert x is not None
    assert float(jnp.sum(WEIGHTS * jnp.log(x))) >= -1e-6
    assert float(x[0] * x[1]) >= 1 - 1e-6 and float(x[1] * x[2]) >= 1 - 1e-6


def test_invalid_constraint() -> None:
    """Checks the error of a constraint of an unknown type."""
    constraint: Any = 1.0
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=1.0), constraints=(constraint,)
    )
    with pytest.raises(TypeError, match="must be a Constraint"):
        jcp.compile(problem)
