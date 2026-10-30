"""Tests of saving a compiled problem and loading it in the same and in another process."""

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

import cyxla
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_affine import affine_cases, problem_of
from test_solve import Parameters, parametric_problem

import jaxcip as jcp
from jaxcip._block import BlockHandle

Leaf = TypeVar("Leaf")

OPTIONS = jcp.SolveOptions(time_limit=60.0, verbosity=jcp.Verbosity.QUIET)
"""Options of the solves, which have to finish."""

PARAMETERS = Parameters(weight=jnp.array(1.0), centre=jnp.array([0.5, -0.25]))
"""Parameter values of the parametric test problem."""


def parametric_template() -> tuple[object, Parameters]:
    """Returns the trees that `load` gets for the parametric test problem."""
    template = parametric_problem().variables
    return template, Parameters(weight=jnp.zeros(()), centre=jnp.zeros(2))


def test_loaded_problem_solves_like_the_original(tmp_path: Path) -> None:
    """Checks that a saved problem gives the same solution after loading."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    jcp.save(compiled, tmp_path / "problem.jcp")
    template, params = parametric_template()
    loaded = jcp.load(tmp_path / "problem.jcp", template, params)
    assert loaded.problem is None
    assert loaded.sense is compiled.sense
    assert loaded.options == compiled.options
    assert loaded.statistics.compiled == 0
    assert loaded.statistics.lowered == 0
    assert loaded.statistics.loaded > 0
    first = jcp.solve(compiled, PARAMETERS, options=OPTIONS)
    second = jcp.solve(loaded, PARAMETERS, options=OPTIONS)
    assert second.status is first.status is jcp.Status.OPTIMAL
    assert second.objective == pytest.approx(first.objective, abs=1e-5)
    assert second.x is not None and first.x is not None
    np.testing.assert_allclose(second.x.x, first.x.x, atol=1e-5)
    assert second.x.count == first.x.count


def test_a_loaded_problem_takes_other_parameters() -> None:
    """Checks that the parameters of a loaded problem are not fixed by the file."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    template, params = parametric_template()
    loaded = jcp.loads(jcp.dumps(compiled), template, params)
    shifted = PARAMETERS._replace(centre=jnp.array([1.0, -1.0]))
    first = jcp.solve(compiled, shifted, options=OPTIONS)
    second = jcp.solve(loaded, shifted, options=OPTIONS)
    assert second.status is first.status is jcp.Status.OPTIMAL
    assert second.objective == pytest.approx(first.objective, abs=1e-5)


@pytest.mark.parametrize("name", sorted(affine_cases()))
def test_affine_remainders_survive(name: str) -> None:
    """Checks the loaded affine part of a constraint, with and without parameters."""
    problem = problem_of(affine_cases()[name])
    params = jnp.array([0.5, -1.5, 2.0, 0.25, 3.0])
    compiled = jcp.compile(problem, params)
    loaded = jcp.loads(jcp.dumps(compiled), problem.variables, params)
    first = jcp.solve(compiled, params, options=OPTIONS)
    second = jcp.solve(loaded, params, options=OPTIONS)
    assert second.status is first.status
    assert second.objective == pytest.approx(first.objective, abs=1e-5)


def test_block_terms_keep_their_patterns() -> None:
    """Checks that a block of rows loads with its pattern and its executables."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.5, upper=2.5, shape=(6,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: x[:-1] * x[1:], lower=1.0),),
    )
    compiled = jcp.compile(problem)
    data = jcp.dumps(compiled)
    jcp.clear_caches()
    loaded = jcp.loads(data, problem.variables)
    ((before,), (after,)) = (
        [term.handle for term in problem.constraints[0].terms]
        for problem in (compiled, loaded)
    )
    assert isinstance(before, BlockHandle) and isinstance(after, BlockHandle)
    np.testing.assert_array_equal(after.pointers, before.pointers)
    np.testing.assert_array_equal(after.columns, before.columns)
    first = jcp.solve(compiled, options=OPTIONS)
    second = jcp.solve(loaded, options=OPTIONS)
    assert second.objective == pytest.approx(first.objective, abs=1e-5)


def test_loading_reuses_the_handles_of_this_process() -> None:
    """Checks that terms compiled in this process are not loaded from the file again."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    data = jcp.dumps(compiled)
    template, params = parametric_template()
    loaded = jcp.loads(data, template, params)
    assert loaded.statistics.reused > 0
    jcp.clear_caches()
    fresh = jcp.loads(data, template, params)
    assert fresh.statistics.reused == 0
    again = jcp.loads(data, template, params)
    # Only the programs of the affine remainders belong to a problem rather than a term.
    assert again.statistics.reused > 0
    assert again.statistics.loaded < fresh.statistics.loaded


def test_terms_that_repeat_are_stored_once() -> None:
    """Checks that two constraints over the same function share their programs."""
    function = jcp.Constraint(lambda x, p: jnp.sin(x[0]) * x[1], upper=1.0)
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=2.0, shape=(2,)),
        objective=lambda x, p: -jnp.sum(x),
        constraints=(function, function),
    )
    compiled = jcp.compile(problem)
    twice = jcp.dumps(compiled)
    once = jcp.dumps(jcp.compile(problem._replace(constraints=(function,))))
    assert len(twice) < len(once) + 4096
    jcp.clear_caches()
    loaded = jcp.loads(twice, problem.variables, warm_up=False, threads=1)
    assert (
        loaded.constraints[0].terms[0].handle is loaded.constraints[1].terms[0].handle
    )
    assert loaded.statistics.reused == 0
    first = jcp.solve(compiled, options=OPTIONS)
    second = jcp.solve(loaded, options=OPTIONS)
    assert second.objective == pytest.approx(first.objective, abs=1e-5)


def test_parameters_can_be_shape_structs() -> None:
    """Checks that the parameters tree of `load` takes `jax.ShapeDtypeStruct` leaves."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=1.0, shape=(2,)),
        objective=lambda x, p: jnp.sum((x - p) ** 4),
    )
    struct = jax.ShapeDtypeStruct((2,), jnp.float64)
    compiled = jcp.compile(problem, struct)
    loaded = jcp.loads(jcp.dumps(compiled), problem.variables, struct)
    solution = jcp.solve(loaded, jnp.array([0.25, 0.75]), options=OPTIONS)
    assert solution.status is jcp.Status.OPTIMAL
    assert solution.objective == pytest.approx(0.0, abs=1e-5)


def test_trees_are_checked_against_the_file() -> None:
    """Checks the errors for variables and parameters that do not match."""
    problem = jcp.Problem(
        variables={"a": jcp.Variable(lower=0.0, upper=1.0, shape=(2,))},
        objective=lambda x, p: jnp.sum(x["a"] ** 2) + jnp.sum(p["w"]),
    )
    params = {"w": np.ones(3)}
    data = jcp.dumps(jcp.compile(problem, params))
    good = {"a": np.zeros(2)}
    with pytest.raises(ValueError, match="variables have the leaf"):
        jcp.loads(data, {"b": np.zeros(2)}, params)
    with pytest.raises(ValueError, match="variables have the shape"):
        jcp.loads(data, {"a": np.zeros(3)}, params)
    with pytest.raises(ValueError, match="parameters have the shape"):
        jcp.loads(data, good, {"w": np.ones(4)})
    with pytest.raises(ValueError, match="parameters have the leaf"):
        jcp.loads(data, good, {"v": np.ones(3)})
    assert jcp.loads(data, good, params).params.fingerprint


def test_a_file_from_another_machine_is_compiled_again() -> None:
    """Checks the fallback for a fingerprint that the file has no executable for."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    data = jcp.dumps(compiled)
    template, params = parametric_template()
    other = cyxla.target_fingerprint()._replace(machine="other")
    jcp.clear_caches()
    with cyxla.override_target(other):
        with pytest.raises(cyxla.TargetMismatchError):
            jcp.loads(data, template, params, on_mismatch="raise")
        loaded = jcp.loads(data, template, params)
    assert loaded.statistics.compiled > 0
    assert loaded.statistics.loaded == 0
    first = jcp.solve(compiled, PARAMETERS, options=OPTIONS)
    second = jcp.solve(loaded, PARAMETERS, options=OPTIONS)
    assert second.objective == pytest.approx(first.objective, abs=1e-5)


def test_pickling_is_refused() -> None:
    """Checks that a compiled problem points to `dumps` instead of pickling."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    with pytest.raises(TypeError, match="jaxcip.dumps"):
        pickle.dumps(compiled)


def test_damaged_files_are_rejected() -> None:
    """Checks the errors for a file that is not a problem, cut off or of another kind."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=1.0), objective=lambda x, p: x**2
    )
    data = jcp.dumps(jcp.compile(problem))
    with pytest.raises(cyxla.BundleError, match="not a bundle"):
        jcp.loads(b"x" * 100, problem.variables)
    with pytest.raises(cyxla.BundleError, match="cut off"):
        jcp.loads(data[:-100], problem.variables)
    other = cyxla.BundleWriter("other.object", 1).to_bytes({})
    with pytest.raises(cyxla.BundleError, match="holds other.object"):
        jcp.loads(other, problem.variables)


class Pair(NamedTuple, Generic[Leaf]):
    """Variables of a problem whose loaded trees are arrays."""

    x: Leaf
    n: Leaf


def test_arrays_can_stand_in_for_variables() -> None:
    """Checks that the trees given to `load` need no `Variable` leaves."""
    problem = jcp.Problem(
        variables=Pair(
            x=jcp.Variable(lower=-2.0, upper=2.0, shape=(2,)),
            n=jcp.Variable(lower=0, upper=3, domain=jcp.Domain.INTEGER),
        ),
        objective=lambda v, p: jnp.sum(v.x**4 - 2 * v.x**2) + v.n * v.x[0],
    )
    compiled = jcp.compile(problem)
    loaded = jcp.loads(jcp.dumps(compiled), Pair(np.zeros(2), np.zeros(())))
    assert loaded.variables.domains == compiled.variables.domains
    first = jcp.solve(compiled, options=OPTIONS)
    second = jcp.solve(loaded, options=OPTIONS)
    assert second.objective == pytest.approx(first.objective, abs=1e-5)
    assert isinstance(second.x, Pair)


def test_another_process_loads_without_the_problem(tmp_path: Path) -> None:
    """Checks that a fresh process solves a saved problem without detection or tracing."""
    compiled = jcp.compile(parametric_problem(), PARAMETERS)
    path = tmp_path / "problem.jcp"
    jcp.save(compiled, path)
    expected = jcp.solve(compiled, PARAMETERS, options=OPTIONS).objective
    script = textwrap.dedent(
        f"""
        import sys

        import jax
        import jax.numpy as jnp

        import jaxcip as jcp

        from typing import NamedTuple

        class Parameters(NamedTuple):
            weight: jax.Array
            centre: jax.Array

        class Variables(NamedTuple):
            x: jax.Array
            count: jax.Array

        loaded = jcp.load(
            {str(path)!r},
            Variables(x=jnp.zeros(2), count=jnp.zeros(())),
            Parameters(jnp.zeros(()), jnp.zeros(2)),
        )
        params = Parameters(jnp.array(1.0), jnp.array([0.5, -0.25]))
        solution = jcp.solve(
            loaded, params, options=jcp.SolveOptions(verbosity=jcp.Verbosity.QUIET)
        )
        assert solution.status is jcp.Status.OPTIMAL, solution.status
        assert abs(solution.objective - {expected!r}) < 1e-5, solution.objective
        for module in ("asdex", "numba"):
            assert module not in sys.modules, module
        assert loaded.statistics.compiled == 0, loaded.statistics
        """
    )
    environment = {"CYXLA_CACHE_DIR": str(tmp_path / "empty-cache")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **environment},
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
