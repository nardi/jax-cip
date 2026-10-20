"""Solves MINLPLib instances with JAX terms and compares them with SCIP's native solves.

The instances come from `scripts/fetch_minlplib.py`, in the directory named by
`JAXCIP_MINLPLIB_DIR`. `--minlplib=quick` solves the instances in `quick.txt`, which native
SCIP solves in about a second, and `--minlplib=full` solves every downloaded instance. With
`JAXCIP_MINLPLIB_RESULTS` set, each result is appended to that file as a line of JSON.
"""

import json
import math
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from convert import convert
from osil import UnsupportedInstance, read

import jaxcip as jcp
from jaxcip._problem import Scip, ScipError

QUICK = Path(__file__).with_name("quick.txt")
"""Names of the instances of the quick tier."""

RELATIVE_TOLERANCE = 1e-4
"""Allowed relative difference between optimal objectives."""

ABSOLUTE_TOLERANCE = 1e-6
"""Allowed absolute difference between optimal objectives near zero."""


OBJECTIVE_CONSTANT_VARIABLE = "objconstvar"
"""Variable that SCIP's OSiL reader fixes to the constant of the objective."""

NONLINEAR_OBJECTIVE_VARIABLE = "nlobjvar"
"""Variable that SCIP's OSiL reader bounds by the nonlinear part of the objective."""


class KnownSolution(NamedTuple):
    """What `minlplib.solu` says about an instance."""

    infeasible: bool
    optimum: float
    """The optimal objective, `nan` for an infeasible instance."""


def directory() -> Path:
    """Returns the directory of the downloaded instances."""
    value = os.environ.get("JAXCIP_MINLPLIB_DIR")
    if not value:
        pytest.skip("Set JAXCIP_MINLPLIB_DIR to the directory of fetch_minlplib.py.")
    return Path(value)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrizes the tests over the instances of the selected tier."""
    if "name" not in metafunc.fixturenames:
        return
    match metafunc.config.getoption("minlplib"):
        case "quick":
            names = QUICK.read_text().split()
        case "full":
            location = Path(os.environ.get("JAXCIP_MINLPLIB_DIR", "."))
            names = sorted(path.stem for path in (location / "osil").glob("*.osil"))
        case _:
            names = []
    metafunc.parametrize("name", names)


@pytest.fixture(scope="module")
def known_solutions() -> dict[str, KnownSolution]:
    """Returns the optimal and infeasible instances of `minlplib.solu`."""
    solutions = {}
    for line in (directory() / "minlplib.solu").read_text().splitlines():
        match line.split():
            case ["=opt=", name, value]:
                solutions[name] = KnownSolution(False, float(value))
            case ["=inf=", name, *_]:
                solutions[name] = KnownSolution(True, math.nan)
    return solutions


def close(first: float, second: float) -> bool:
    """Returns whether two objectives agree within the tolerances."""
    return math.isclose(
        first, second, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE
    )


def record(result: dict[str, object]) -> None:
    """Appends a result to the file in `JAXCIP_MINLPLIB_RESULTS`, if set."""
    if path := os.environ.get("JAXCIP_MINLPLIB_RESULTS"):
        with Path(path).open("a") as file:
            file.write(json.dumps(result) + "\n")


@pytest.mark.minlplib
def test_instance(
    name: str, known_solutions: dict[str, KnownSolution], request: pytest.FixtureRequest
) -> None:
    """Solves an instance natively and with JAX terms and checks both results.

    Optimal objectives agree with each other and with the known optimum, a primal solution
    is feasible for SCIP's own reading of the instance and the dual bound never passes the
    known optimum. A timeout where native SCIP solved is recorded, not failed. A SCIP error
    is recorded and fails.
    """
    path = directory() / "osil" / f"{name}.osil"
    try:
        instance = read(path)
        problem = convert(instance)
    except UnsupportedInstance as error:
        record({"name": name, "unsupported": str(error)})
        pytest.skip(f"Unsupported: {error}")
    options = jcp.SolveOptions(
        time_limit=request.config.getoption("minlplib_time_limit")
    )
    native = jcp.solve_file(path, options)
    native_result = {
        "status": native.status.name,
        "objective": native.objective,
        "dual_bound": native.dual_bound,
        "nodes": native.nodes,
        "seconds": native.solving_seconds,
    }
    try:
        solution = jcp.solve(problem, options=options)
    except ScipError as error:
        record({"name": name, "native": native_result, "error": str(error)})
        raise
    record(
        {
            "name": name,
            "native": native_result,
            "jaxcip": {
                "status": solution.status.name,
                "objective": solution.objective,
                "dual_bound": solution.dual_bound,
                "nodes": solution.nodes,
                "seconds": solution.solving_seconds,
                "build_seconds": solution.build.lower_seconds
                + solution.build.compile_seconds
                + solution.build.load_seconds,
            },
        }
    )
    known = known_solutions.get(name)
    minimize = instance.sense is jcp.Sense.MINIMIZE
    if known is not None and known.infeasible:
        assert solution.x is None
        return
    if solution.x is not None:
        reference = Scip.read(path)
        x = np.array(solution.x)
        values = dict(zip(instance.names, x, strict=True))
        objective = instance.objective
        constant = objective.lower
        linear = float(objective.linear_values @ x[objective.linear_indices])
        values[OBJECTIVE_CONSTANT_VARIABLE] = constant
        if problem.objective is not None:
            values[NONLINEAR_OBJECTIVE_VARIABLE] = (
                float(problem.objective(tuple(x), None)) - constant - linear
            )
        assert reference.check_solution(
            np.array([values[item] for item in reference.variable_names])
        )
    if native.status is jcp.Status.OPTIMAL and solution.status is jcp.Status.OPTIMAL:
        assert close(solution.objective, native.objective)
    if known is not None:
        if solution.status is jcp.Status.OPTIMAL:
            assert close(solution.objective, known.optimum)
        bound_gap = (solution.dual_bound - known.optimum) * (1 if minimize else -1)
        assert bound_gap <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(known.optimum)
