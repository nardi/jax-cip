"""Tests of the SCIP problem wrapper."""

import math
from pathlib import Path

import numpy as np
import pytest
from conftest import TEST_FUNCTIONS, build_handle

import jaxcip as jcp
from jaxcip._problem import Scip, ScipError, forward_blas, parameter_table
from jaxcip._types import JaxTerm, ParameterType

NO_INDICES = np.zeros(0, np.intc)
NO_COEFFICIENTS = np.zeros(0)


def knapsack() -> Scip:
    """Returns `max x0 + 2 x1 + 3 x2` subject to `x0 + x1 + x2 ≤ 2` over binaries."""
    scip = Scip("knapsack")
    scip.set_sense(jcp.Sense.MAXIMIZE)
    for coefficient in (1.0, 2.0, 3.0):
        scip.add_variable(0.0, 1.0, coefficient, jcp.Domain.BINARY)
    scip.add_linear_constraint(np.arange(3, dtype=np.intc), np.ones(3), -math.inf, 2.0)
    scip.add_objective_offset(0.5)
    scip.set_parameter("display/verblevel", 0)
    return scip


def test_scip_version() -> None:
    """Checks the version of the linked SCIP."""
    assert jcp.scip_version() >= jcp.ScipVersion(10, 0, 0)


def test_forward_blas() -> None:
    """Checks that the OpenBLAS symbols are forwarded again without error."""
    assert forward_blas() > 0


def test_parameter_table_matches_the_enum() -> None:
    """Checks that the committed parameter enum lists every parameter of the linked SCIP."""
    table = parameter_table()
    assert {name for name, _, _ in table} == {
        parameter.path for parameter in jcp.ScipParameter
    }
    types = {name: parameter_type for name, parameter_type, _ in table}
    for parameter in jcp.ScipParameter:
        assert types[parameter.path] is parameter.parameter_type


def test_linear_problem(tmp_path: Path) -> None:
    """Checks a solve, its solution, a written copy and the solution checks."""
    scip = knapsack()
    assert scip.infinity >= 1e20
    assert scip.num_variables == 3
    assert scip.variable_names == ("x0", "x1", "x2")
    scip.solve()
    assert scip.status is jcp.Status.OPTIMAL
    assert scip.primal_bound == scip.dual_bound == 5.5
    assert scip.gap == 0.0
    assert scip.nodes >= 0
    assert scip.solving_time >= 0
    best = scip.best_solution()
    assert best is not None
    objective, values = best
    assert objective == 5.5
    np.testing.assert_array_equal(values, [0, 1, 1])
    assert scip.check_solution(np.array([0.0, 1.0, 1.0]))
    assert not scip.check_solution(np.array([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="3 variables, got 2"):
        scip.check_solution(np.zeros(2))
    path = tmp_path / "knapsack.cip"
    scip.write(path)
    copy = Scip.read(path)
    copy.set_parameter("display/verblevel", 0)
    copy.solve()
    assert copy.primal_bound == 5.5
    assert copy.num_variables == 3


def test_empty_problem() -> None:
    """Checks a problem without variables."""
    scip = Scip("empty")
    scip.set_parameter("display/verblevel", 0)
    assert scip.check_solution(np.zeros(0))
    scip.solve()
    assert scip.status is jcp.Status.OPTIMAL
    best = scip.best_solution()
    assert best is not None and best[1].shape == (0,)


def test_many_variables() -> None:
    """Checks that the variable list grows past its initial size."""
    scip = Scip("many")
    for index in range(40):
        assert scip.add_variable(0.0, 1.0, 1.0, jcp.Domain.CONTINUOUS) == index
    assert scip.variable_names[-1] == "x39"


def test_infeasible_problem() -> None:
    """Checks that an infeasible problem has no solution."""
    scip = knapsack()
    scip.add_linear_constraint(np.arange(3, dtype=np.intc), np.ones(3), 4.0, math.inf)
    scip.solve()
    assert scip.status is jcp.Status.INFEASIBLE
    assert scip.best_solution() is None


def test_errors(tmp_path: Path) -> None:
    """Checks the errors of bad variables, coefficients and files."""
    scip = knapsack()
    with pytest.raises(IndexError, match="no variable 3"):
        scip.add_linear_constraint(np.array([3], np.intc), np.ones(1), 0.0, 1.0)
    with pytest.raises(IndexError, match="no variable -1"):
        scip.variable_array(np.array([-1], np.intc))
    with pytest.raises(ValueError, match="one coefficient per variable"):
        scip.add_linear_constraint(np.array([0], np.intc), np.ones(2), 0.0, 1.0)
    with pytest.raises(ScipError, match="create a variable"):
        scip.add_variable(-5.0, 5.0, 0.0, jcp.Domain.BINARY)
    with pytest.raises(ScipError, match="read"):
        Scip.read(tmp_path / "missing.cip")


def test_variable_array() -> None:
    """Checks that variable arrays hold the addresses of the chosen variables."""
    scip = knapsack()
    addresses = scip.variable_array(np.array([2, 0], np.intc))
    assert addresses.shape == (2,) and addresses[0] != addresses[1]


@pytest.mark.parametrize(
    ("name", "parameter_type", "value"),
    [
        ("lp/presolving", ParameterType.BOOL, False),
        ("display/verblevel", ParameterType.INT, 2),
        ("limits/totalnodes", ParameterType.LONGINT, 1000),
        ("limits/time", ParameterType.REAL, 12.5),
        ("lp/initalgorithm", ParameterType.CHAR, "d"),
        ("visual/vbcfilename", ParameterType.STRING, "tree.vbc"),
    ],
)
def test_parameters(
    name: str, parameter_type: ParameterType, value: bool | int | float | str
) -> None:
    """Checks that parameters of every type are set and read back."""
    assert jcp.ScipParameter[name.upper().replace("/", "_")].parameter_type is (
        parameter_type
    )
    scip = Scip("parameters")
    scip.set_parameter(name, value)
    assert scip.get_parameter(name) == value


def test_parameter_errors() -> None:
    """Checks the errors of unknown parameters and bad values."""
    scip = Scip("parameters")
    with pytest.raises(KeyError, match="no parameter"):
        scip.set_parameter("no/such", 1)
    with pytest.raises(KeyError, match="no parameter"):
        scip.get_parameter("no/such")
    with pytest.raises(ValueError, match="one character"):
        scip.set_parameter("lp/initalgorithm", "dual")
    with pytest.raises(ScipError, match="set display/verblevel"):
        scip.set_parameter("display/verblevel", 99)


def test_jax_constraint_errors() -> None:
    """Checks the errors of terms whose sizes do not match their handle."""
    handle = build_handle(TEST_FUNCTIONS["product"])
    scip = knapsack()
    inputs = np.array([0, 1], np.intc)
    params = np.zeros(2)
    options = jcp.HandlerOptions()
    for term, message in (
        (JaxTerm(handle, params, inputs[:1], 1.0), "inputs"),
        (JaxTerm(handle, np.zeros(3), inputs, 1.0), "parameters"),
    ):
        with pytest.raises(ValueError, match=message):
            scip.add_jax_constraint(
                [term], NO_INDICES, NO_COEFFICIENTS, 0.0, 0.0, 1.0, options
            )
    with pytest.raises(ValueError, match="one coefficient per variable"):
        scip.add_jax_constraint(
            [JaxTerm(handle, params, inputs, 1.0)],
            inputs,
            NO_COEFFICIENTS,
            0.0,
            0.0,
            1.0,
            options,
        )
    with pytest.raises(IndexError, match="no variable 7"):
        scip.add_jax_constraint(
            [JaxTerm(handle, params, inputs, 1.0)],
            np.array([7], np.intc),
            np.ones(1),
            0.0,
            0.0,
            1.0,
            options,
        )


def test_written_jax_constraint(tmp_path: Path) -> None:
    """Checks that a JAX constraint is written as an opaque expression."""
    handle = build_handle(TEST_FUNCTIONS["product"])
    scip = knapsack()
    scip.add_jax_constraint(
        [JaxTerm(handle, np.zeros(2), np.array([0, 1], np.intc), 2.0)],
        np.array([2], np.intc),
        np.ones(1),
        0.5,
        -math.inf,
        3.0,
        jcp.HandlerOptions(),
    )
    path = tmp_path / "jax.cip"
    scip.write(path)
    assert "jax" in path.read_text()
