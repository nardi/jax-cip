"""Tests of the sparse affine part of a block, compared with dense derivatives."""

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

import jaxcip as jcp
from jaxcip._compile import Block, CompiledProblem, linear_rows


class Variables(NamedTuple):
    """Variables of the test problems, a vector and a matrix."""

    vector: Any
    matrix: Any


def affine_cases() -> dict[str, Callable[[Variables, jax.Array], jax.Array]]:
    """Returns affine functions of the variables and a parameter vector, by name."""
    index = np.array([4, 0, 0, 2])
    return {
        "differences": lambda x, p: x.vector[:-1] - 2 * x.vector[1:] + p[0],
        "sum": lambda x, p: jnp.sum(x.vector) * p[1] + jnp.sum(x.matrix),
        "matrix_product": lambda x, p: (
            jnp.arange(6.0).reshape(3, 2) @ x.vector[:2] + p[:3]
        ),
        "parameter_matrix": lambda x, p: jnp.outer(p, p)[:2, :3] @ x.matrix[:, 0],
        "gather": lambda x, p: x.vector[index] * p[:4] - x.matrix.reshape(-1)[:4],
        "cumulative": lambda x, p: (
            jnp.cumsum(x.vector) + jnp.flip(x.matrix[0, :2], 0)[0]
        ),
        "padded": lambda x, p: (
            jnp.pad(x.vector, (1, 2)) + jnp.concatenate([p[:3], x.vector])
        ),
        "transposed": lambda x, p: (x.matrix.T * p[:3]).ravel(),
        "constant": lambda x, p: jnp.full(3, 2.0) + p[:3],
    }


def problem_of(function: Callable[[Variables, jax.Array], jax.Array]) -> jcp.Problem:
    """Returns a problem with one constraint over the function."""
    variable = jcp.Variable(lower=-1.0, upper=2.0)
    return jcp.Problem(
        variables=Variables(
            vector=variable._replace(shape=(5,)), matrix=variable._replace(shape=(3, 2))
        ),
        objective=lambda x, p: jnp.sum(x.vector),
        constraints=(jcp.Constraint(function, lower=-100.0, upper=100.0),),
    )


def dense_matrix(
    block: Block, compiled: CompiledProblem, params: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the constants and the dense Jacobian of the affine part of a block."""
    rows = linear_rows(block, compiled.variables, compiled.params.leaves(params))
    matrix = sparse.csr_matrix(
        (rows.coefficients, rows.indices, rows.pointers),
        shape=(block.rows, compiled.variables.size),
    )
    return rows.constants, matrix.toarray()


@pytest.mark.parametrize("name", sorted(affine_cases()))
def test_sparse_rows_match_dense_jacobian(name: str) -> None:
    """Checks the constants and coefficients against `jax.jacfwd` of the function."""
    function = affine_cases()[name]
    params = jnp.array([0.5, -1.5, 2.0, 0.25, 3.0])
    compiled = jcp.compile(problem_of(function), params)
    block = compiled.constraints[0]
    constants, matrix = dense_matrix(block, compiled, params)

    def flat(values: jax.Array) -> jax.Array:
        variables = Variables(values[:5], values[5:].reshape(3, 2))
        return jnp.ravel(jnp.asarray(function(variables, params)))

    zeros = jnp.zeros(11)
    np.testing.assert_allclose(constants, flat(zeros), atol=1e-12)
    np.testing.assert_allclose(matrix, jax.jacfwd(flat)(zeros), atol=1e-12)


def test_zero_coefficients_are_dropped() -> None:
    """Checks that entries that vanish for the given parameters are not stored."""
    compiled = jcp.compile(
        problem_of(lambda x, p: p[0] * x.vector[:3] + p[1] * x.vector[2:]), jnp.zeros(5)
    )
    block = compiled.constraints[0]
    full = linear_rows(block, compiled.variables, [np.array([1.0, 2.0, 0, 0, 0])])
    assert full.coefficients.size == 6
    none = linear_rows(block, compiled.variables, [np.zeros(5)])
    assert none.coefficients.size == 0
    np.testing.assert_array_equal(none.pointers, np.zeros(4))
    half = linear_rows(block, compiled.variables, [np.array([0.0, 2.0, 0, 0, 0])])
    assert half.coefficients.size == 3
    assert sorted(half.indices) == [2, 3, 4]


def test_rows_are_ordered() -> None:
    """Checks that the pointers delimit the entries of each row, in row order."""
    compiled = jcp.compile(
        problem_of(lambda x, p: x.vector[::-1][:3] + x.matrix[:, 1]), jnp.zeros(5)
    )
    rows = linear_rows(compiled.constraints[0], compiled.variables, [np.zeros(5)])
    assert rows.pointers.tolist() == [0, 2, 4, 6]
    first, coefficients = rows.row(0)
    assert sorted(first.tolist()) == [4, 6]
    np.testing.assert_array_equal(coefficients, [1.0, 1.0])


LARGE_PROGRAM = textwrap.dedent(
    """
    import resource

    import jax.numpy as jnp
    import numpy as np

    import jaxcip as jcp
    from jaxcip._compile import linear_rows

    n = 10_000
    problem = jcp.Problem(
        variables=jcp.Variable(lower=0.0, upper=1.0, shape=(n,)),
        objective=lambda x, p: jnp.sum(x),
        constraints=(jcp.Constraint(lambda x, p: x[:-1] - 2 * x[1:] + p[0], upper=1.0),),
    )
    compiled = jcp.compile(problem, jnp.zeros(1))
    rows = linear_rows(compiled.constraints[0], compiled.variables, [np.zeros(1)])
    assert rows.coefficients.size == 2 * (n - 1)
    assert rows.pointers[-1] == 2 * (n - 1)
    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    """
)
"""A program that builds 10⁴ rows over 10⁴ variables and prints its peak memory in KiB."""


@pytest.mark.integration
def test_no_dense_jacobian_for_many_rows() -> None:
    """Checks that 10⁴ rows over 10⁴ variables never allocate a dense Jacobian.

    A dense Jacobian of that size takes 760 MiB. The process peaks at 480 to 850 MiB with
    the Jacobian stored sparsely, depending on the machine, most of it from JAX and SCIP, so
    a dense one would exceed the limit on any of them.
    """
    result = subprocess.run(
        [sys.executable, "-c", LARGE_PROGRAM],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    peak_kib = int(result.stdout.strip().splitlines()[-1])
    assert peak_kib < 1100 * 1024
