"""Tests of the disk cache for the affine remainder of a block."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from test_affine import affine_cases, dense_matrix, problem_of

import jaxcip as jcp
from jaxcip._cache import AFFINE_SUFFIX

PARAMS = jnp.array([0.5, -1.5, 2.0, 0.25, 3.0])
"""Parameter values of the test problems."""


def test_warm_build_skips_detection_and_lowering(cache_directory: Path) -> None:
    """Checks that a second build loads the pattern and the program of the remainder."""
    problem = problem_of(affine_cases()["parameter_matrix"])
    options = jcp.CompileOptions(threads=1)
    first = jcp.compile(problem, PARAMS, options)
    jcp.clear_caches()
    second = jcp.compile(problem, PARAMS, options)
    assert first.statistics.lower_seconds > 0.0
    assert second.statistics.lower_seconds == 0.0
    assert second.statistics.load_seconds > 0.0
    for compiled in (first, second):
        constants, matrix = dense_matrix(compiled.constraints[0], compiled, PARAMS)
        np.testing.assert_allclose(constants, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        dense_matrix(first.constraints[0], first, PARAMS)[1],
        dense_matrix(second.constraints[0], second, PARAMS)[1],
    )


def test_remainder_without_parameters_is_stored_with_its_values() -> None:
    """Checks that a remainder that reads no parameter needs no program, even when loaded."""
    problem = problem_of(lambda x, p: x.vector[:-1] - 2 * x.vector[1:] + 3.0)
    options = jcp.CompileOptions(threads=1)
    for _ in range(2):
        jcp.clear_caches()
        compiled = jcp.compile(problem, PARAMS, options)
        affine = compiled.constraints[0].affine
        assert affine is not None and affine.program is None
        constants, matrix = dense_matrix(compiled.constraints[0], compiled, PARAMS)
        np.testing.assert_allclose(constants, 3.0)
        assert matrix.shape == (4, 11)


def test_damaged_cache_files_are_rebuilt(cache_directory: Path) -> None:
    """Checks that an unreadable file of entries is removed and the remainder is built again."""
    problem = problem_of(affine_cases()["gather"])
    options = jcp.CompileOptions(threads=1)
    before = set(cache_directory.glob(f"*{AFFINE_SUFFIX}"))
    reference = jcp.compile(problem, PARAMS, options)
    (damaged,) = set(cache_directory.glob(f"*{AFFINE_SUFFIX}")) - before
    damaged.write_bytes(b"not an archive")
    jcp.clear_caches()
    rebuilt = jcp.compile(problem, PARAMS, options)
    assert damaged.read_bytes() != b"not an archive"
    np.testing.assert_allclose(
        dense_matrix(rebuilt.constraints[0], rebuilt, PARAMS)[1],
        dense_matrix(reference.constraints[0], reference, PARAMS)[1],
    )
