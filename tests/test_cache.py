"""Tests of the disk cache of lowered programs and executables."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcip._blocks import BlockPattern, LoweredTerm
from jaxcip._cache import (
    LOWERED_SUFFIX,
    BuildStatistics,
    compile_program,
    decode_lowered,
    encode_lowered,
    load_lowered,
    store_lowered,
)
from jaxcip._programs import LoweredProgram, LoweredPrograms, lower

PROGRAM = LoweredProgram(code=b"\x00code\xff", text="module @x {}")
"""A stand-in for a lowered program, enough for encoding."""

PROGRAMS = LoweredTerm(
    LoweredPrograms(PROGRAM, None, PROGRAM, PROGRAM, PROGRAM, None), None
)
"""A lowered term with both optional programs left out and no pattern."""

BLOCK = LoweredTerm(
    PROGRAMS.programs,
    BlockPattern(np.array([0, 2, 3], np.intc), np.array([0, 1, 1], np.intc)),
)
"""A lowered block term, which has a pattern of two entries."""


def test_encoding_round_trip() -> None:
    """Checks that decoding returns the encoded programs and pattern."""
    assert decode_lowered(encode_lowered(PROGRAMS)) == PROGRAMS
    decoded = decode_lowered(encode_lowered(BLOCK))
    assert decoded.programs == BLOCK.programs
    assert decoded.pattern is not None
    np.testing.assert_array_equal(decoded.pattern.pointers, [0, 2, 3])
    np.testing.assert_array_equal(decoded.pattern.columns, [0, 1, 1])


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"other", "Not a file"),
        (encode_lowered(PROGRAMS)[:-3], "Malformed"),
        (encode_lowered(PROGRAMS)[:30], "Malformed"),
        (encode_lowered(BLOCK)[:-3], "Malformed"),
        (encode_lowered(PROGRAMS) + b"x", "trailing bytes"),
        (encode_lowered(PROGRAMS).replace(b"module", b"\xffodule", 1), "Malformed"),
    ],
)
def test_malformed_files(data: bytes, message: str) -> None:
    """Checks that malformed files raise a `ValueError`."""
    with pytest.raises(ValueError, match=message):
        decode_lowered(data)


def test_store_and_load(cache_directory: Path) -> None:
    """Checks that stored programs load and that a malformed file is removed."""
    assert load_lowered("missing") is None
    store_lowered("key", PROGRAMS)
    assert load_lowered("key") == PROGRAMS
    path = cache_directory / f"key{LOWERED_SUFFIX}"
    path.write_bytes(b"broken")
    assert load_lowered("key") is None
    assert not path.exists()


def test_compile_program_uses_the_disk_cache() -> None:
    """Checks that executables are stored, loaded and compiled again without the cache."""
    program = lower(
        lambda x: jnp.cos(x) + 3.25, (jax.ShapeDtypeStruct((2,), jnp.float64),)
    )
    _, statistics = compile_program(program, disk_cache=True)
    assert statistics.compiled == 1 and statistics.loaded == 0
    _, statistics = compile_program(program, disk_cache=True)
    assert statistics.loaded == 1 and statistics.compiled == 0
    _, statistics = compile_program(program, disk_cache=False)
    assert statistics.compiled == 1 and statistics.loaded == 0


def test_statistics_add() -> None:
    """Checks that statistics add field by field."""
    first = BuildStatistics(lower_seconds=1.0, lowered=1, skipped=("a",))
    second = BuildStatistics(lower_seconds=0.5, compiled=2, skipped=("b",))
    assert first.add(second) == BuildStatistics(
        lower_seconds=1.5, lowered=1, compiled=2, skipped=("a", "b")
    )
