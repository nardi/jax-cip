"""Tests of the disk cache of lowered programs and executables."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcip import _cache
from jaxcip._blocks import BlockPattern, LoweredTerm
from jaxcip._cache import (
    BuildStatistics,
    cache_directory,
    compile_program,
    decode_lowered,
    encode_lowered,
    executable_key,
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
    path = cache_directory / "key.lowered"
    path.write_bytes(b"broken")
    assert load_lowered("key") is None
    assert not path.exists()


def test_cache_directory_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checks that the cache lives under the home directory without the variable."""
    monkeypatch.delenv(_cache.CACHE_ENVIRONMENT_VARIABLE)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_directory() == tmp_path / ".cache" / "jaxcip"
    assert cache_directory().is_dir()


def test_executable_key_needs_a_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks that executable keys depend on the loaded plugin."""
    monkeypatch.setattr(_cache, "plugin_identity", None)
    with pytest.raises(RuntimeError, match="No PJRT plugin"):
        executable_key(PROGRAM)


def test_executable_key_ignores_locations() -> None:
    """Checks that programs that differ only in source locations share a key."""
    shapes = (jax.ShapeDtypeStruct((2,), jnp.float64),)
    first = lower(lambda x: jnp.sin(x) * 2, shapes)

    second = lower(lambda x: jnp.sin(x) * 2, shapes)
    assert first.text != second.text
    assert executable_key(first) == executable_key(second)


def test_compile_program_uses_the_disk_cache(cache_directory: Path) -> None:
    """Checks that executables are stored, loaded, and compiled again when corrupted."""
    program = lower(jnp.cos, (jax.ShapeDtypeStruct((2,), jnp.float64),))
    path = cache_directory / f"{executable_key(program)}.pjrt"
    path.unlink(missing_ok=True)
    _, statistics = compile_program(program, disk_cache=True)
    assert statistics.compiled == 1 and path.exists()
    _, statistics = compile_program(program, disk_cache=True)
    assert statistics.loaded == 1 and statistics.compiled == 0
    path.write_bytes(b"corrupted")
    _, statistics = compile_program(program, disk_cache=True)
    assert statistics.compiled == 1
    assert path.read_bytes() != b"corrupted"
    modified = os.stat(path).st_mtime_ns
    _, statistics = compile_program(program, disk_cache=False)
    assert statistics.compiled == 1 and os.stat(path).st_mtime_ns == modified


def test_statistics_add() -> None:
    """Checks that statistics add field by field."""
    first = BuildStatistics(lower_seconds=1.0, lowered=1, skipped=("a",))
    second = BuildStatistics(lower_seconds=0.5, compiled=2, skipped=("b",))
    assert first.add(second) == BuildStatistics(
        lower_seconds=1.5, lowered=1, compiled=2, skipped=("a", "b")
    )
