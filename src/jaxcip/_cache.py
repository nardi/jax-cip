"""The caches that let later builds skip lowering and compiling.

Building a term has two slow steps, lowering its programs to StableHLO and compiling them
with the PJRT plugin. The disk cache stores the result of both, in the directory of cyxla.
Lowered programs are stored under a key of the term's jaxpr, signature and options, so a
warm build traces the function but neither lowers nor compiles. Cyxla stores the
executables.
"""

import json
import os
import struct
import threading
from pathlib import Path
from typing import NamedTuple

import cyxla
import jax
import numpy as np
from cyxla import ArrayType, Executable
from jaxtyping import Float, Int

from jaxcip._blocks import BlockPattern, LoweredTerm
from jaxcip._programs import (
    LoweredProgram,
    LoweredPrograms,
    ProgramOptions,
    TermSignature,
)

LOWERED_MAGIC = b"JAXCIP-LOWERED-4\n"
"""First bytes of a file of lowered programs."""

LOWERED_SUFFIX = ".jaxcip-lowered"
"""Suffix of the files of lowered programs, which differs from the one cyxla uses."""

AFFINE_SUFFIX = ".jaxcip-affine.npz"
"""Suffix of the files with the entries of the affine remainder of a block."""

KEY_DISTRIBUTIONS = ("jaxcip", "jaxormick")
"""Distributions besides JAX and cyxla whose code determines the lowered programs."""


class BuildStatistics(NamedTuple):
    """Time and counts of building the terms of a problem."""

    lower_seconds: float = 0.0
    """Seconds spent tracing and lowering programs to StableHLO."""
    compile_seconds: float = 0.0
    """Seconds spent compiling StableHLO with the PJRT plugin."""
    load_seconds: float = 0.0
    """Seconds spent reading lowered programs and executables from the disk cache."""
    lowered: int = 0
    """Terms whose programs were lowered."""
    compiled: int = 0
    """Programs compiled."""
    loaded: int = 0
    """Programs whose executables came from the disk cache."""
    reused: int = 0
    """Terms whose compiled programs were already in memory."""
    skipped: tuple[str, ...] = ()
    """Optional programs that were left out, with the reason."""
    wall_seconds: float = 0.0
    """Seconds the build took on the wall clock, which is less than the sum of the seconds
    above when several threads build programs at once."""

    def add(self, other: "BuildStatistics") -> "BuildStatistics":
        """Returns the sum of two statistics."""
        return BuildStatistics(
            *(first + second for first, second in zip(self, other, strict=True))
        )


def term_key(
    jaxpr_text: str,
    signs: tuple[float, ...],
    signature: TermSignature,
    options: ProgramOptions,
) -> str:
    """Returns the key of a term's programs, equal for terms that compile to the same code."""
    return cyxla.cache_key(
        jaxpr_text,
        repr(signs),
        repr(signature),
        repr(options),
        distributions=KEY_DISTRIBUTIONS,
    )


def write_atomically(path: Path, data: bytes) -> None:
    """Writes a file that appears under its name only when complete."""
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def encode_lowered(term: LoweredTerm) -> bytes:
    """Returns the lowered programs of a term, and its pattern if it has one, as bytes."""
    parts = [LOWERED_MAGIC]
    for program in term.programs:
        if program is None:
            parts.append(struct.pack("<B", 0))
            continue
        text = program.text.encode()
        types = json.dumps(
            [[item.dtype.name, list(item.shape)] for item in program.input_types]
        ).encode()
        parts.append(struct.pack("<BQQQ", 1, len(program.code), len(text), len(types)))
        parts += [program.code, text, types]
    pattern = term.pattern
    if pattern is None:
        parts.append(struct.pack("<B", 0))
    else:
        parts.append(
            struct.pack("<BQQ", 1, len(pattern.pointers), len(pattern.columns))
        )
        parts += [pattern.pointers.tobytes(), pattern.columns.tobytes()]
    return b"".join(parts)


def decode_lowered(data: bytes) -> LoweredTerm:
    """Returns a lowered term from bytes, raising a `ValueError` if they are malformed."""
    if not data.startswith(LOWERED_MAGIC):
        raise ValueError("Not a file of lowered programs")
    offset = len(LOWERED_MAGIC)
    programs: list[LoweredProgram | None] = []
    pattern = None
    try:
        for _ in LoweredPrograms._fields:
            (is_present,) = struct.unpack_from("<B", data, offset)
            offset += 1
            if not is_present:
                programs.append(None)
                continue
            code_size, text_size, types_size = struct.unpack_from("<QQQ", data, offset)
            offset += 24
            code = data[offset : offset + code_size]
            text_end = offset + code_size + text_size
            text = data[offset + code_size : text_end].decode()
            types = tuple(
                ArrayType(np.dtype(name), tuple(shape))
                for name, shape in json.loads(data[text_end : text_end + types_size])
            )
            offset = text_end + types_size
            programs.append(LoweredProgram(code, text, types))
        (has_pattern,) = struct.unpack_from("<B", data, offset)
        offset += 1
        if has_pattern:
            count, nnz = struct.unpack_from("<QQ", data, offset)
            offset += 16
            size = np.dtype(np.intc).itemsize
            pointers = np.frombuffer(data, np.intc, count, offset)
            offset += count * size
            columns = np.frombuffer(data, np.intc, nnz, offset)
            offset += nnz * size
            pattern = BlockPattern(pointers.copy(), columns.copy())
    except (struct.error, UnicodeDecodeError, ValueError, TypeError) as error:
        raise ValueError(f"Malformed file of lowered programs: {error}") from error
    if offset != len(data):
        raise ValueError("Malformed file of lowered programs: trailing bytes")
    return LoweredTerm(LoweredPrograms._make(programs), pattern)


def load_lowered(key: str) -> LoweredTerm | None:
    """Returns the lowered term stored under a key, or `None` if there is none.

    A malformed file is removed.
    """
    path = cyxla.cache_directory() / f"{key}{LOWERED_SUFFIX}"
    if not path.exists():
        return None
    try:
        return decode_lowered(path.read_bytes())
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def store_lowered(key: str, term: LoweredTerm) -> None:
    """Stores a lowered term under a key."""
    write_atomically(
        cyxla.cache_directory() / f"{key}{LOWERED_SUFFIX}", encode_lowered(term)
    )


def compile_program(
    program: LoweredProgram, disk_cache: bool
) -> tuple[Executable, BuildStatistics]:
    """Compiles a program for the CPU, or loads its executable from the disk cache."""
    executable, compiled = cyxla.compile_program(
        cyxla.LoweredProgram(
            program.code, program.text, program.input_types, platform="cpu"
        ),
        disk_cache,
    )
    return executable, BuildStatistics(
        compile_seconds=compiled.compile_seconds,
        load_seconds=compiled.load_seconds,
        compiled=compiled.compiled,
        loaded=compiled.loaded,
    )


def enable_x64() -> None:
    """Switches JAX to 64-bit floats, which the expression handler uses throughout."""
    jax.config.update("jax_enable_x64", True)


class AffineEntries(NamedTuple):
    """Which entries of the Jacobian of an affine remainder exist, and its values if fixed."""

    rows: Int[np.ndarray, " nnz"]
    columns: Int[np.ndarray, " nnz"]
    constants: Float[np.ndarray, " rows"] | None = None
    """The constants of a remainder that does not depend on the parameters, else `None`."""
    values: Float[np.ndarray, " nnz"] | None = None
    """The Jacobian entries of such a remainder, else `None`."""


def affine_key(*parts: str | bytes) -> str:
    """Returns the key of an affine remainder, equal for remainders that compile alike."""
    return cyxla.cache_key(*parts, distributions=KEY_DISTRIBUTIONS)


def store_affine(key: str, entries: AffineEntries) -> None:
    """Stores the entries of an affine remainder under a key."""
    path = cyxla.cache_directory() / f"{key}{AFFINE_SUFFIX}"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fields = {
        name: value for name, value in entries._asdict().items() if value is not None
    }
    with temporary.open("wb") as file:
        np.savez(file, **fields)
    temporary.replace(path)


def load_affine(key: str) -> AffineEntries | None:
    """Returns the entries stored under a key, or `None` if there are none.

    A file that cannot be read is removed.
    """
    path = cyxla.cache_directory() / f"{key}{AFFINE_SUFFIX}"
    if not path.exists():
        return None
    try:
        with np.load(path) as archive:
            return AffineEntries(**{name: archive[name] for name in archive.files})
    except (OSError, ValueError, KeyError, TypeError):
        path.unlink(missing_ok=True)
        return None
