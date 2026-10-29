"""The caches that let later builds skip lowering and compiling.

Building a term has two slow steps, lowering its programs to StableHLO and compiling them
with the PJRT plugin. The disk cache stores the result of both, in the directory of cyxla.
Lowered programs are stored under a key of the term's jaxpr, signature and options, so a
warm build traces the function but neither lowers nor compiles. Cyxla stores the
executables.
"""

import os
import struct
import threading
from pathlib import Path
from typing import NamedTuple

import cyxla
import jax
import numpy as np
from cyxla import Executable

from jaxcip._blocks import BlockPattern, LoweredTerm
from jaxcip._programs import (
    LoweredProgram,
    LoweredPrograms,
    ProgramOptions,
    TermSignature,
)

LOWERED_MAGIC = b"JAXCIP-LOWERED-3\n"
"""First bytes of a file of lowered programs."""

LOWERED_SUFFIX = ".jaxcip-lowered"
"""Suffix of the files of lowered programs, which differs from the one cyxla uses."""

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
        parts.append(struct.pack("<BQQ", 1, len(program.code), len(text)))
        parts += [program.code, text]
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
            code_size, text_size = struct.unpack_from("<QQ", data, offset)
            offset += 16
            code = data[offset : offset + code_size]
            text = data[offset + code_size : offset + code_size + text_size].decode()
            offset += code_size + text_size
            programs.append(LoweredProgram(code, text))
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
    except (struct.error, UnicodeDecodeError, ValueError) as error:
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
        cyxla.LoweredProgram(program.code, program.text, platform="cpu"), disk_cache
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
