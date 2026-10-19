"""The PJRT runtime and the caches that let later builds skip lowering and compiling.

Building a term has two slow steps, lowering its programs to StableHLO and compiling them
with the PJRT plugin. The disk cache stores the result of both. Lowered programs are stored
under a key of the term's jaxpr, signature and options, so a warm build traces the function
but neither lowers nor compiles. Executables are stored under a hash of their StableHLO, the
compile options and the plugin.
"""

import functools
import hashlib
import importlib.metadata
import os
import re
import struct
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jaxormick

from jaxcip import _runtime
from jaxcip._programs import (
    LoweredProgram,
    LoweredPrograms,
    ProgramOptions,
    TermSignature,
)
from jaxcip._runtime import Executable
from jaxcip._types import RuntimeInfo

PLUGIN_ENVIRONMENT_VARIABLE = "JAXCIP_PJRT_PLUGIN"
"""Environment variable with the path of the PJRT plugin that is loaded by default."""

CACHE_ENVIRONMENT_VARIABLE = "JAXCIP_CACHE_DIR"
"""Environment variable with the directory of the disk cache."""

LOCATION_PATTERN = re.compile(r"^#loc.*$|\s*loc\((?:[^()]|\([^()]*\))*\)", re.MULTILINE)
"""Source locations in StableHLO text, which differ between traces of the same function."""

LOWERED_MAGIC = b"JAXCIP-LOWERED-1\n"
"""First bytes of a file of lowered programs."""


class PluginIdentity(NamedTuple):
    """What identifies the loaded plugin in cache keys."""

    path: Path
    size: int
    modified_ns: int


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

    def add(self, other: "BuildStatistics") -> "BuildStatistics":
        """Returns the sum of two statistics."""
        return BuildStatistics(
            *(first + second for first, second in zip(self, other, strict=True))
        )


plugin_identity: PluginIdentity | None = None
"""The loaded plugin, set by `initialize`."""


def initialize(plugin: Path | None = None) -> RuntimeInfo:
    """Loads a PJRT plugin and returns details of its runtime.

    Without a path the plugin comes from the `JAXCIP_PJRT_PLUGIN` environment variable.
    Building a problem calls this when no plugin is loaded yet. A process loads one plugin,
    so later calls return the details of the first.
    """
    global plugin_identity
    if _runtime.is_initialized():
        return _runtime.runtime_info()
    if plugin is None:
        value = os.environ.get(PLUGIN_ENVIRONMENT_VARIABLE)
        if not value:
            raise RuntimeError(
                f"Pass the path of a PJRT plugin or set {PLUGIN_ENVIRONMENT_VARIABLE}."
            )
        plugin = Path(value)
    path = plugin.resolve()
    info = _runtime.initialize(str(path))
    status = path.stat()
    plugin_identity = PluginIdentity(path, status.st_size, status.st_mtime_ns)
    return info


@functools.cache
def compile_options() -> bytes:
    """Returns the serialized `CompileOptionsProto` for one replica and one partition."""
    from jax._src import compiler

    options = compiler.get_compile_options(num_replicas=1, num_partitions=1)
    return options.SerializeAsString()


@functools.cache
def library_identity() -> bytes:
    """Returns the versions of the libraries that lowered programs depend on.

    JAXormick is installed from git, so its version number does not change with its code.
    Its source files are hashed instead.
    """
    digest = hashlib.sha256()
    for name in ("jax", "jaxlib", "jaxcip"):
        digest.update(f"{name}={importlib.metadata.version(name)}\0".encode())
    for path in sorted(Path(jaxormick.__file__).parent.rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.digest()


def cache_directory() -> Path:
    """Returns the directory of the disk cache, creating it if needed."""
    value = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    directory = Path(value) if value else Path.home() / ".cache" / "jaxcip"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def term_key(jaxpr_text: str, signature: TermSignature, options: ProgramOptions) -> str:
    """Returns the key of a term's programs, equal for terms that compile to the same code."""
    digest = hashlib.sha256(library_identity())
    digest.update(jaxpr_text.encode())
    digest.update(repr(signature).encode())
    digest.update(repr(options).encode())
    return digest.hexdigest()


def write_atomically(path: Path, data: bytes) -> None:
    """Writes a file that appears under its name only when complete."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def encode_lowered(programs: LoweredPrograms) -> bytes:
    """Returns lowered programs as bytes."""
    parts = [LOWERED_MAGIC]
    for program in programs:
        if program is None:
            parts.append(struct.pack("<B", 0))
            continue
        text = program.text.encode()
        parts.append(struct.pack("<BQQ", 1, len(program.code), len(text)))
        parts += [program.code, text]
    return b"".join(parts)


def decode_lowered(data: bytes) -> LoweredPrograms:
    """Returns lowered programs from bytes, raising a `ValueError` if they are malformed."""
    if not data.startswith(LOWERED_MAGIC):
        raise ValueError("Not a file of lowered programs")
    offset = len(LOWERED_MAGIC)
    programs: list[LoweredProgram | None] = []
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
    except (struct.error, UnicodeDecodeError) as error:
        raise ValueError(f"Malformed file of lowered programs: {error}") from error
    if offset != len(data):
        raise ValueError("Malformed file of lowered programs: trailing bytes")
    return LoweredPrograms._make(programs)


def load_lowered(key: str) -> LoweredPrograms | None:
    """Returns the lowered programs stored under a key, or `None` if there are none.

    A malformed file is removed.
    """
    path = cache_directory() / f"{key}.lowered"
    if not path.exists():
        return None
    try:
        return decode_lowered(path.read_bytes())
    except ValueError:
        path.unlink(missing_ok=True)
        return None


def store_lowered(key: str, programs: LoweredPrograms) -> None:
    """Stores lowered programs under a key."""
    write_atomically(cache_directory() / f"{key}.lowered", encode_lowered(programs))


def executable_key(program: LoweredProgram) -> str:
    """Returns the cache key of a program's executable."""
    if plugin_identity is None:
        raise RuntimeError("No PJRT plugin is loaded.")
    digest = hashlib.sha256(LOCATION_PATTERN.sub("", program.text).encode())
    digest.update(b"\0" + compile_options() + b"\0" + repr(plugin_identity).encode())
    return digest.hexdigest()


def compile_program(
    program: LoweredProgram, disk_cache: bool
) -> tuple[Executable, BuildStatistics]:
    """Compiles a program, or loads its executable from the disk cache.

    A cached executable that the plugin rejects is removed and compiled again.
    """
    path = cache_directory() / f"{executable_key(program)}.pjrt" if disk_cache else None
    if path is not None and path.exists():
        start = time.perf_counter()
        try:
            executable = Executable.deserialize(path.read_bytes())
        except RuntimeError:
            path.unlink(missing_ok=True)
        else:
            return executable, BuildStatistics(
                load_seconds=time.perf_counter() - start, loaded=1
            )
    start = time.perf_counter()
    executable = Executable.compile(program.code, compile_options())
    statistics = BuildStatistics(
        compile_seconds=time.perf_counter() - start, compiled=1
    )
    if path is not None:
        write_atomically(path, executable.serialize())
    return executable, statistics


def enable_x64() -> None:
    """Switches JAX to 64-bit floats, which the expression handler uses throughout."""
    jax.config.update("jax_enable_x64", True)
