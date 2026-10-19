"""Tests of the PJRT runtime: plugin loading, executables and errors."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import jaxcip as jcp
from jaxcip import _runtime
from jaxcip._cache import compile_options
from jaxcip._programs import lower
from jaxcip._runtime import Executable

HEADER_DIRECTORY = Path(__file__).parents[1] / "third_party" / "xla"
"""Directory from which the PJRT C API header is included."""

OLD_PLUGIN_SOURCE = """
#include "xla/pjrt/c/pjrt_c_api.h"

static PJRT_Api api = {
    .struct_size = PJRT_Api_STRUCT_SIZE,
    .pjrt_api_version = {
        .struct_size = PJRT_Api_Version_STRUCT_SIZE,
        .major_version = 0,
        .minor_version = 1,
    },
};

const PJRT_Api* GetPjrtApi(void) { return &api; }
"""
"""A plugin that reports a PJRT C API version older than the runtime needs."""


def run_python(code: str) -> subprocess.CompletedProcess:
    """Runs code in a fresh interpreter without a loaded plugin."""
    environment = {
        key: value for key, value in os.environ.items() if key != "JAXCIP_PJRT_PLUGIN"
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_runtime_avoids_copies(runtime: jcp.RuntimeInfo) -> None:
    """Checks that the CPU plugin aliases inputs and exposes outputs in host memory."""
    assert runtime.platform == "cpu"
    assert runtime.api_major_version == 0
    assert runtime.api_minor_version >= _runtime.MINIMUM_API_MINOR_VERSION
    assert runtime.aliased_inputs and runtime.inplace_outputs
    assert jcp.initialize() == runtime
    assert _runtime.initialize("ignored") == runtime
    assert _runtime.is_initialized()


def test_executable_round_trip() -> None:
    """Checks that an executable survives serialization."""
    program = lower(
        lambda x: (jnp.sin(x), x), (jax.ShapeDtypeStruct((3,), jnp.float64),)
    )
    executable = Executable.compile(program.code, compile_options())
    assert executable.num_outputs == 2
    serialized = executable.serialize()
    assert Executable.deserialize(serialized).num_outputs == 2


def test_errors_are_recorded() -> None:
    """Checks that rejected bytes raise and that the error is recorded."""
    count = _runtime.error_info().count
    with pytest.raises(RuntimeError, match="PJRT_Executable_DeserializeAndLoad"):
        Executable.deserialize(b"not an executable")
    with pytest.raises(RuntimeError, match="PJRT_Client_Compile"):
        Executable.compile(b"not bytecode", compile_options())
    info = _runtime.error_info()
    assert info.count == count + 2
    assert info.last_message


def test_uninitialized_runtime() -> None:
    """Checks the errors of using the runtime before a plugin is loaded."""
    result = run_python(
        """
        import pytest
        import jaxcip as jcp
        from jaxcip import _runtime
        from jaxcip._handler import FunctionHandle

        assert not _runtime.is_initialized()
        for call in (
            _runtime.runtime_info,
            lambda: _runtime.Executable.compile(b"", b""),
            lambda: _runtime.Executable.deserialize(b""),
            lambda: FunctionHandle(1, 1, None),
        ):
            with pytest.raises(RuntimeError, match="not initialized"):
                call()
        with pytest.raises(RuntimeError, match="JAXCIP_PJRT_PLUGIN"):
            jcp.initialize()
        """
    )
    assert result.returncode == 0, result.stderr


def compile_library(source: str, path: Path) -> Path:
    """Compiles C source into a shared library."""
    subprocess.run(
        ["cc", "-shared", "-fPIC", f"-I{HEADER_DIRECTORY}", "-x", "c", "-", "-o", path],
        input=source,
        text=True,
        check=True,
    )
    return path


def test_bad_plugins(tmp_path: Path) -> None:
    """Checks the errors of a missing plugin, a library that is not one, and an old one."""
    not_plugin = compile_library("int answer(void) { return 42; }", tmp_path / "a.so")
    old_plugin = compile_library(OLD_PLUGIN_SOURCE, tmp_path / "b.so")
    result = run_python(
        f"""
        from pathlib import Path

        import pytest
        import jaxcip as jcp

        with pytest.raises(OSError, match="Cannot load"):
            jcp.initialize(Path("/nonexistent/libpjrt.so"))
        with pytest.raises(OSError, match="does not export GetPjrtApi"):
            jcp.initialize(Path({str(not_plugin)!r}))
        with pytest.raises(RuntimeError, match="too old"):
            jcp.initialize(Path({str(old_plugin)!r}))
        """
    )
    assert result.returncode == 0, result.stderr
