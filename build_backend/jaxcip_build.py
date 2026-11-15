"""The build backend of jaxcip, which is meson-python after the JLL wraps are written.

Meson reads the wrap files of `subprojects` once, when it parses `project()`, so wraps that
`meson-jll sync` writes later in the same run stay unknown to it. The first configuration
therefore fails at the first `dependency()` call, though it leaves the wraps behind. This
backend makes that first run itself, so the real run finds them.
"""

import functools
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mesonpy

SUBPROJECTS = Path("subprojects")
"""Where `meson-jll` writes the wraps, relative to the project root that the build runs in."""


def meson_command() -> list[str]:
    """Returns the command that runs Meson, which is the `meson` of the build environment."""
    found = shutil.which("meson")
    if found is not None:
        return [found]
    return [
        sys.executable,
        "-c",
        "import sys; from mesonbuild.mesonmain import main; sys.exit(main())",
    ]


def wraps_exist() -> bool:
    """Returns whether `meson-jll sync` has written the wraps of SCIP."""
    return any(SUBPROJECTS.glob("SCIP-*.wrap"))


def write_wraps() -> None:
    """Configures the project in a scratch directory unless the wraps exist already.

    The configuration is expected to fail after `meson-jll sync` has run, so its result is
    not checked. If the wraps are missing afterwards, its output is printed, because then
    the sync itself failed and the build that follows will fail with a less helpful error.
    """
    if wraps_exist():
        return
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            [*meson_command(), "setup", "--vsenv", scratch, "."],
            check=False,
            capture_output=True,
            text=True,
        )
    if not wraps_exist():
        sys.stderr.write(result.stdout + result.stderr)


def after_writing_wraps(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Returns a PEP 517 hook of meson-python that first writes the wraps."""

    @functools.wraps(hook)
    def wrapped(*arguments: Any, **keywords: Any) -> Any:
        write_wraps()
        return hook(*arguments, **keywords)

    return wrapped


build_sdist = after_writing_wraps(mesonpy.build_sdist)
build_wheel = after_writing_wraps(mesonpy.build_wheel)
build_editable = after_writing_wraps(mesonpy.build_editable)
get_requires_for_build_sdist = mesonpy.get_requires_for_build_sdist
get_requires_for_build_wheel = mesonpy.get_requires_for_build_wheel
get_requires_for_build_editable = mesonpy.get_requires_for_build_editable
