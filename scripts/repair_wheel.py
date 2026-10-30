"""Repairs a Linux wheel of JAX-CIP and checks that it conforms to a manylinux policy.

Meson-python copies each JLL library into the wheel under the file name that the linker
found, such as `libscip.so`. The extension modules need the library by its soname, such as
`libscip.so.10.1`, so no library would load from the wheel. This script renames every
bundled library to its soname and then runs `auditwheel repair`, which tags the wheel.
Meson-python only bundles the libraries that the extension modules link directly. The
libraries that those need in turn, such as `libmumps_common.so`, are found in the JLL
directories of the build and copied in by auditwheel.

Usage: `python repair_wheel.py <wheel> <destination directory>`.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

LIBRARY_DIRECTORY = ".jaxcip.mesonpy.libs"
"""The directory of the wheel that meson-python puts the bundled libraries in."""


def soname_of(library: Path) -> str:
    """Returns the soname of a shared library, which is its file name when it has none."""
    result = subprocess.run(
        ["patchelf", "--print-soname", str(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or library.name


def jll_library_path() -> str:
    """Returns a library search path with the lib directories of the downloaded JLLs."""
    subprojects = Path(__file__).resolve().parent.parent / "subprojects"
    directories = [
        str(path) for path in sorted(subprojects.glob("*/lib*")) if path.is_dir()
    ]
    inherited = os.environ.get("LD_LIBRARY_PATH")
    return os.pathsep.join([*directories, *([inherited] if inherited else [])])


def rename_to_sonames(unpacked: Path) -> None:
    """Renames the bundled libraries of an unpacked wheel to their sonames."""
    for library in sorted((unpacked / LIBRARY_DIRECTORY).glob("*.so*")):
        soname = soname_of(library)
        if soname != library.name:
            library.rename(library.with_name(soname))


def main(wheel: Path, destination: Path) -> None:
    """Renames the libraries of a wheel and repairs it into the destination directory."""
    with tempfile.TemporaryDirectory() as scratch_name:
        scratch = Path(scratch_name)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "unpack",
                "--dest",
                str(scratch),
                str(wheel),
            ],
            check=True,
        )
        (unpacked,) = [path for path in scratch.iterdir() if path.is_dir()]
        rename_to_sonames(unpacked)
        renamed = scratch / "renamed"
        renamed.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                "--dest-dir",
                str(renamed),
                str(unpacked),
            ],
            check=True,
        )
        (packed,) = renamed.glob("*.whl")
        subprocess.run(
            ["auditwheel", "repair", "-w", str(destination), str(packed)],
            check=True,
            env={**os.environ, "LD_LIBRARY_PATH": jll_library_path()},
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
