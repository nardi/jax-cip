#!/usr/bin/env python3
"""Regenerates an MSVC-compatible import library from a DLL's own export
table, using dumpbin and lib.exe. See the meson-jll comment that generated
this file for why this exists.

Usage: dll_to_lib.py <dll-path> <output-lib-path> <machine>
"""
import os
import re
import subprocess
import sys
import tempfile


def main():
    dll_path, output_path, machine = sys.argv[1:4]

    exports = subprocess.run(
        ["dumpbin", "/exports", dll_path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    # Each exported symbol's line looks like:
    #   1    0 00001080 amd_control
    # (ordinal, hint, RVA, name), the name being the last column.
    names = []
    for line in exports.splitlines():
        match = re.match(r"^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)", line)
        if match:
            names.append(match.group(1))

    # lib.exe embeds a LIBRARY statement's name as the DLL to load at
    # runtime. Without one, it falls back to the .def file's own base
    # name, which is this script's randomly named temp file, not the DLL
    # actually being wrapped, silently producing an import library that
    # points at a DLL that does not exist.
    dll_name = os.path.basename(dll_path)

    definition_fd, definition_path = tempfile.mkstemp(suffix=".def")
    try:
        with os.fdopen(definition_fd, "w") as definition_file:
            definition_file.write(f'LIBRARY "{dll_name}"\n')
            definition_file.write("EXPORTS\n")
            for name in names:
                definition_file.write(f"{name}\n")

        subprocess.run(
            [
                "lib",
                f"/def:{definition_path}",
                f"/out:{output_path}",
                f"/machine:{machine}",
            ],
            check=True,
        )
    finally:
        os.unlink(definition_path)


if __name__ == "__main__":
    main()
