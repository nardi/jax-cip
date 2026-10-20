"""Compares benchmark results with a baseline, relative to reference times.

Usage: `python compare.py results.json baseline.json`.

Each time is divided by the time of its reference in the same run, which `extra_info`
names: the value program of the same function for a program, and a fixed NumPy
computation for a build or a solve. These ratios are compared, so the comparison holds
across machines. The script prints a table and exits with status 1 when a ratio grew by
more than the allowed factor.
"""

import json
import sys
from pathlib import Path

ALLOWED_GROWTH = 1.5
"""Factor by which a ratio may grow before it counts as a regression."""


def ratios(path: Path) -> dict[tuple[str, str], float]:
    """Returns the time relative to its reference for every benchmark in a file."""
    benchmarks = json.loads(path.read_text())["benchmarks"]
    times = {
        (entry["extra_info"]["kind"], entry["extra_info"]["name"]): (
            entry["stats"]["median"],
            entry["extra_info"]["reference"],
        )
        for entry in benchmarks
    }
    references = {
        name: time
        for (kind, name), (time, _) in times.items()
        if kind == "reference" or name.endswith(".value")
    }
    return {
        key: time / references[reference if kind != "program" else f"{reference}.value"]
        for key, (time, reference) in times.items()
        for kind in [key[0]]
        if kind != "reference"
    }


def main(results: Path, baseline: Path) -> int:
    """Prints the comparison and returns the exit status."""
    current, previous = ratios(results), ratios(baseline)
    regressions = 0
    print(f"{'kind':<11} {'benchmark':<32} {'ratio':>9} {'baseline':>9}")
    for key in sorted(current):
        ratio = current[key]
        reference = previous.get(key)
        mark = ""
        if reference is not None and ratio > ALLOWED_GROWTH * reference:
            mark = "  slower"
            regressions += 1
        shown = f"{reference:9.2f}" if reference is not None else f"{'-':>9}"
        print(f"{key[0]:<11} {key[1]:<32} {ratio:9.2f} {shown}{mark}")
    print(f"\n{regressions} regressions beyond {ALLOWED_GROWTH}x.")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
