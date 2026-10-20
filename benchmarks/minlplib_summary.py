"""Summarizes the results of a MINLPLib run as Markdown tables.

Usage: `python minlplib_summary.py results.jsonl [results.jsonl ...]`.

The input is the file that `tests/minlplib` writes with `JAXCIP_MINLPLIB_RESULTS` set,
one JSON object per instance. Later lines replace earlier ones for the same instance, so
reruns can be appended. The output gives the counts of each outcome, the median ratios
of JAX-CIP to native SCIP over the instances both solved, and a table of every instance.
"""

import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

SOLVED = frozenset({"OPTIMAL", "INFEASIBLE"})
"""Statuses that end a solve with a proof."""


class Solve(NamedTuple):
    """The result of one solve of an instance."""

    status: str
    objective: float
    nodes: int
    seconds: float


class Result(NamedTuple):
    """The results of an instance, with `jaxcip` absent when it was not solved."""

    name: str
    native: Solve | None
    jaxcip: Solve | None
    outcome: str
    """`unsupported`, `error` or the status of the JAX-CIP solve."""


def read(paths: list[Path]) -> list[Result]:
    """Returns the last result of every instance in the files."""
    results: dict[str, Result] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            native = jaxcip = None
            if "native" in entry:
                native = solve(entry["native"])
            if "jaxcip" in entry:
                jaxcip = solve(entry["jaxcip"])
            if "unsupported" in entry:
                outcome = "unsupported"
            elif "error" in entry:
                outcome = "error"
            else:
                outcome = entry["jaxcip"]["status"]
            results[entry["name"]] = Result(entry["name"], native, jaxcip, outcome)
    return sorted(results.values())


def solve(entry: dict[str, object]) -> Solve:
    """Returns a solve from its JSON object."""
    objective = entry["objective"]
    return Solve(
        status=str(entry["status"]),
        objective=math.nan if objective is None else float(str(objective)),
        nodes=int(str(entry["nodes"])),
        seconds=float(str(entry["seconds"])),
    )


def ratio(numerator: float, denominator: float) -> float:
    """Returns a ratio of two counts or times, with a floor that avoids dividing by zero."""
    return max(numerator, 1e-2) / max(denominator, 1e-2)


def summary(results: list[Result]) -> str:
    """Returns the Markdown summary of the results."""
    native_solved = sum(
        1 for item in results if item.native and item.native.status in SOLVED
    )
    both = [
        (native, jaxcip)
        for item in results
        if (native := item.native) and (jaxcip := item.jaxcip)
        if native.status in SOLVED and jaxcip.status in SOLVED
    ]
    outcomes = Counter(item.outcome for item in results)
    lines = [
        "| Outcome | Instances |",
        "| --- | ---: |",
        f"| Instances | {len(results)} |",
        *(f"| JAX-CIP {name.lower()} | {count} |" for name, count in outcomes.items()),
        f"| Native SCIP solved | {native_solved} |",
        f"| Both solved | {len(both)} |",
        "",
    ]
    if both:
        times = [ratio(jaxcip.seconds, native.seconds) for native, jaxcip in both]
        nodes = [ratio(jaxcip.nodes, native.nodes) for native, jaxcip in both]
        lines += [
            "| Ratio of JAX-CIP to native SCIP, over instances both solved | Median |",
            "| --- | ---: |",
            f"| Solving time | {statistics.median(times):.2f} |",
            f"| Nodes | {statistics.median(nodes):.2f} |",
            "",
        ]
    lines += [
        "| Instance | Native | Seconds | JAX-CIP | Seconds | Nodes ratio |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for item in results:
        native = item.native
        jaxcip = item.jaxcip
        native_cells = (
            f"{native.status.lower()} | {native.seconds:.2f}" if native else "- | -"
        )
        jaxcip_cells = (
            f"{jaxcip.status.lower()} | {jaxcip.seconds:.2f}"
            if jaxcip
            else f"{item.outcome} | -"
        )
        nodes_cell = (
            f"{ratio(jaxcip.nodes, native.nodes):.2f}" if native and jaxcip else "-"
        )
        lines.append(
            f"| `{item.name}` | {native_cells} | {jaxcip_cells} | {nodes_cell} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(summary(read([Path(item) for item in sys.argv[1:]])), end="")
