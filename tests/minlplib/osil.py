"""A reader of OSiL files, the XML format of MINLPLib instances.

The format is described by the [OSiL schema](https://www.coin-or.org/OS/OSiL.html). The
reader takes the parts that MINLPLib uses: variables, one objective, constraint bounds,
linear and quadratic coefficients and nonlinear expression trees.
"""

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from jaxtyping import Float, Int
from lxml import etree

from jaxcip import Domain, Sense

NAMESPACE = "{os.optimizationservices.org}"
"""XML namespace of OSiL elements."""

OBJECTIVE_INDEX = -1
"""Row index that OSiL gives the objective in quadratic and nonlinear parts."""

DOMAINS = {"C": Domain.CONTINUOUS, "I": Domain.INTEGER, "B": Domain.BINARY}
"""Domains of the OSiL variable types."""


class UnsupportedInstance(Exception):
    """Raised for an instance with parts that the reader or converter does not handle."""


class Node(NamedTuple):
    """A node of a nonlinear expression tree."""

    operator: str
    """The OSnL element name, such as `plus` or `variable`."""
    children: tuple["Node", ...] = ()
    value: float = 0.0
    """The number of a `number` node or the coefficient of a `variable` node."""
    index: int = 0
    """The variable index of a `variable` node."""


class Row(NamedTuple):
    """The parts of the objective or of one constraint."""

    lower: float
    upper: float
    linear_indices: Int[np.ndarray, " k"]
    linear_values: Float[np.ndarray, " k"]
    quadratic: tuple[tuple[int, int, float], ...]
    """Terms `coefficient · x[first] · x[second]`."""
    tree: Node | None


class Instance(NamedTuple):
    """A MINLPLib instance."""

    name: str
    names: tuple[str, ...]
    """Names of the variables."""
    lower: Float[np.ndarray, " n"]
    upper: Float[np.ndarray, " n"]
    domains: tuple[Domain, ...]
    sense: Sense
    objective: Row
    """The objective, with its constant as both bounds."""
    constraints: tuple[Row, ...]


def number(text: str | None, default: float) -> float:
    """Returns an OSiL number, which may be `INF` or `-INF`."""
    match text:
        case None:
            return default
        case "INF" | "Infinity":
            return math.inf
        case "-INF" | "-Infinity":
            return -math.inf
        case _:
            return float(text)


def tag(element: etree._Element) -> str:
    """Returns the name of an element without its namespace."""
    return etree.QName(element).localname


def child(element: etree._Element, name: str) -> etree._Element | None:
    """Returns the first child of an element with a name, or `None`."""
    return element.find(NAMESPACE + name)


def required(element: etree._Element, name: str) -> etree._Element:
    """Returns the first child of an element with a name, which must exist."""
    if (found := child(element, name)) is None:
        raise UnsupportedInstance(f"missing element {name}")
    return found


def attribute(element: etree._Element, name: str) -> str:
    """Returns an attribute of an element, which must exist."""
    if (value := element.get(name)) is None:
        raise UnsupportedInstance(f"missing attribute {name}")
    return value


def text(element: etree._Element) -> str:
    """Returns the text of an element, which must exist."""
    if element.text is None:
        raise UnsupportedInstance(f"empty element {tag(element)}")
    return element.text


def elements(element: etree._Element, name: str, item: str) -> list[etree._Element]:
    """Returns the `item` children of the `name` child of an element."""
    parent = child(element, name)
    return [] if parent is None else parent.findall(NAMESPACE + item)


def expand(element: etree._Element | None, dtype: type) -> np.ndarray:
    """Returns the values of an array of `el` elements with `mult` and `incr` expanded."""
    values: list[float] = []
    if element is None:
        return np.zeros(0, dtype)
    for item in element.iter(NAMESPACE + "el"):
        start = float(text(item))
        increment = float(item.get("incr", 0))
        values += [start + step * increment for step in range(int(item.get("mult", 1)))]
    return np.asarray(values, dtype)


def read_tree(element: etree._Element) -> Node:
    """Returns the expression tree of an OSnL element."""
    operator = tag(element)
    match operator:
        case "number":
            return Node(operator, value=float(attribute(element, "value")))
        case "variable":
            return Node(
                operator,
                value=float(element.get("coef", 1.0)),
                index=int(attribute(element, "idx")),
            )
        case _:
            return Node(operator, tuple(read_tree(item) for item in element))


def linear_rows(
    element: etree._Element | None, num_rows: int, num_variables: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns the indices and values of the linear coefficients of each row."""
    if element is None:
        return [(np.zeros(0, int), np.zeros(0))] * num_rows
    starts = expand(child(element, "start"), int)
    values = expand(child(element, "value"), float)
    if (columns := child(element, "colIdx")) is not None:
        indices = expand(columns, int)
        return [
            (indices[first:last], values[first:last])
            for first, last in zip(starts[:-1], starts[1:], strict=True)
        ]
    rows = expand(child(element, "rowIdx"), int)
    columns_of_entries = np.repeat(np.arange(num_variables), np.diff(starts))
    order = np.argsort(rows, kind="stable")
    bounds = np.searchsorted(rows[order], np.arange(num_rows + 1))
    return [
        (columns_of_entries[order[first:last]], values[order[first:last]])
        for first, last in zip(bounds[:-1], bounds[1:], strict=True)
    ]


def read(path: Path) -> Instance:
    """Reads an OSiL file."""
    root = etree.parse(str(path)).getroot()
    data = required(root, "instanceData")
    name = path.stem
    variables = elements(data, "variables", "var")
    if any(int(item.get("mult", 1)) != 1 for item in variables):
        raise UnsupportedInstance("repeated variable elements")
    num_variables = len(variables)
    types = [item.get("type", "C") for item in variables]
    if unknown := set(types) - DOMAINS.keys():
        raise UnsupportedInstance(f"variable types {sorted(unknown)}")
    lower = np.array([number(item.get("lb"), 0.0) for item in variables])
    upper = np.array([number(item.get("ub"), math.inf) for item in variables])

    objectives = child(data, "objectives")
    objective = None if objectives is None else child(objectives, "obj")
    sense = Sense.MINIMIZE
    objective_indices, objective_values = np.zeros(0, int), np.zeros(0)
    constant = 0.0
    if objective is not None:
        sense = Sense.MAXIMIZE if objective.get("maxOrMin") == "max" else Sense.MINIMIZE
        constant = float(objective.get("constant", 0.0))
        coefficients = list(objective.iter(NAMESPACE + "coef"))
        objective_indices = np.array(
            [int(attribute(item, "idx")) for item in coefficients], int
        )
        objective_values = np.array([float(text(item)) for item in coefficients])

    constraints = elements(data, "constraints", "con")
    num_rows = len(constraints)
    linear = linear_rows(
        child(data, "linearConstraintCoefficients"), num_rows, num_variables
    )
    quadratic: list[list[tuple[int, int, float]]] = [[] for _ in range(num_rows + 1)]
    if (terms := child(data, "quadraticCoefficients")) is not None:
        for term in terms.iter(NAMESPACE + "qTerm"):
            quadratic[int(attribute(term, "idx"))].append(
                (
                    int(attribute(term, "idxOne")),
                    int(attribute(term, "idxTwo")),
                    float(term.get("coef", 1.0)),
                )
            )
    trees: list[Node | None] = [None] * (num_rows + 1)
    if (expressions := child(data, "nonlinearExpressions")) is not None:
        for expression in expressions.iter(NAMESPACE + "nl"):
            (tree,) = list(expression)
            trees[int(attribute(expression, "idx"))] = read_tree(tree)

    rows = tuple(
        Row(
            lower=number(item.get("lb"), -math.inf) - float(item.get("constant", 0.0)),
            upper=number(item.get("ub"), math.inf) - float(item.get("constant", 0.0)),
            linear_indices=indices,
            linear_values=values,
            quadratic=tuple(quadratic[index]),
            tree=trees[index],
        )
        for index, (item, (indices, values)) in enumerate(
            zip(constraints, linear, strict=True)
        )
    )
    return Instance(
        name=name,
        names=tuple(
            item.get("name", f"x{index}") for index, item in enumerate(variables)
        ),
        lower=lower,
        upper=upper,
        domains=tuple(DOMAINS[item] for item in types),
        sense=sense,
        objective=Row(
            lower=constant,
            upper=constant,
            linear_indices=objective_indices,
            linear_values=objective_values,
            quadratic=tuple(quadratic[OBJECTIVE_INDEX]),
            tree=trees[OBJECTIVE_INDEX],
        ),
        constraints=rows,
    )
