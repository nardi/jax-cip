"""Tests of the conversion of OSiL rows into functions, without any instance files."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from convert import MonomialGroup, as_monomial, row_function, row_shape
from osil import Node, Row, UnsupportedInstance


def variable(index: int, coefficient: float = 1.0) -> Node:
    """Returns a variable node."""
    return Node("variable", value=coefficient, index=index)


def number(value: float) -> Node:
    """Returns a number node."""
    return Node("number", value=value)


def reference(node: Node, x: np.ndarray) -> float:
    """Evaluates a tree directly, node by node."""
    operands = [reference(item, x) for item in node.children]
    match node.operator:
        case "number":
            return node.value
        case "variable":
            return node.value * x[node.index]
        case "plus" | "sum":
            return sum(operands)
        case "minus":
            return operands[0] - operands[1]
        case "negate":
            return -operands[0]
        case "times" | "product":
            return math.prod(operands)
        case "divide":
            return operands[0] / operands[1]
        case "power":
            return operands[0] ** operands[1]
        case "square":
            return operands[0] ** 2
        case "exp":
            return math.exp(operands[0])
    raise AssertionError(node.operator)


def row_of(
    tree: Node | None, quadratic: tuple[tuple[int, int, float], ...] = ()
) -> Row:
    """Returns a row with a linear part on variables 0 and 5."""
    return Row(
        lower=-math.inf,
        upper=0.0,
        linear_indices=np.array([5, 0]),
        linear_values=np.array([2.0, -3.0]),
        quadratic=quadratic,
        tree=tree,
    )


def value_of(row: Row, x: np.ndarray) -> float:
    """Evaluates a row directly from its parts."""
    total = float(row.linear_values @ x[row.linear_indices])
    total += sum(coefficient * x[a] * x[b] for a, b, coefficient in row.quadratic)
    return total + (0.0 if row.tree is None else reference(row.tree, x))


TREES = {
    "trilinear": Node(
        "sum",
        (
            Node("times", (variable(1, 2.0), variable(2), variable(3))),
            Node("times", (number(-1.5), variable(4), variable(2), variable(0))),
            Node("negate", (Node("product", (variable(1), variable(3), variable(5))),)),
        ),
    ),
    "powers": Node(
        "plus",
        (
            Node("power", (variable(1), number(3.0))),
            Node("times", (Node("square", (variable(2),)), variable(3), number(0.5))),
        ),
    ),
    "mixed": Node(
        "minus",
        (
            Node(
                "sum",
                (
                    Node("times", (variable(1), variable(2))),
                    Node("exp", (Node("times", (variable(3), variable(4))),)),
                    number(4.0),
                    variable(4, 3.0),
                ),
            ),
            Node("divide", (Node("times", (variable(1), variable(1))), number(4.0))),
        ),
    ),
    "wide_product": Node("product", tuple(variable(index) for index in range(1, 6))),
    "wide_powers": Node(
        "plus",
        (
            Node("product", tuple(variable(index, 1.5) for index in (1, 2, 3, 4))),
            Node(
                "times",
                (
                    Node("square", (variable(1),)),
                    Node("product", (variable(2), variable(3), variable(4))),
                ),
            ),
        ),
    ),
    "nonmonomial_product": Node(
        "times",
        (
            Node("times", (Node("exp", (variable(1),)), variable(2))),
            variable(3),
        ),
    ),
}
"""Trees that cover monomials, powers, constants, and terms that stay in the tree."""


@pytest.mark.parametrize("name", sorted(TREES))
@pytest.mark.parametrize("quadratic", [(), ((1, 2, 0.5), (3, 3, -2.0), (5, 0, 1.0))])
def test_row_function_matches_the_tree(
    name: str, quadratic: tuple[tuple[int, int, float], ...]
) -> None:
    """Checks that the grouped function equals the direct evaluation of the row."""
    row = row_of(TREES[name], quadratic)
    shape, variables, params = row_shape(row)
    function = row_function(shape)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.uniform(0.3, 1.7, size=6)
        local = jnp.asarray(x[variables])
        value = function(local, jnp.asarray(params))
        assert float(value) == pytest.approx(value_of(row, x), rel=1e-12)


def test_monomials_are_grouped_by_exponents() -> None:
    """Checks the groups of a row with trilinear terms, a square and a bilinear term."""
    tree = Node(
        "sum",
        (
            Node("times", (variable(3), variable(1), variable(2))),
            Node("times", (variable(4), variable(1), variable(2))),
            Node("square", (variable(1),)),
            Node("times", (Node("square", (variable(2),)), variable(5))),
        ),
    )
    shape, variables, params = row_shape(row_of(tree, ((0, 4, 1.0),)))
    assert variables == [0, 1, 2, 3, 4, 5]
    assert shape.groups == (
        MonomialGroup((2, 1), ((2, 5),)),
        MonomialGroup((2,), ((1,),)),
        MonomialGroup((1, 1, 1), ((1, 2, 3), (1, 2, 4))),
        MonomialGroup((1, 1), ((0, 4),)),
    )
    assert shape.tree is None
    assert not shape.has_constant


def test_repeated_variables_are_written_as_powers() -> None:
    """Checks that no group gathers one variable twice."""
    tree = Node("times", (variable(1), variable(1), variable(2)))
    shape, _, _ = row_shape(row_of(tree, ((1, 1, 1.0),)))
    for group in shape.groups:
        for positions in group.positions:
            assert len(set(positions)) == len(positions)
    assert shape.groups[0].exponents == (2, 1)


def test_rows_of_equal_structure_share_a_shape() -> None:
    """Checks that rows that differ only in their numbers and monomial order share a shape."""
    first = Node(
        "sum",
        (
            Node("times", (number(2.0), variable(1), variable(2), variable(3))),
            Node("times", (number(5.0), variable(1), variable(2), variable(4))),
        ),
    )
    second = Node(
        "sum",
        (
            Node("times", (variable(4), number(-1.0), variable(2), variable(1))),
            Node("times", (variable(3), variable(2), variable(1), number(0.25))),
        ),
    )
    assert row_shape(row_of(first))[0] == row_shape(row_of(second))[0]


def test_as_monomial_rejects_other_trees() -> None:
    """Checks the trees that are not monomials."""
    assert as_monomial(Node("exp", (variable(0),))) is None
    assert as_monomial(Node("power", (variable(0), number(-1.0)))) is None
    assert as_monomial(Node("power", (variable(0), number(0.5)))) is None
    assert as_monomial(Node("divide", (variable(0), variable(1)))) is None
    assert (
        as_monomial(Node("times", (variable(0), Node("exp", (variable(1),))))) is None
    )
    assert as_monomial(Node("negate", (Node("exp", (variable(0),)),))) is None
    assert as_monomial(Node("square", (Node("exp", (variable(0),)),))) is None
    assert (
        as_monomial(Node("divide", (Node("exp", (variable(0),)), number(2.0)))) is None
    )
    zeroth = as_monomial(Node("power", (variable(0, 3.0), number(0.0))))
    assert zeroth is not None
    assert (zeroth.coefficient, zeroth.degree) == (1.0, 0)


def test_unsupported_operator() -> None:
    """Checks that an operator without a translation still raises an error."""
    row = row_of(Node("sinh", (variable(1),)))
    with pytest.raises(UnsupportedInstance, match="sinh"):
        row_shape(row)


def test_jit_of_grouped_function() -> None:
    """Checks that the grouped function can be traced with `jax.jit`."""
    shape, variables, params = row_shape(row_of(TREES["trilinear"]))
    function = jax.jit(row_function(shape))
    x = jnp.linspace(0.5, 1.5, len(variables))
    assert jnp.isfinite(function(x, jnp.asarray(params)))
