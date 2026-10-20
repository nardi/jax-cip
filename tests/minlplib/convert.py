"""Conversion of MINLPLib instances into JAX-CIP problems.

Every row becomes a function of the variables it reads. Rows whose functions have the same
expression tree, up to their numbers, form one `BatchedConstraint`, so they share compiled
programs and differ only in their inputs and parameters. Numbers become parameters, except
exponents, which stay in the code because the relaxation of `x ** a` depends on `a`.
"""

import math
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
from osil import Instance, Node, Row, UnsupportedInstance

import jaxcip as jcp

INFINITY = 1e20
"""Bounds at or beyond this magnitude are infinite, as in SCIP."""

Function = Callable[..., jax.Array]

UNARY: dict[str, Function] = {
    "negate": jnp.negative,
    "square": jnp.square,
    "sqrt": jnp.sqrt,
    "exp": jnp.exp,
    "ln": jnp.log,
    "log10": jnp.log10,
    "abs": jnp.abs,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tanh": jnp.tanh,
    "erf": jax.scipy.special.erf,
}
"""OSnL operators of one operand."""

BINARY: dict[str, Function] = {
    "plus": jnp.add,
    "minus": jnp.subtract,
    "times": jnp.multiply,
    "divide": jnp.divide,
    "power": jnp.power,
}
"""OSnL operators of two operands."""

VARIADIC: dict[str, Function] = {
    "sum": lambda *operands: sum(operands[1:], operands[0]),
    "product": lambda *operands: math.prod(operands[1:], start=operands[0]),
    "max": lambda *operands: jnp.max(jnp.stack(operands)),
    "min": lambda *operands: jnp.min(jnp.stack(operands)),
}
"""OSnL operators of any number of operands."""


class Shape(NamedTuple):
    """The structure of a row: its tree with variables and parameters numbered."""

    linear: tuple[int, ...]
    """Local positions of the linear terms."""
    quadratic: tuple[tuple[int, int], ...]
    """Local positions of the factors of the quadratic terms."""
    tree: Any
    """The tree as nested tuples, with exponents kept and other numbers left out."""


class Group(NamedTuple):
    """Rows of one shape."""

    inputs: list[list[int]]
    params: list[list[float]]
    lower: list[float]
    upper: list[float]


def variables_of(node: Node) -> Iterator[int]:
    """Yields the variable indices of a tree in depth-first order."""
    if node.operator == "variable":
        yield node.index
    for item in node.children:
        yield from variables_of(item)


def tree_shape(node: Node, positions: dict[int, int], params: list[float]) -> Any:
    """Returns the shape of a tree and appends its numbers to `params`."""
    match node:
        case Node("number", value=value):
            params.append(value)
            return "number"
        case Node("variable", value=value, index=index):
            params.append(value)
            return ("variable", positions[index])
        case Node("power", (base, Node("number", value=exponent))):
            return ("power", tree_shape(base, positions, params), exponent)
        case Node("signpower", (base, Node("number", value=exponent))):
            return ("signpower", tree_shape(base, positions, params), exponent)
        case Node(operator, children) if (
            operator in UNARY or operator in BINARY or operator in VARIADIC
        ):
            return (
                operator,
                *(tree_shape(item, positions, params) for item in children),
            )
        case _:
            raise UnsupportedInstance(f"operator {node.operator}")


def power(base: jax.Array, exponent: float) -> jax.Array:
    """Returns `base ** exponent` with integer exponents kept integer."""
    if exponent == int(exponent):
        return base ** int(exponent)
    return base**exponent


def evaluate_tree(shape: Any, x: jax.Array, params: Iterator[jax.Array]) -> jax.Array:
    """Evaluates a tree shape, taking its numbers from `params` in depth-first order."""
    match shape:
        case "number":
            return next(params)
        case ("variable", position):
            return next(params) * x[position]
        case ("power", base, float() as exponent):
            return power(evaluate_tree(base, x, params), exponent)
        case ("signpower", base, float() as exponent):
            value = evaluate_tree(base, x, params)
            return jnp.sign(value) * power(jnp.abs(value), exponent)
        case (operator, *children):
            operands = [evaluate_tree(item, x, params) for item in children]
            if operator in UNARY:
                return UNARY[operator](*operands)
            if operator in BINARY:
                return BINARY[operator](*operands)
            return VARIADIC[operator](*operands)
    raise UnsupportedInstance(f"tree shape {shape}")


def row_function(shape: Shape) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Returns the function `f(x, params)` of the rows of a shape."""
    num_linear = len(shape.linear)
    num_quadratic = len(shape.quadratic)

    def function(x: jax.Array, params: jax.Array) -> jax.Array:
        total = jnp.zeros((), x.dtype)
        if num_linear:
            total += jnp.dot(params[:num_linear], x[jnp.array(shape.linear)])
        if num_quadratic:
            first, second = (jnp.array(item) for item in zip(*shape.quadratic))
            coefficients = params[num_linear : num_linear + num_quadratic]
            total += jnp.sum(coefficients * x[first] * x[second])
        if shape.tree is not None:
            rest = params[num_linear + num_quadratic :]
            # Static indices, since iterating over more than 100 entries traces a loop.
            numbers = (rest[index] for index in range(rest.shape[0]))
            total += evaluate_tree(shape.tree, x, numbers)
        return total

    return function


def row_shape(row: Row) -> tuple[Shape, list[int], list[float]]:
    """Returns the shape of a row with its variable indices and parameters."""
    indices = list(row.linear_indices)
    for first, second, _ in row.quadratic:
        indices += [first, second]
    if row.tree is not None:
        indices += list(variables_of(row.tree))
    variables = sorted(set(int(index) for index in indices))
    positions = {index: position for position, index in enumerate(variables)}
    params = [float(value) for value in row.linear_values]
    params += [coefficient for _, _, coefficient in row.quadratic]
    tree = None if row.tree is None else tree_shape(row.tree, positions, params)
    shape = Shape(
        linear=tuple(positions[int(index)] for index in row.linear_indices),
        quadratic=tuple(
            (positions[first], positions[second]) for first, second, _ in row.quadratic
        ),
        tree=tree,
    )
    return shape, variables, params


def finite(bound: float) -> float:
    """Returns a bound with magnitudes from `INFINITY` on made infinite."""
    return math.copysign(math.inf, bound) if abs(bound) >= INFINITY else bound


def convert(instance: Instance) -> jcp.Problem:
    """Returns the problem of an instance over one variable per entry, in OSiL order.

    The variables are a tuple of scalar `Variable`s, so that each has its own domain and
    the flat variable vector keeps the order of the instance.
    """
    variables = tuple(
        jcp.Variable(lower=finite(lower), upper=finite(upper), domain=domain)
        for lower, upper, domain in zip(
            instance.lower, instance.upper, instance.domains, strict=True
        )
    )
    groups: dict[Shape, Group] = {}
    for row in instance.constraints:
        lower, upper = finite(row.lower), finite(row.upper)
        shape, inputs, params = row_shape(row)
        if not inputs or (lower == -math.inf and upper == math.inf):
            continue
        group = groups.setdefault(shape, Group([], [], [], []))
        group.inputs.append(inputs)
        group.params.append(params)
        group.lower.append(lower)
        group.upper.append(upper)
    constraints = tuple(
        jcp.BatchedConstraint(
            function=row_function(shape),
            inputs=np.array(group.inputs),
            params=jnp.array(group.params) if group.params[0] else None,
            lower=np.array(group.lower),
            upper=np.array(group.upper),
        )
        for shape, group in groups.items()
    )
    shape, inputs, params = row_shape(instance.objective)
    objective_function = row_function(shape)
    objective_inputs = jnp.array(inputs, int)
    objective_params = jnp.array(params)
    constant = instance.objective.lower

    def objective(x: tuple[jax.Array, ...], _: None) -> jax.Array:
        flat = jnp.stack(x)
        local = flat[objective_inputs] if inputs else flat[:0]
        return constant + objective_function(local, objective_params)

    return jcp.Problem(
        variables=variables,
        objective=objective,
        sense=instance.sense,
        constraints=constraints,
    )
