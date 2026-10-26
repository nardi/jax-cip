"""Conversion of MINLPLib instances into JAX-CIP problems.

Every row becomes a function of the variables it reads. Rows whose functions have the same
expression tree, up to their numbers, form one `Constraint` over `rows`, so they share compiled
programs and differ only in their inputs and parameters. Numbers become parameters, except
exponents, which stay in the code because the relaxation of `x ** a` depends on `a`.

The monomials of a row are grouped by their exponents, and each group is one gathered
product over all of its monomials. A group of trilinear terms becomes
`w * x[a] * x[b] * x[c]` with index arrays `a`, `b` and `c`, which compiles far faster than
one product per monomial. A variable that appears with an exponent above one is written as
a power, because a repeated variable in a gathered product would relax as two different
factors.
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

REDUCTION_WIDTH = 4
"""Number of factors from which a group of monomials multiplies its factors as a reduction."""

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
    "product": lambda *operands: jnp.prod(jnp.stack(operands)),
    "max": lambda *operands: jnp.max(jnp.stack(operands)),
    "min": lambda *operands: jnp.min(jnp.stack(operands)),
}
"""OSnL operators of any number of operands."""


class MonomialGroup(NamedTuple):
    """Monomials of one row that have the same exponents."""

    exponents: tuple[int, ...]
    """The exponent of each factor, largest first."""
    positions: tuple[tuple[int, ...], ...]
    """For each monomial, the local position of the variable of each factor."""


class Shape(NamedTuple):
    """The structure of a row: its monomials and its tree with variables numbered."""

    linear: tuple[int, ...]
    """Local positions of the linear terms."""
    has_constant: bool
    """Whether the row has a constant, which is the first parameter after the linear ones."""
    groups: tuple[MonomialGroup, ...]
    """Groups of monomials of degree two and higher, which follow the constant."""
    tree: Any
    """The rest as nested tuples, with exponents kept and other numbers left out."""


class Monomial(NamedTuple):
    """A coefficient times variables raised to integer powers."""

    coefficient: float
    powers: dict[int, int]
    """The exponent of each variable index."""

    @property
    def degree(self) -> int:
        """Sum of the exponents."""
        return sum(self.powers.values())

    def times(self, other: "Monomial") -> "Monomial":
        """Returns the product of two monomials."""
        powers = dict(self.powers)
        for index, exponent in other.powers.items():
            powers[index] = powers.get(index, 0) + exponent
        return Monomial(self.coefficient * other.coefficient, powers)

    def raised(self, exponent: int) -> "Monomial":
        """Returns the monomial to a non-negative integer power."""
        return Monomial(
            self.coefficient**exponent,
            {index: power * exponent for index, power in self.powers.items()}
            if exponent
            else {},
        )


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
        case Node("power", (Node("number", value=base), exponent)) if base > 0:
            # JAXormick relaxes `exp` but not `pow` with a variable exponent.
            params.append(math.log(base))
            return ("exponential", tree_shape(exponent, positions, params))
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
        case ("exponential", exponent):
            log_base = next(params)
            return jnp.exp(log_base * evaluate_tree(exponent, x, params))
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


def as_monomial(node: Node) -> Monomial | None:
    """Returns a tree as a monomial, or `None` if it is not a product of powers of variables."""
    match node:
        case Node("number", value=value):
            return Monomial(value, {})
        case Node("variable", value=value, index=index):
            return Monomial(value, {index: 1})
        case Node("negate", (operand,)):
            inner = as_monomial(operand)
            return None if inner is None else Monomial(-inner.coefficient, inner.powers)
        case Node("square", (operand,)):
            inner = as_monomial(operand)
            return None if inner is None else inner.raised(2)
        case Node("power", (base, Node("number", value=exponent))) if (
            exponent == int(exponent) and exponent >= 0
        ):
            inner = as_monomial(base)
            return None if inner is None else inner.raised(int(exponent))
        case Node("divide", (numerator, Node("number", value=divisor))) if divisor != 0:
            inner = as_monomial(numerator)
            if inner is None:
                return None
            return Monomial(inner.coefficient / divisor, inner.powers)
        case Node("times" | "product", operands):
            product = Monomial(1.0, {})
            for operand in operands:
                factor = as_monomial(operand)
                if factor is None:
                    return None
                product = product.times(factor)
            return product
        case _:
            return None


def signed_terms(node: Node, sign: float = 1.0) -> Iterator[tuple[float, Node]]:
    """Yields the terms of the sums at the top of a tree with their signs."""
    match node:
        case Node("sum" | "plus", operands):
            for operand in operands:
                yield from signed_terms(operand, sign)
        case Node("minus", (first, second)):
            yield from signed_terms(first, sign)
            yield from signed_terms(second, -sign)
        case Node("negate", (operand,)):
            yield from signed_terms(operand, -sign)
        case _:
            yield sign, node


class Decomposition(NamedTuple):
    """A row split into its linear terms, constant, higher-degree monomials and the rest."""

    linear: list[tuple[int, float]]
    """Pairs of a variable index and its coefficient."""
    constant: float
    monomials: list[Monomial]
    tree: Node | None
    """The terms that are not monomials, or `None` if there are none."""


def decompose(row: Row) -> Decomposition:
    """Splits a row by degree, taking its monomials out of the top-level sums of its tree."""
    linear = [
        (int(index), float(value))
        for index, value in zip(row.linear_indices, row.linear_values, strict=True)
    ]
    constant = 0.0
    # Multiplying merges the exponents, so a square comes out as one variable to the power 2.
    monomials = [
        Monomial(coefficient, {first: 1}).times(Monomial(1.0, {second: 1}))
        for first, second, coefficient in row.quadratic
    ]
    rest: list[Node] = []
    terms = [] if row.tree is None else list(signed_terms(row.tree))
    for sign, term in terms:
        monomial = as_monomial(term)
        if monomial is None:
            rest.append(term if sign > 0 else Node("negate", (term,)))
            continue
        scaled = Monomial(sign * monomial.coefficient, monomial.powers)
        match scaled.degree:
            case 0:
                constant += scaled.coefficient
            case 1:
                ((index, _),) = scaled.powers.items()
                linear.append((index, scaled.coefficient))
            case _:
                monomials.append(scaled)
    tree = None if not rest else rest[0] if len(rest) == 1 else Node("sum", tuple(rest))
    return Decomposition(linear, constant, monomials, tree)


def group_monomials(
    monomials: list[Monomial], positions: dict[int, int]
) -> tuple[tuple[MonomialGroup, ...], list[float]]:
    """Groups monomials by exponents and returns the groups with their coefficients.

    The factors of a monomial are ordered by exponent, largest first, then by position,
    and the monomials of a group are ordered by their positions, so that rows with the same
    structure get equal groups.
    """
    grouped: dict[tuple[int, ...], list[tuple[tuple[int, ...], float]]] = {}
    for monomial in monomials:
        factors = sorted(
            (-exponent, positions[index]) for index, exponent in monomial.powers.items()
        )
        exponents = tuple(-exponent for exponent, _ in factors)
        where = tuple(position for _, position in factors)
        grouped.setdefault(exponents, []).append((where, monomial.coefficient))
    groups = []
    coefficients: list[float] = []
    for exponents in sorted(grouped, reverse=True):
        members = sorted(grouped[exponents])
        groups.append(MonomialGroup(exponents, tuple(where for where, _ in members)))
        coefficients += [coefficient for _, coefficient in members]
    return tuple(groups), coefficients


def row_function(shape: Shape) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Returns the function `f(x, params)` of the rows of a shape."""
    num_linear = len(shape.linear)
    num_constant = int(shape.has_constant)
    num_monomials = sum(len(group.positions) for group in shape.groups)
    start = num_linear + num_constant

    def function(x: jax.Array, params: jax.Array) -> jax.Array:
        total = jnp.zeros((), x.dtype)
        if num_linear:
            total += jnp.dot(params[:num_linear], x[jnp.array(shape.linear)])
        if num_constant:
            total += params[num_linear]
        offset = start
        for group in shape.groups:
            count = len(group.positions)
            gathered = np.array(group.positions)
            columns = [
                x[gathered[:, column]] ** exponent
                if exponent > 1
                else x[gathered[:, column]]
                for column, exponent in enumerate(group.exponents)
            ]
            coefficients = params[offset : offset + count]
            if len(columns) < REDUCTION_WIDTH:
                term = coefficients
                for column in columns:
                    term = term * column
            else:
                # A chain of products compiles slowly, so wide products are one reduction,
                # which JAXormick relaxes in logarithmically many levels.
                term = coefficients * jnp.prod(jnp.stack(columns, axis=1), axis=1)
            total += jnp.sum(term)
            offset += count
        if shape.tree is not None:
            rest = params[start + num_monomials :]
            # Static indices, since iterating over more than 100 entries traces a loop.
            numbers = (rest[index] for index in range(rest.shape[0]))
            total += evaluate_tree(shape.tree, x, numbers)
        return total

    return function


def row_shape(row: Row) -> tuple[Shape, list[int], list[float]]:
    """Returns the shape of a row with its variable indices and parameters."""
    parts = decompose(row)
    indices = [index for index, _ in parts.linear]
    for monomial in parts.monomials:
        indices += list(monomial.powers)
    if parts.tree is not None:
        indices += list(variables_of(parts.tree))
    variables = sorted(set(int(index) for index in indices))
    positions = {index: position for position, index in enumerate(variables)}
    params = [value for _, value in parts.linear]
    has_constant = parts.constant != 0.0
    if has_constant:
        params.append(parts.constant)
    groups, coefficients = group_monomials(parts.monomials, positions)
    params += coefficients
    tree = None if parts.tree is None else tree_shape(parts.tree, positions, params)
    shape = Shape(
        linear=tuple(positions[index] for index, _ in parts.linear),
        has_constant=has_constant,
        groups=groups,
        tree=tree,
    )
    return shape, variables, params


def finite(bound: float) -> float:
    """Returns a bound with magnitudes from `INFINITY` on made infinite."""
    return math.copysign(math.inf, bound) if abs(bound) >= INFINITY else bound


def group_constraint(shape: Shape, group: Group) -> jcp.Constraint:
    """Returns the constraint of the rows of one shape, as `rows` of the variable vector."""
    evaluate = jcp.rows(
        row_function(shape),
        np.array(group.inputs),
        jnp.array(group.params) if group.params[0] else None,
    )
    return jcp.Constraint(
        lambda x, _: evaluate(jnp.stack(x)),
        lower=np.array(group.lower),
        upper=np.array(group.upper),
    )


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
        group_constraint(shape, group) for shape, group in groups.items()
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
