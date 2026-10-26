"""Warm starts for the roots of a term, which `lax.custom_root` defines.

A term with a root solves an equation `r(p, y) = 0` inside every evaluation. The solver is a
pure function of its guess, and Newton's method converges in a step or two from the root of
a nearby point. This module rebuilds the term so that the guess of each root is an input,
and the root it finds is an output. The expression handler keeps the output of the last run
and passes it as the next guess.

Only roots at the top level of a term's jaxpr take part. A root inside a call such as `jit`,
or inside a loop, keeps its own guess. The guess only changes how fast the solver converges
and not the root it converges to, provided the root is unique, which JAXormick's relaxation
of the same root proves on every box. A single Newton step after the solver makes the result
the same to machine precision for every guess.
"""

from collections.abc import Sequence
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var
from jaxormick._implicit import is_custom_root
from jaxtyping import Array, Float

from jaxcip._analysis import dtype_of, shape_of

Operand = TypeVar("Operand")
"""The type of the operands of a root equation, an array or a jaxpr atom."""


def split_operands(
    operands: Sequence[Operand], subtrees: Sequence[jax.tree_util.PyTreeDef]
) -> list[list[Operand]]:
    """Returns the operands of a root split into the leaves of its consecutive subtrees.

    These are the constants of the solver, the guess, the constants of the residual and the
    constants of the tangent solver.
    """
    groups, start = [], 0
    for subtree in subtrees:
        groups.append(list(operands[start : start + subtree.num_leaves]))
        start += subtree.num_leaves
    return groups


def has_state(equation: JaxprEqn) -> bool:
    """Returns whether a root equation takes a warm start: a root of floats without auxiliary data."""
    if not is_custom_root(equation):
        return False
    root = equation.params["_prim"]
    if root.has_aux:
        return False
    guess = split_operands(
        equation.invars, jax.tree_util.treedef_children(root.in_tree)
    )[1]
    return all(jnp.issubdtype(dtype_of(atom), jnp.floating) for atom in guess)


def state_equations(jaxpr: Jaxpr) -> list[JaxprEqn]:
    """Returns the root equations at the top level of a jaxpr that take a warm start."""
    return [equation for equation in jaxpr.eqns if has_state(equation)]


def guess_sizes(equation: JaxprEqn) -> list[int]:
    """Returns the number of entries in each leaf of the guess of a root equation."""
    root = equation.params["_prim"]
    subtrees = jax.tree_util.treedef_children(root.in_tree)
    guess = split_operands(equation.invars, subtrees)[1]
    return [int(np.prod(shape_of(atom))) for atom in guess]


def state_size(jaxpr: Jaxpr) -> int:
    """Returns the number of entries of the warm start of all roots of a jaxpr."""
    return sum(sum(guess_sizes(equation)) for equation in state_equations(jaxpr))


def solve_root(
    equation: JaxprEqn, inputs: Sequence[Array], state: Float[Array, " k"]
) -> list[Array]:
    """Solves a root from the warm start in `state` and returns the leaves of the polished root.

    Entries of `state` that are NaN fall back to the guess written in the term. One Newton step follows the solver, so that two guesses give
    the same root to machine precision. That step has no derivative: the derivative of a
    root comes from the implicit function theorem, which `lax.custom_root` applies.
    """
    root = equation.params["_prim"]
    subtrees = jax.tree_util.treedef_children(root.in_tree)
    solve_consts, guess, residual_consts, tangent_consts = split_operands(
        inputs, subtrees
    )
    warm: list[Array] = []
    offset = 0
    for atom in guess:
        leaf = jnp.asarray(atom)
        part = state[offset : offset + leaf.size].reshape(leaf.shape).astype(leaf.dtype)
        warm.append(jnp.where(jnp.isnan(part), leaf, part))
        offset += leaf.size
    arguments = [
        jax.tree.unflatten(subtree, group)
        for subtree, group in zip(
            subtrees, [solve_consts, warm, residual_consts, tangent_consts], strict=True
        )
    ]
    solution = root(*arguments)
    residual = root.f(arguments[2], solution)
    step = root.tangent_solve(arguments[3], solution, residual)
    polished = jax.tree.map(
        lambda value, change: value - lax.stop_gradient(change), solution, step
    )
    return jax.tree.leaves(polished)


def evaluate_with_state(
    jaxpr: Jaxpr, arguments: Sequence[Any], state: Float[Array, " k"]
) -> tuple[list[Any], Float[Array, " k"]]:
    """Evaluates a jaxpr with the warm start in `state` and returns its outputs and roots.

    The roots are the flattened solutions of the equations that take a warm start, in the
    order of the equations, so they fit `state` as the next warm start.
    """
    environment: dict[Var, Any] = dict(zip(jaxpr.invars, arguments, strict=True))

    def read(atom: Var | Literal) -> Any:
        return atom.val if isinstance(atom, Literal) else environment[atom]

    offset = 0
    roots: list[Array] = []
    for equation in jaxpr.eqns:
        inputs = [read(atom) for atom in equation.invars]
        if has_state(equation):
            size = sum(guess_sizes(equation))
            outputs = solve_root(equation, inputs, state[offset : offset + size])
            roots += [leaf.ravel() for leaf in outputs]
            offset += size
        else:
            parameters = equation.primitive.get_bind_params(equation.params)
            result = equation.primitive.bind(*inputs, **parameters)
            outputs = result if equation.primitive.multiple_results else [result]
        environment.update(zip(equation.outvars, outputs, strict=True))
    flat_roots = jnp.concatenate(roots) if roots else jnp.zeros(0)
    return [read(atom) for atom in jaxpr.outvars], flat_roots
