"""The programs of a term, from its jaxpr to StableHLO.

A term becomes a scalar function `f(x, params)` of two flat vectors: the entries of the
variables it reads and its parameters, which include the constants it closes over. Six
programs of `f` give the expression handler values, derivatives, relaxations and bounds.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jaxormick as jmk
from jax.extend.core import Jaxpr
from jaxtyping import Array, Float

from jaxcip._analysis import (
    Origin,
    Term,
    combine_outputs,
    dtype_of,
    shape_of,
    sum_outputs,
)
from jaxcip._options import RelaxationOptions
from jaxcip._roots import evaluate_with_state, state_size
from jaxcip._types import Program

TermFunction = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, ""]]
"""A scalar function `f(x, params)` of two flat vectors."""

StatefulFunction = Callable[
    [Float[Array, " n"], Float[Array, " m"], Float[Array, " k"]],
    tuple[Float[Array, ""], Float[Array, " k"]],
]
"""A scalar function `f(x, params, state)` that also returns the roots it found.

The warm start `state` holds a guess for each root of the term, and NaN entries use the
guess written in the term. The roots come back in the same layout.
"""


class LoweredProgram(NamedTuple):
    """A program as StableHLO."""

    code: bytes
    """Portable StableHLO bytecode, which the PJRT plugin compiles."""
    text: str
    """The module as text, from which the executable cache key is computed."""


class LoweredPrograms(NamedTuple):
    """The lowered programs of one term, in the order of `Program`."""

    value_and_gradient: LoweredProgram
    hessian_vector: LoweredProgram | None
    convex: LoweredProgram
    concave: LoweredProgram
    interval: LoweredProgram
    gradient_interval: LoweredProgram | None


class ProgramOptions(NamedTuple):
    """What to build for each term."""

    relaxation: RelaxationOptions
    hessian: bool
    """Whether to build the Hessian-vector program."""
    gradient_interval: bool
    """Whether to build the gradient interval program, if the term is small enough."""
    gradient_interval_limit: int
    """Largest term, in jaxpr equations, that gets the gradient interval program."""


class TermInput(NamedTuple):
    """Shape and type of one input of a term's jaxpr."""

    shape: tuple[int, ...]
    dtype: jnp.dtype
    is_variable: bool

    @property
    def size(self) -> int:
        """Number of entries the input takes from its flat vector."""
        return math.prod(self.shape)


class TermSignature(NamedTuple):
    """The inputs of a term and the sizes of its flat vectors."""

    inputs: tuple[TermInput, ...]
    shape: tuple[int, ...]
    """Shape of the term's value. Each entry becomes one SCIP expression."""
    num_inputs: int
    """Length of the flat variable vector."""
    num_params: int
    """Length of the flat parameter vector, at least one."""
    state_size: int = 0
    """Length of the warm start of the term's roots, or 0 if it has none."""

    @property
    def num_entries(self) -> int:
        """Number of entries of the term's value."""
        return math.prod(self.shape)


def term_signature(term: Term, shape: tuple[int, ...]) -> TermSignature:
    """Returns the signature of a term with an output of the given shape."""
    inputs = tuple(
        TermInput(
            shape=shape_of(var),
            dtype=dtype_of(var),
            is_variable=ref.origin is Origin.VARIABLE,
        )
        for ref, var in zip(term.inputs, term.jaxpr.invars, strict=True)
    )
    num_inputs = sum(item.size for item in inputs if item.is_variable)
    num_params = sum(item.size for item in inputs if not item.is_variable)
    num_params += int(math.prod(shape) > 1)
    return TermSignature(
        inputs, shape, num_inputs, max(num_params, 1), state_size(term.jaxpr)
    )


def term_arguments(
    signature: TermSignature, x: Float[Array, " n"], params: Float[Array, " m"]
) -> tuple[list[Array], int]:
    """Returns the arguments of a term's jaxpr from the flat vectors and the parameters used."""
    arguments: list[Array] = []
    variable_offset = 0
    param_offset = 0
    for item in signature.inputs:
        if item.is_variable:
            part = x[variable_offset : variable_offset + item.size]
            variable_offset += item.size
        else:
            part = params[param_offset : param_offset + item.size]
            param_offset += item.size
        arguments.append(part.reshape(item.shape).astype(item.dtype))
    return arguments, param_offset


def select_entry(
    total: Array, signature: TermSignature, params: Float[Array, " m"], offset: int
) -> Float[Array, ""]:
    """Returns the entry of a term's value that the parameter at `offset` selects, if any."""
    if signature.num_entries == 1:
        return total.reshape(())
    index = params[offset].astype(jnp.int32)
    return jax.lax.dynamic_index_in_dim(total, index, keepdims=False)


def term_function(term: Term, signature: TermSignature) -> TermFunction:
    """Returns the scalar function of one entry of a term.

    If the term has several entries, the last parameter selects the entry.
    """

    def evaluate(x: Float[Array, " n"], params: Float[Array, " m"]) -> Float[Array, ""]:
        arguments, offset = term_arguments(signature, x, params)
        total = sum_outputs(term, signature.shape, arguments).ravel()
        return select_entry(total, signature, params, offset)

    return evaluate


def stateful_term_function(term: Term, signature: TermSignature) -> StatefulFunction:
    """Returns the scalar function of one entry of a term whose roots take a warm start."""

    def evaluate(
        x: Float[Array, " n"], params: Float[Array, " m"], state: Float[Array, " k"]
    ) -> tuple[Float[Array, ""], Float[Array, " k"]]:
        arguments, offset = term_arguments(signature, x, params)
        outputs, roots = evaluate_with_state(term.jaxpr, arguments, state)
        total = combine_outputs(term, signature.shape, outputs).ravel()
        return select_entry(total, signature, params, offset), roots

    return evaluate


def program_functions(
    function: TermFunction,
    options: RelaxationOptions,
    stateful: StatefulFunction | None = None,
) -> tuple[Callable[..., Any], ...]:
    """Returns the six programs of a scalar function in the order of `Program`.

    With a `stateful` version of the function, the gradient and Hessian-vector programs take
    the warm start of its roots as their last input, and the gradient program also returns
    the roots it found.
    """
    mode = options.mode.value

    def hessian_vector(
        x: Float[Array, " n"], direction: Float[Array, " n"], params: Float[Array, " m"]
    ) -> Float[Array, " n"]:
        gradient = jax.grad(function)
        return jax.jvp(lambda y: gradient(y, params), (x,), (direction,))[1]

    def stateful_hessian_vector(
        x: Float[Array, " n"],
        direction: Float[Array, " n"],
        params: Float[Array, " m"],
        state: Float[Array, " k"],
    ) -> Float[Array, " n"]:
        assert stateful is not None
        gradient = jax.grad(lambda y: stateful(y, params, state)[0])
        return jax.jvp(gradient, (x,), (direction,))[1]

    def stateful_value_and_gradient(
        x: Float[Array, " n"], params: Float[Array, " m"], state: Float[Array, " k"]
    ) -> tuple[Float[Array, ""], Float[Array, " n"], Float[Array, " k"]]:
        assert stateful is not None
        (value, roots), gradient = jax.value_and_grad(stateful, has_aux=True)(
            x, params, state
        )
        return value, gradient, roots

    return (
        jax.value_and_grad(function)
        if stateful is None
        else stateful_value_and_gradient,
        hessian_vector if stateful is None else stateful_hessian_vector,
        jmk.convex_side(
            function,
            mode,
            alpha_bb=options.alpha_bb,
            tighten_bounds=options.tighten_bounds,
        ),
        jmk.concave_side(
            function,
            mode,
            alpha_bb=options.alpha_bb,
            tighten_bounds=options.tighten_bounds,
        ),
        jmk.interval(function, tighten_bounds=options.tighten_bounds),
        jmk.interval(jax.grad(function), tighten_bounds=options.tighten_bounds),
    )


def program_shapes(
    program: Program, num_inputs: int, num_params: int, state_size: int = 0
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Returns the shapes of the arguments of a program."""
    vector = jax.ShapeDtypeStruct((num_inputs,), jnp.float64)
    params = jax.ShapeDtypeStruct((num_params,), jnp.float64)
    state = (jax.ShapeDtypeStruct((state_size,), jnp.float64),) if state_size else ()
    match program:
        case Program.VALUE_AND_GRADIENT:
            return (vector, params, *state)
        case Program.HESSIAN_VECTOR:
            return (vector, vector, params, *state)
        case Program.CONVEX | Program.CONCAVE:
            return (vector, vector, vector, params)
        case Program.INTERVAL | Program.GRADIENT_INTERVAL:
            return (vector, vector, params)
        case _:
            raise AssertionError(f"Unknown program {program}.")


def lower(
    function: Callable[..., Any], shapes: Sequence[jax.ShapeDtypeStruct]
) -> LoweredProgram:
    """Lowers a function to StableHLO for the CPU platform.

    Unused arguments stay in the signature, since the handler passes every input of a program.
    """
    exported = jax.export.export(
        jax.jit(function, keep_unused=True), platforms=["cpu"]
    )(*shapes)
    return LoweredProgram(
        code=bytes(exported.mlir_module_serialized), text=exported.mlir_module()
    )


def is_wanted(program: Program, options: ProgramOptions, jaxpr: Jaxpr) -> str | None:
    """Returns why an optional program is left out, or `None` if it is built."""
    match program:
        case Program.HESSIAN_VECTOR if not options.hessian:
            return "Hessians are switched off"
        case Program.GRADIENT_INTERVAL if not options.gradient_interval:
            return "monotonicity is switched off"
        case Program.GRADIENT_INTERVAL if (
            len(jaxpr.eqns) > options.gradient_interval_limit
        ):
            return f"{len(jaxpr.eqns)} equations exceed the limit"
    return None


OPTIONAL_PROGRAMS = frozenset({Program.HESSIAN_VECTOR, Program.GRADIENT_INTERVAL})
"""Programs without which the handler still works, leaving their callbacks to SCIP."""


def lower_programs(
    term: Term, signature: TermSignature, options: ProgramOptions
) -> tuple[LoweredPrograms, tuple[str, ...]]:
    """Lowers the programs of a term and returns them with the reasons for left out ones.

    An optional program that cannot be lowered is left out. A required one raises.
    """
    function = term_function(term, signature)
    stateful = stateful_term_function(term, signature) if signature.state_size else None
    lowered: list[LoweredProgram | None] = []
    skipped: list[str] = []
    for program, program_function in zip(
        Program, program_functions(function, options.relaxation, stateful), strict=True
    ):
        shapes = program_shapes(
            program, signature.num_inputs, signature.num_params, signature.state_size
        )
        reason = is_wanted(program, options, term.jaxpr)
        if reason is None:
            try:
                lowered.append(lower(program_function, shapes))
                continue
            except Exception as error:
                if program not in OPTIONAL_PROGRAMS:
                    raise
                reason = f"{type(error).__name__}: {error}"
        lowered.append(None)
        skipped.append(f"{program.name}: {reason}")
    return LoweredPrograms._make(lowered), tuple(skipped)
