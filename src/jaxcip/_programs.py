"""The programs of a term, from its jaxpr to StableHLO.

A term becomes a scalar function `f(x, params)` of two flat vectors: the entries of the
variables it reads and its parameters, which include the constants it closes over. Seven
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

from jaxcip._analysis import Origin, Term, sum_outputs
from jaxcip._options import RelaxationOptions
from jaxcip._types import Program

TermFunction = Callable[[Float[Array, " n"], Float[Array, " m"]], Float[Array, ""]]
"""A scalar function `f(x, params)` of two flat vectors."""


class LoweredProgram(NamedTuple):
    """A program as StableHLO."""

    code: bytes
    """Portable StableHLO bytecode, which the PJRT plugin compiles."""
    text: str
    """The module as text, from which the executable cache key is computed."""


class LoweredPrograms(NamedTuple):
    """The lowered programs of one term, in the order of `Program`."""

    value: LoweredProgram
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
    """Shape and type of one input of a term's jaxpr, and the entries the term reads."""

    shape: tuple[int, ...]
    dtype: jnp.dtype
    selection: tuple[int, ...] | None
    """Flat indices of the entries read, or `None` for all of them."""
    is_variable: bool

    @property
    def size(self) -> int:
        """Number of entries the input takes from its flat vector."""
        if self.selection is not None:
            return len(self.selection)
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

    @property
    def num_entries(self) -> int:
        """Number of entries of the term's value."""
        return math.prod(self.shape)


def term_signature(
    term: Term, shape: tuple[int, ...], avals: Sequence[jax.ShapeDtypeStruct]
) -> TermSignature:
    """Returns the signature of a term whose jaxpr inputs have the given shapes."""
    inputs = tuple(
        TermInput(
            shape=tuple(aval.shape),
            dtype=aval.dtype,
            selection=selection,
            is_variable=ref.origin is Origin.VARIABLE,
        )
        for ref, aval, selection in zip(
            term.inputs, avals, term.selections, strict=True
        )
    )
    num_inputs = sum(item.size for item in inputs if item.is_variable)
    num_params = sum(item.size for item in inputs if not item.is_variable)
    num_params += int(math.prod(shape) > 1)
    return TermSignature(inputs, shape, num_inputs, max(num_params, 1))


def term_function(term: Term, signature: TermSignature) -> TermFunction:
    """Returns the scalar function of one entry of a term.

    If the term has several entries, the last parameter selects the entry.
    """

    def evaluate(x: Float[Array, " n"], params: Float[Array, " m"]) -> Float[Array, ""]:
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
            if item.selection is not None:
                full = jnp.zeros(math.prod(item.shape), x.dtype)
                part = full.at[jnp.asarray(item.selection)].set(part)
            arguments.append(part.reshape(item.shape).astype(item.dtype))
        total = sum_outputs(term, signature.shape, arguments).ravel()
        if signature.num_entries == 1:
            return total.reshape(())
        index = params[param_offset].astype(jnp.int32)
        return jax.lax.dynamic_index_in_dim(total, index, keepdims=False)

    return evaluate


def program_functions(
    function: TermFunction, options: RelaxationOptions
) -> tuple[Callable[..., Any], ...]:
    """Returns the seven programs of a scalar function in the order of `Program`."""
    mode = options.mode.value

    def hessian_vector(
        x: Float[Array, " n"], direction: Float[Array, " n"], params: Float[Array, " m"]
    ) -> Float[Array, " n"]:
        gradient = jax.grad(function)
        return jax.jvp(lambda y: gradient(y, params), (x,), (direction,))[1]

    return (
        function,
        jax.value_and_grad(function),
        hessian_vector,
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
    program: Program, num_inputs: int, num_params: int
) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Returns the shapes of the arguments of a program."""
    vector = jax.ShapeDtypeStruct((num_inputs,), jnp.float64)
    params = jax.ShapeDtypeStruct((num_params,), jnp.float64)
    match program:
        case Program.VALUE | Program.VALUE_AND_GRADIENT:
            return (vector, params)
        case Program.HESSIAN_VECTOR:
            return (vector, vector, params)
        case Program.CONVEX | Program.CONCAVE:
            return (vector, vector, vector, params)
        case Program.INTERVAL | Program.GRADIENT_INTERVAL:
            return (vector, vector, params)


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
    lowered: list[LoweredProgram | None] = []
    skipped: list[str] = []
    for program, program_function in zip(
        Program, program_functions(function, options.relaxation), strict=True
    ):
        shapes = program_shapes(program, signature.num_inputs, signature.num_params)
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
