"""SCIP expression handler for JAX functions, executed through the PJRT runtime.

Each expression of the `jax` handler stands for one scalar function `f(x, params)` of its
children `x`, with a constant parameter vector `params` per expression. Expressions with the
same structure share one `Function`, which holds the executables of six programs: the value
with its gradient, a Hessian-vector product, the convex and the concave relaxation with their
subgradients, interval bounds, and interval bounds of the gradient.

All callbacks run without the GIL. Each expression keeps small caches of recent inputs and
results, because SCIP often asks for the same value, bounds or estimator again.
"""

import cython
import numpy as np
from cython.cimports.jaxcip._runtime import (
    MAX_OUTPUTS,
    Executable,
    Input,
    ProgramSpec,
    alias_buffer,
    aligned_doubles,
    destroy_buffer,
    execute,
    now,
)
from cython.cimports.jaxcip._scip import (
    SCIP,
    SCIP_EXPR,
    SCIP_EXPR_MAXINITESTIMATES,
    SCIP_EXPRDATA,
    SCIP_EXPRHDLR,
    SCIP_INTERVAL,
    SCIP_INTERVAL_INFINITY,
    SCIP_INVALID,
    SCIP_MONOTONE,
    SCIP_MONOTONE_CONST,
    SCIP_MONOTONE_DEC,
    SCIP_MONOTONE_INC,
    SCIP_MONOTONE_UNKNOWN,
    SCIP_NOMEMORY,
    SCIP_OKAY,
    SCIP_RETCODE,
    SCIP_SOL,
    SYM_EXPRDATA,
    ExprIntevalVar,
    SCIP_Bool,
    SCIP_Real,
    SCIPcompareExpr,
    SCIPcreateExpr,
    SCIPevalExprActivity,
    SCIPexprGetActivity,
    SCIPexprGetChildren,
    SCIPexprGetData,
    SCIPexprGetDot,
    SCIPexprGetEvalValue,
    SCIPexprGetNChildren,
    SCIPexprhdlrSetCompare,
    SCIPexprhdlrSetCopyFreeData,
    SCIPexprhdlrSetCopyFreeHdlr,
    SCIPexprhdlrSetDiff,
    SCIPexprhdlrSetEstimate,
    SCIPexprhdlrSetGetSymdata,
    SCIPexprhdlrSetHash,
    SCIPexprhdlrSetIntEval,
    SCIPexprhdlrSetMonotonicity,
    SCIPexprhdlrSetReverseProp,
    SCIPexprSetData,
    SCIPfeastol,
    SCIPfindExprhdlr,
    SCIPincludeExprhdlr,
    SCIPisInfinity,
    allocExprArray,
    allocRealArray,
    allocSymData,
)
from cython.cimports.libc.math import fabs, isfinite, isnan
from cython.cimports.libc.stdint import uintptr_t
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcmp, memcpy, memset

from jaxcip._options import HandlerOptions
from jaxcip._runtime import raise_last_error, require_runtime
from jaxcip._types import (
    CallbackStatistics,
    Estimator,
    HandlerStatistics,
    Interval,
    Program,
    Programs,
)

HANDLER_NAME = cython.declare(cython.pointer[cython.const[cython.char]], b"jax")
"""Name of the expression handler in SCIP."""

MAX_ESTIMATOR_COEFFICIENT = cython.declare(cython.double, 1e9)
"""Largest absolute estimator coefficient passed to SCIP, beyond which the LP gets unstable."""

PROGRAM_SLOTS: tuple[tuple[int, ...], ...] = (
    (SLOT_X, SLOT_PARAMS),
    (SLOT_X, SLOT_DIRECTION, SLOT_PARAMS),
    (SLOT_X, SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_X, SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
)
"""Inputs of each program in the order of `Program`, as the exported functions take them."""


def output_sizes(program: Program, num_inputs: int) -> tuple[int, ...]:
    """Returns the number of doubles in each output of a program."""
    match program:
        case Program.VALUE_AND_GRADIENT | Program.CONVEX | Program.CONCAVE:
            return (1, num_inputs)
        case Program.HESSIAN_VECTOR:
            return (num_inputs,)
        case Program.INTERVAL:
            return (1, 1)
        case Program.GRADIENT_INTERVAL:
            return (num_inputs, num_inputs)


@cython.cfunc
def switches_from(options: object) -> Switches:
    """Returns the C switches of a `HandlerOptions`."""
    handler_options: HandlerOptions = options
    switches: Switches
    switches.caches = handler_options.caches
    switches.init_estimates = handler_options.init_estimates
    switches.monotonicity = handler_options.monotonicity
    switches.global_estimators = handler_options.global_estimators
    switches.reverse_propagation = handler_options.reverse_propagation
    switches.hessian = handler_options.hessian
    switches.zero_copy = handler_options.zero_copy
    return switches


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def run(
    data: cython.pointer[ExprData],
    program: cython.int,
    first: cython.p_double,
    second: cython.p_double,
    callback: cython.int,
) -> cython.int:
    """Executes a program on the current host inputs and writes its outputs, returning -1 on error.

    The caller fills the function's host inputs before the call. The first output goes to
    `first` and the second, if any, to `second`.
    """
    function: cython.pointer[Function] = data.function
    spec: cython.pointer[ProgramSpec] = cython.address(function.programs[program])
    if spec.handle == cython.NULL:
        return -1
    inputs: Input[MAX_INPUTS]
    index: cython.int
    slot: cython.int
    for index in range(spec.num_inputs):
        slot = function.slots[program][index]
        if slot == SLOT_PARAMS:
            inputs[index] = data.params
        else:
            inputs[index] = function.inputs[slot]
    outputs: cython.p_double[MAX_OUTPUTS]
    outputs[0] = first
    outputs[1] = second
    start: cython.double = now()
    status: cython.int = execute(spec, inputs, outputs, data.switches.zero_copy)
    function.statistics.runs[callback] += 1
    function.statistics.seconds[callback] += now() - start
    return status


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def same(
    first: cython.p_double, second: cython.p_double, count: cython.int
) -> cython.bint:
    """Returns whether two arrays of doubles are bitwise equal."""
    return memcmp(first, second, count * cython.sizeof(cython.double)) == 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def new_expr_data(
    function: cython.pointer[Function],
    params: cython.pointer[cython.const[cython.double]],
    switches: Switches,
) -> cython.pointer[ExprData]:
    """Returns new expression data with a copy of `params` and empty caches, or NULL on failure."""
    n: cython.int = function.n
    m: cython.int = function.m
    data: cython.pointer[ExprData] = cython.cast(
        cython.pointer[ExprData], malloc(cython.sizeof(ExprData))
    )
    if data == cython.NULL:
        return cython.NULL  # pragma: no cover
    memset(data, 0, cython.sizeof(ExprData))
    data.function = function
    data.switches = switches
    data.params.host = aligned_doubles(m)
    data.params.size = m
    memcpy(data.params.host, params, m * cython.sizeof(cython.double))
    data.params.buffer = alias_buffer(data.params.host, m)

    # All caches live in one block, laid out one after another.
    total: cython.size_t = (
        VALUE_CACHE_SIZE * (n + 1)
        + 2 * n
        + 3 * n
        + BOUNDS_CACHE_SIZE * (2 * n + 2)
        + 2 * ESTIMATE_CACHE_SIZE * (4 * n + 1)
        + 4 * n
    )
    cursor: cython.p_double = cython.cast(
        cython.p_double, malloc(total * cython.sizeof(cython.double))
    )
    data.cache_block = cursor
    data.value_keys = cursor
    cursor += VALUE_CACHE_SIZE * n
    data.value_results = cursor
    cursor += VALUE_CACHE_SIZE
    data.gradient_key = cursor
    cursor += n
    data.gradient = cursor
    cursor += n
    data.hessian_key = cursor
    cursor += 2 * n
    data.hessian = cursor
    cursor += n
    data.bounds_keys = cursor
    cursor += BOUNDS_CACHE_SIZE * 2 * n
    data.bounds_results = cursor
    cursor += BOUNDS_CACHE_SIZE * 2
    side: cython.int
    for side in range(2):
        data.estimate_keys[side] = cursor
        cursor += ESTIMATE_CACHE_SIZE * 3 * n
        data.estimate_results[side] = cursor
        cursor += ESTIMATE_CACHE_SIZE * (n + 1)
    data.monotonicity_key = cursor
    cursor += 2 * n
    data.monotonicity_result = cursor
    return data


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def free_expr_data(data: cython.pointer[ExprData]) -> cython.void:
    """Frees expression data and its parameter buffer."""
    destroy_buffer(data.params.buffer)
    free(data.params.host)
    free(data.cache_block)
    free(data)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def get_data(expr: cython.pointer[SCIP_EXPR]) -> cython.pointer[ExprData]:
    """Returns the handler's data of an expression."""
    return cython.cast(cython.pointer[ExprData], SCIPexprGetData(expr))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_child_values(
    expr: cython.pointer[SCIP_EXPR], destination: cython.p_double
) -> cython.bint:
    """Copies the children's evaluation values and returns whether all are valid."""
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        destination[index] = SCIPexprGetEvalValue(children[index])
        # The children are variable expressions, whose values are never invalid.
        if destination[index] == SCIP_INVALID:
            return False  # pragma: no cover
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def eval_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    val: cython.pointer[SCIP_Real],
    sol: cython.pointer[SCIP_SOL],
) -> SCIP_RETCODE:
    """Evaluates the function at the children's values."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    n: cython.int = function.n
    index: cython.int
    function.statistics.calls[CALLBACK_EVAL] += 1
    val[0] = SCIP_INVALID
    if not read_child_values(expr, x):
        return SCIP_OKAY  # pragma: no cover
    if data.switches.caches:
        for index in range(data.value_count):
            if same(x, data.value_keys + index * n, n):
                function.statistics.hits[CALLBACK_EVAL] += 1
                val[0] = data.value_results[index]
                return SCIP_OKAY

    # The value comes with the gradient, which SCIP often asks for next.
    if not ensure_gradient(data, CALLBACK_EVAL):
        return SCIP_OKAY
    value: cython.double = data.gradient_value
    val[0] = value
    if data.switches.caches:
        index = data.value_next
        memcpy(data.value_keys + index * n, x, n * cython.sizeof(cython.double))
        data.value_results[index] = value
        data.value_next = (index + 1) % VALUE_CACHE_SIZE
        data.value_count = min(data.value_count + 1, VALUE_CACHE_SIZE)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ensure_gradient(
    data: cython.pointer[ExprData], callback: cython.int
) -> cython.bint:
    """Fills the gradient cache for the point in the `x` input and returns whether it is valid.

    SCIP asks for one partial derivative or one directional derivative per call, so a single
    execution serves all of them at the same point.
    """
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    n: cython.int = function.n
    if data.gradient_valid and same(x, data.gradient_key, n):
        function.statistics.hits[callback] += 1
        return True
    value: cython.double
    if (
        run(
            data,
            PROGRAM_VALUE_AND_GRADIENT,
            cython.address(value),
            data.gradient,
            callback,
        )
        != 0
    ):
        # A program fails to run only if the plugin fails.
        data.gradient_valid = False  # pragma: no cover
        return False  # pragma: no cover
    memcpy(data.gradient_key, x, n * cython.sizeof(cython.double))
    data.gradient_value = value
    data.gradient_valid = all_finite(cython.address(value), 1) and all_finite(
        data.gradient, n
    )
    return data.gradient_valid


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def all_finite(values: cython.p_double, count: cython.int) -> cython.bint:
    """Returns whether every value of an array is finite."""
    index: cython.int
    for index in range(count):
        if not isfinite(values[index]):
            return False
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def bwdiff_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    childidx: cython.int,
    val: cython.pointer[SCIP_Real],
) -> SCIP_RETCODE:
    """Returns the partial derivative with respect to one child."""
    data: cython.pointer[ExprData] = get_data(expr)
    data.function.statistics.calls[CALLBACK_BWDIFF] += 1
    val[0] = SCIP_INVALID
    if read_child_values(expr, data.function.inputs[SLOT_X].host) and ensure_gradient(
        data, CALLBACK_BWDIFF
    ):
        val[0] = data.gradient[childidx]
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_child_dots(
    expr: cython.pointer[SCIP_EXPR], destination: cython.p_double
) -> cython.bint:
    """Copies the children's directional derivatives and returns whether all are valid."""
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        destination[index] = SCIPexprGetDot(children[index])
        # The children are variable expressions, whose derivatives are never invalid.
        if destination[index] == SCIP_INVALID:
            return False  # pragma: no cover
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def fwdiff_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    dot: cython.pointer[SCIP_Real],
    direction: cython.pointer[SCIP_SOL],
) -> SCIP_RETCODE:
    """Returns the directional derivative along the children's derivatives, from the gradient."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    dots: cython.p_double = function.inputs[SLOT_DIRECTION].host
    index: cython.int
    total: cython.double = 0.0
    function.statistics.calls[CALLBACK_FWDIFF] += 1
    dot[0] = SCIP_INVALID
    if (
        not read_child_values(expr, function.inputs[SLOT_X].host)
        or not read_child_dots(expr, dots)
        or not ensure_gradient(data, CALLBACK_FWDIFF)
    ):
        # SCIP evaluates a point before it differentiates there, and that evaluation
        # already fails if the gradient is not finite.
        return SCIP_OKAY  # pragma: no cover
    for index in range(function.n):
        if dots[index] != 0.0:
            total += data.gradient[index] * dots[index]
    dot[0] = total
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def bwfwdiff_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    childidx: cython.int,
    bardot: cython.pointer[SCIP_Real],
    direction: cython.pointer[SCIP_SOL],
) -> SCIP_RETCODE:
    """Returns one entry of the Hessian times the children's directional derivatives.

    SCIP asks for the entries one child at a time, so one execution per point and direction
    serves all of them.
    """
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    dots: cython.p_double = function.inputs[SLOT_DIRECTION].host
    n: cython.int = function.n
    function.statistics.calls[CALLBACK_BWFWDIFF] += 1
    bardot[0] = SCIP_INVALID
    if (
        not data.switches.hessian
        or not read_child_values(expr, x)
        or not read_child_dots(expr, dots)
    ):
        return SCIP_OKAY
    if (
        data.hessian_valid
        and same(x, data.hessian_key, n)
        and same(dots, data.hessian_key + n, n)
    ):
        function.statistics.hits[CALLBACK_BWFWDIFF] += 1
    else:
        if (
            run(
                data,
                PROGRAM_HESSIAN_VECTOR,
                data.hessian,
                cython.NULL,
                CALLBACK_BWFWDIFF,
            )
            != 0
        ):
            data.hessian_valid = False
            return SCIP_OKAY
        memcpy(data.hessian_key, x, n * cython.sizeof(cython.double))
        memcpy(data.hessian_key + n, dots, n * cython.sizeof(cython.double))
        data.hessian_valid = all_finite(data.hessian, n)
        if not data.hessian_valid:
            return SCIP_OKAY
    bardot[0] = data.hessian[childidx]
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_child_box(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    lower: cython.p_double,
    upper: cython.p_double,
) -> cython.bint:
    """Copies the children's activity bounds and returns whether all are finite and nonempty."""
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    activity: SCIP_INTERVAL
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        activity = SCIPexprGetActivity(children[index])
        if (
            activity.inf > activity.sup
            or SCIPisInfinity(scip, -activity.inf)
            or SCIPisInfinity(scip, activity.sup)
        ):
            return False
        lower[index] = activity.inf
        upper[index] = activity.sup
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def inteval_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    interval: cython.pointer[SCIP_INTERVAL],
    intevalvar: ExprIntevalVar,
    intevalvardata: cython.p_void,
) -> SCIP_RETCODE:
    """Encloses the function's range over the children's activities."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    n: cython.int = function.n
    index: cython.int
    key: cython.p_double
    result: cython.double[2]
    function.statistics.calls[CALLBACK_INTEVAL] += 1
    interval.inf = -SCIP_INTERVAL_INFINITY
    interval.sup = SCIP_INTERVAL_INFINITY
    if not read_child_box(scip, expr, lower, upper):
        return SCIP_OKAY
    if data.switches.caches:
        for index in range(data.bounds_count):
            key = data.bounds_keys + 2 * n * index
            if same(lower, key, n) and same(upper, key + n, n):
                function.statistics.hits[CALLBACK_INTEVAL] += 1
                set_interval(
                    interval,
                    data.bounds_results[2 * index],
                    data.bounds_results[2 * index + 1],
                )
                return SCIP_OKAY
    if (
        run(
            data,
            PROGRAM_INTERVAL,
            cython.address(result[0]),
            cython.address(result[1]),
            CALLBACK_INTEVAL,
        )
        != 0
    ):
        # A program fails to run only if the plugin fails.
        return SCIP_OKAY  # pragma: no cover
    set_interval(interval, result[0], result[1])
    if data.switches.caches:
        index = data.bounds_next
        key = data.bounds_keys + 2 * n * index
        memcpy(key, lower, n * cython.sizeof(cython.double))
        memcpy(key + n, upper, n * cython.sizeof(cython.double))
        data.bounds_results[2 * index] = result[0]
        data.bounds_results[2 * index + 1] = result[1]
        data.bounds_next = (index + 1) % BOUNDS_CACHE_SIZE
        data.bounds_count = min(data.bounds_count + 1, BOUNDS_CACHE_SIZE)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_interval(
    interval: cython.pointer[SCIP_INTERVAL], low: cython.double, high: cython.double
) -> cython.void:
    """Sets an interval from bounds, mapping infinities and leaving it entire on NaN or disorder."""
    if isnan(low) or isnan(high) or low > high:
        return
    interval.inf = max(low, -SCIP_INTERVAL_INFINITY)
    interval.sup = min(high, SCIP_INTERVAL_INFINITY)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compute_estimator(
    data: cython.pointer[ExprData],
    side: cython.int,
    callback: cython.int,
    constant: cython.p_double,
    coefficients: cython.p_double,
) -> cython.bint:
    """Computes a linear estimator at the `x` input over the box of the bound inputs.

    Side 0 is the underestimator from the convex relaxation and side 1 the overestimator from
    the concave relaxation. Returns whether the estimator is finite with coefficients in range.
    """
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    n: cython.int = function.n
    size: cython.size_t = n * cython.sizeof(cython.double)
    index: cython.int
    key: cython.p_double
    result: cython.p_double
    if data.switches.caches:
        for index in range(data.estimate_count[side]):
            key = data.estimate_keys[side] + 3 * n * index
            if (
                same(x, key, n)
                and same(lower, key + n, n)
                and same(upper, key + 2 * n, n)
            ):
                function.statistics.hits[callback] += 1
                result = data.estimate_results[side] + (n + 1) * index
                constant[0] = result[0]
                memcpy(coefficients, result + 1, size)
                return data.estimate_success[side][index]

    program: cython.int = PROGRAM_CONCAVE if side == 1 else PROGRAM_CONVEX
    value: cython.double
    success: cython.bint = run(
        data, program, cython.address(value), coefficients, callback
    ) == 0 and isfinite(value)
    if success:
        constant[0] = value
        for index in range(n):
            if (
                not isfinite(coefficients[index])
                or fabs(coefficients[index]) > MAX_ESTIMATOR_COEFFICIENT
            ):
                success = False
                break
            constant[0] -= coefficients[index] * x[index]
    if data.switches.caches:
        index = data.estimate_next[side]
        key = data.estimate_keys[side] + 3 * n * index
        memcpy(key, x, size)
        memcpy(key + n, lower, size)
        memcpy(key + 2 * n, upper, size)
        result = data.estimate_results[side] + (n + 1) * index
        result[0] = constant[0]
        memcpy(result + 1, coefficients, size)
        data.estimate_success[side][index] = success
        data.estimate_next[side] = (index + 1) % ESTIMATE_CACHE_SIZE
        data.estimate_count[side] = min(
            data.estimate_count[side] + 1, ESTIMATE_CACHE_SIZE
        )
    return success


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def estimate_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    localbounds: cython.pointer[SCIP_INTERVAL],
    globalbounds: cython.pointer[SCIP_INTERVAL],
    refpoint: cython.pointer[SCIP_Real],
    overestimate: SCIP_Bool,
    targetvalue: SCIP_Real,
    coefs: cython.pointer[SCIP_Real],
    constant: cython.pointer[SCIP_Real],
    islocal: cython.pointer[SCIP_Bool],
    success: cython.pointer[SCIP_Bool],
    branchcand: cython.pointer[SCIP_Bool],
) -> SCIP_RETCODE:
    """Returns a linear under- or overestimator from a relaxation over the local box."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    index: cython.int
    is_global: cython.bint = data.switches.global_estimators
    function.statistics.calls[CALLBACK_ESTIMATE] += 1
    success[0] = False
    islocal[0] = True
    for index in range(function.n):
        if (
            SCIPisInfinity(scip, -localbounds[index].inf)
            or SCIPisInfinity(scip, localbounds[index].sup)
            or refpoint[index] == SCIP_INVALID
        ):
            return SCIP_OKAY
        lower[index] = localbounds[index].inf
        upper[index] = localbounds[index].sup

        # The relaxations are defined on the box, and a plane anchored anywhere in it holds on
        # all of it.
        x[index] = min(max(refpoint[index], lower[index]), upper[index])
        if (
            localbounds[index].inf != globalbounds[index].inf
            or localbounds[index].sup != globalbounds[index].sup
        ):
            is_global = False
    success[0] = compute_estimator(
        data, 1 if overestimate else 0, CALLBACK_ESTIMATE, constant, coefs
    )
    islocal[0] = not is_global
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def initestimates_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    bounds: cython.pointer[SCIP_INTERVAL],
    overestimate: SCIP_Bool,
    coefs: cython.pointer[cython.pointer[SCIP_Real]],
    constant: cython.pointer[SCIP_Real],
    nreturned: cython.p_int,
) -> SCIP_RETCODE:
    """Returns estimators at the box center and at two points on its diagonal, for the initial LP."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    x: cython.p_double = function.inputs[SLOT_X].host
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    index: cython.int
    point: cython.int
    fractions: cython.double[NUM_INIT_POINTS] = [0.5, 0.25, 0.75]
    nreturned[0] = 0
    function.statistics.calls[CALLBACK_INITESTIMATES] += 1
    if not data.switches.init_estimates:
        return SCIP_OKAY
    for index in range(function.n):
        if SCIPisInfinity(scip, -bounds[index].inf) or SCIPisInfinity(
            scip, bounds[index].sup
        ):
            return SCIP_OKAY
        lower[index] = bounds[index].inf
        upper[index] = bounds[index].sup
    for point in range(min(NUM_INIT_POINTS, SCIP_EXPR_MAXINITESTIMATES)):
        for index in range(function.n):
            x[index] = lower[index] + fractions[point] * (upper[index] - lower[index])
        if compute_estimator(
            data,
            1 if overestimate else 0,
            CALLBACK_INITESTIMATES,
            cython.address(constant[nreturned[0]]),
            coefs[nreturned[0]],
        ):
            nreturned[0] += 1
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def monotonicity_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    childidx: cython.int,
    result: cython.pointer[SCIP_MONOTONE],
) -> SCIP_RETCODE:
    """Reports monotonicity in one child from bounds on the gradient over the children's activities."""
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    n: cython.int = function.n
    index: cython.int
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    function.statistics.calls[CALLBACK_MONOTONICITY] += 1
    result[0] = SCIP_MONOTONE_UNKNOWN
    if (
        not data.switches.monotonicity
        or function.programs[PROGRAM_GRADIENT_INTERVAL].handle == cython.NULL
    ):
        return SCIP_OKAY
    for index in range(n):
        if SCIPevalExprActivity(scip, children[index]) != SCIP_OKAY:
            return SCIP_OKAY  # pragma: no cover
    if not read_child_box(scip, expr, lower, upper):
        return SCIP_OKAY
    if (
        data.monotonicity_valid
        and same(lower, data.monotonicity_key, n)
        and same(upper, data.monotonicity_key + n, n)
    ):
        function.statistics.hits[CALLBACK_MONOTONICITY] += 1
    else:
        if (
            run(
                data,
                PROGRAM_GRADIENT_INTERVAL,
                data.monotonicity_result,
                data.monotonicity_result + n,
                CALLBACK_MONOTONICITY,
            )
            != 0
        ):
            # A program fails to run only if the plugin fails.
            data.monotonicity_valid = False  # pragma: no cover
            return SCIP_OKAY  # pragma: no cover
        memcpy(data.monotonicity_key, lower, n * cython.sizeof(cython.double))
        memcpy(data.monotonicity_key + n, upper, n * cython.sizeof(cython.double))
        data.monotonicity_valid = True
    low: cython.double = data.monotonicity_result[childidx]
    high: cython.double = data.monotonicity_result[n + childidx]
    if isnan(low) or isnan(high):
        return SCIP_OKAY
    if low >= 0.0 and high <= 0.0:
        result[0] = SCIP_MONOTONE_CONST
    elif low >= 0.0:
        result[0] = SCIP_MONOTONE_INC
    elif high <= 0.0:
        result[0] = SCIP_MONOTONE_DEC
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def tighten_from_plane(
    constant: cython.double,
    weights: cython.p_double,
    rhs: cython.double,
    n: cython.int,
    tolerance: cython.double,
    childrenbounds: cython.pointer[SCIP_INTERVAL],
) -> cython.bint:
    """Tightens the children's bounds with `constant + weights · x ≤ rhs`, returning whether that is infeasible.

    Each child's bound follows from the inequality with every other term at its smallest value
    over the box. The new bounds are relaxed by `tolerance`, so rounding cannot cut off points.
    """
    index: cython.int
    smallest: cython.double = 0.0
    term: cython.double
    bound: cython.double
    for index in range(n):
        smallest += min(
            weights[index] * childrenbounds[index].inf,
            weights[index] * childrenbounds[index].sup,
        )
    if constant + smallest > rhs + tolerance * (1.0 + fabs(rhs)):
        return True
    for index in range(n):
        if fabs(weights[index]) < 1e-9:
            continue
        term = min(
            weights[index] * childrenbounds[index].inf,
            weights[index] * childrenbounds[index].sup,
        )
        bound = (rhs - constant - (smallest - term)) / weights[index]
        if weights[index] > 0.0:
            bound += tolerance * (1.0 + fabs(bound))
            childrenbounds[index].sup = min(childrenbounds[index].sup, bound)
        else:
            bound -= tolerance * (1.0 + fabs(bound))
            childrenbounds[index].inf = max(childrenbounds[index].inf, bound)
        if childrenbounds[index].inf > childrenbounds[index].sup:
            return True
    return False


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def center_box(
    function: cython.pointer[Function], childrenbounds: cython.pointer[SCIP_INTERVAL]
) -> cython.void:
    """Sets the bound inputs to the children's bounds and the `x` input to the box center."""
    x: cython.p_double = function.inputs[SLOT_X].host
    lower: cython.p_double = function.inputs[SLOT_LOWER].host
    upper: cython.p_double = function.inputs[SLOT_UPPER].host
    index: cython.int
    for index in range(function.n):
        lower[index] = childrenbounds[index].inf
        upper[index] = childrenbounds[index].sup
        x[index] = 0.5 * (lower[index] + upper[index])


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reverseprop_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    bounds: SCIP_INTERVAL,
    childrenbounds: cython.pointer[SCIP_INTERVAL],
    infeasible: cython.pointer[SCIP_Bool],
) -> SCIP_RETCODE:
    """Tightens the children's bounds from bounds on the function, with estimators at the box center.

    The underestimator `l` satisfies `l(x) ≤ f(x) ≤ sup`, and the overestimator `u` satisfies
    `u(x) ≥ f(x) ≥ inf`. Both are linear, so each gives bounds on every child.
    """
    data: cython.pointer[ExprData] = get_data(expr)
    function: cython.pointer[Function] = data.function
    n: cython.int = function.n
    index: cython.int
    constant: cython.double
    weights: cython.p_double = function.scratch
    tolerance: cython.double = SCIPfeastol(scip)
    infeasible[0] = False
    function.statistics.calls[CALLBACK_REVERSEPROP] += 1
    if not data.switches.reverse_propagation:
        return SCIP_OKAY
    for index in range(n):
        if (
            SCIPisInfinity(scip, -childrenbounds[index].inf)
            or SCIPisInfinity(scip, childrenbounds[index].sup)
            or childrenbounds[index].inf > childrenbounds[index].sup
        ):
            return SCIP_OKAY
    center_box(function, childrenbounds)
    if (
        bounds.sup < SCIP_INTERVAL_INFINITY
        and compute_estimator(
            data, 0, CALLBACK_REVERSEPROP, cython.address(constant), weights
        )
        and tighten_from_plane(
            constant, weights, bounds.sup, n, tolerance, childrenbounds
        )
    ):
        infeasible[0] = True
        return SCIP_OKAY
    if bounds.inf > -SCIP_INTERVAL_INFINITY:
        center_box(function, childrenbounds)
        if compute_estimator(
            data, 1, CALLBACK_REVERSEPROP, cython.address(constant), weights
        ):
            for index in range(n):
                weights[index] = -weights[index]
            infeasible[0] = tighten_from_plane(
                -constant, weights, -bounds.inf, n, tolerance, childrenbounds
            )
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def getsymdata_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    symdata: cython.pointer[cython.pointer[SYM_EXPRDATA]],
) -> SCIP_RETCODE:
    """Describes the expression to SCIP's symmetry detection.

    Without this callback SCIP treats the children of an unknown operator as interchangeable
    and can cut off optimal solutions. A distinct coefficient per child marks the input order,
    and the constants identify the function and its parameters.
    """
    data: cython.pointer[ExprData] = get_data(expr)
    n: cython.int = data.function.n
    m: cython.int = data.function.m
    index: cython.int
    if allocSymData(scip, symdata) != SCIP_OKAY:
        return SCIP_NOMEMORY  # pragma: no cover
    description: cython.pointer[SYM_EXPRDATA] = symdata[0]
    description.nconstants = m + 1
    description.ncoefficients = n
    if (
        allocRealArray(scip, cython.address(description.constants), m + 1) != SCIP_OKAY
        or allocRealArray(scip, cython.address(description.coefficients), n)
        != SCIP_OKAY
        or allocExprArray(scip, cython.address(description.children), n) != SCIP_OKAY
    ):
        return SCIP_NOMEMORY  # pragma: no cover
    description.constants[0] = cython.cast(
        cython.double, cython.cast(uintptr_t, data.function) % 1000003
    )
    for index in range(m):
        description.constants[index + 1] = data.params.host[index]
    for index in range(n):
        description.coefficients[index] = index + 1.0
        description.children[index] = SCIPexprGetChildren(expr)[index]
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def copydata_callback(
    targetscip: cython.pointer[SCIP],
    targetexprhdlr: cython.pointer[SCIP_EXPRHDLR],
    targetexprdata: cython.pointer[cython.pointer[SCIP_EXPRDATA]],
    sourcescip: cython.pointer[SCIP],
    sourceexpr: cython.pointer[SCIP_EXPR],
) -> SCIP_RETCODE:
    """Copies the expression data into a copy of SCIP, sharing the function with empty caches."""
    source: cython.pointer[ExprData] = get_data(sourceexpr)
    copy: cython.pointer[ExprData] = new_expr_data(
        source.function, source.params.host, source.switches
    )
    if copy == cython.NULL:
        return SCIP_NOMEMORY  # pragma: no cover
    targetexprdata[0] = cython.cast(cython.pointer[SCIP_EXPRDATA], copy)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def freedata_callback(
    scip: cython.pointer[SCIP], expr: cython.pointer[SCIP_EXPR]
) -> SCIP_RETCODE:
    """Frees the expression data."""
    free_expr_data(get_data(expr))
    SCIPexprSetData(expr, cython.NULL)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compare_callback(
    scip: cython.pointer[SCIP],
    expr1: cython.pointer[SCIP_EXPR],
    expr2: cython.pointer[SCIP_EXPR],
) -> cython.int:
    """Orders expressions by function, then by parameters, then by children."""
    first: cython.pointer[ExprData] = get_data(expr1)
    second: cython.pointer[ExprData] = get_data(expr2)
    index: cython.int
    order: cython.int
    if first.function != second.function:
        return (
            -1
            if cython.cast(uintptr_t, first.function)
            < cython.cast(uintptr_t, second.function)
            else 1
        )
    for index in range(first.function.m):
        if first.params.host[index] != second.params.host[index]:
            return -1 if first.params.host[index] < second.params.host[index] else 1
    for index in range(first.function.n):
        order = SCIPcompareExpr(
            scip, SCIPexprGetChildren(expr1)[index], SCIPexprGetChildren(expr2)[index]
        )
        if order != 0:
            return order
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def hash_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    hashkey: cython.p_uint,
    childrenhashes: cython.p_uint,
) -> SCIP_RETCODE:
    """Hashes the function, the parameters and the children."""
    data: cython.pointer[ExprData] = get_data(expr)
    address: uintptr_t = cython.cast(uintptr_t, data.function)
    key: cython.uint = cython.cast(cython.uint, address >> 4) ^ cython.cast(
        cython.uint, address >> 36
    )
    bits: cython.ulonglong
    index: cython.int
    for index in range(data.function.m):
        memcpy(
            cython.address(bits),
            cython.address(data.params.host[index]),
            cython.sizeof(bits),
        )
        key = key * 31 + cython.cast(cython.uint, bits ^ (bits >> 32))
    for index in range(data.function.n):
        key = key * 31 + childrenhashes[index]
    hashkey[0] = key
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def copyhdlr_callback(
    scip: cython.pointer[SCIP], sourceexprhdlr: cython.pointer[SCIP_EXPRHDLR]
) -> SCIP_RETCODE:
    """Includes the handler into copies of SCIP, which some primal heuristics solve."""
    return include_handler(scip)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def include_handler(scip: cython.pointer[SCIP]) -> SCIP_RETCODE:
    """Includes the JAX expression handler into a SCIP instance unless it is already there."""
    if SCIPfindExprhdlr(scip, HANDLER_NAME) != cython.NULL:
        return SCIP_OKAY
    handler: cython.pointer[SCIP_EXPRHDLR]
    retcode: SCIP_RETCODE = SCIPincludeExprhdlr(
        scip,
        cython.address(handler),
        HANDLER_NAME,
        b"scalar JAX function executed through PJRT",
        0,
        eval_callback,
        cython.NULL,
    )
    if retcode != SCIP_OKAY:
        return retcode  # pragma: no cover
    SCIPexprhdlrSetCopyFreeHdlr(handler, copyhdlr_callback, cython.NULL)
    SCIPexprhdlrSetCopyFreeData(handler, copydata_callback, freedata_callback)
    SCIPexprhdlrSetCompare(handler, compare_callback)
    SCIPexprhdlrSetHash(handler, hash_callback)
    SCIPexprhdlrSetIntEval(handler, inteval_callback)
    SCIPexprhdlrSetEstimate(handler, initestimates_callback, estimate_callback)
    SCIPexprhdlrSetDiff(handler, bwdiff_callback, fwdiff_callback, bwfwdiff_callback)
    SCIPexprhdlrSetMonotonicity(handler, monotonicity_callback)
    SCIPexprhdlrSetGetSymdata(handler, getsymdata_callback)
    SCIPexprhdlrSetReverseProp(handler, reverseprop_callback)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def create_expression(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[cython.pointer[SCIP_EXPR]],
    data: cython.pointer[ExprData],
    children: cython.pointer[cython.pointer[SCIP_EXPR]],
) -> SCIP_RETCODE:
    """Creates an expression of the JAX handler that owns `data`, including the handler if needed."""
    retcode: SCIP_RETCODE = include_handler(scip)
    if retcode != SCIP_OKAY:
        free_expr_data(data)  # pragma: no cover
        return retcode  # pragma: no cover
    return SCIPcreateExpr(
        scip,
        expr,
        SCIPfindExprhdlr(scip, HANDLER_NAME),
        cython.cast(cython.pointer[SCIP_EXPRDATA], data),
        data.function.n,
        children,
        cython.NULL,
        cython.NULL,
    )


def check_length(name: str, values: object, expected: int) -> None:
    """Raises a `ValueError` unless an array has the expected length."""
    actual: int = len(values)
    if actual != expected:
        raise ValueError(f"{name} has {actual} entries, expected {expected}")


DIRECT_SWITCHES = HandlerOptions(caches=False)
"""Switches of the direct calls of `FunctionHandle`, which never reuse results."""


@cython.cclass
class FunctionHandle:
    """The executables of one function structure, shared by all expressions with that structure.

    A SCIP problem keeps the handles of its expressions alive for as long as it exists.
    """

    def __cinit__(
        self, num_inputs: cython.int, num_params: cython.int, programs: Programs
    ) -> None:
        require_runtime()
        if num_inputs < 1:
            raise ValueError("A function needs at least one input")
        self.num_inputs = num_inputs
        self.num_params = max(num_params, 1)
        self.programs = programs
        function: cython.pointer[Function] = cython.cast(
            cython.pointer[Function], malloc(cython.sizeof(Function))
        )
        memset(function, 0, cython.sizeof(Function))
        self.function = function
        function.n = num_inputs
        function.m = self.num_params
        function.scratch = aligned_doubles(num_inputs)
        slot: cython.int
        for slot in range(SLOT_PARAMS):
            function.inputs[slot].host = aligned_doubles(num_inputs)
            function.inputs[slot].size = num_inputs
            function.inputs[slot].buffer = alias_buffer(
                function.inputs[slot].host, num_inputs
            )

        executable: Executable
        index: cython.int
        position: cython.int
        for program in Program:
            executable = programs[program]
            if executable is None:
                if program not in (Program.HESSIAN_VECTOR, Program.GRADIENT_INTERVAL):
                    raise ValueError(f"The {program.name.lower()} program is required")
                continue
            sizes = output_sizes(program, num_inputs)
            if executable.num_outputs != len(sizes):
                raise ValueError(
                    f"The {program.name.lower()} program has {executable.num_outputs} outputs, "
                    f"expected {len(sizes)}"
                )
            index = program
            function.programs[index].handle = executable.handle
            function.programs[index].num_inputs = len(PROGRAM_SLOTS[index])
            for position, slot in enumerate(PROGRAM_SLOTS[index]):
                function.slots[index][position] = slot
            function.programs[index].num_outputs = len(sizes)
            for position, size in enumerate(sizes):
                function.programs[index].output_sizes[position] = size

    def __dealloc__(self) -> None:
        function: cython.pointer[Function] = self.function
        slot: cython.int
        if function == cython.NULL:
            return  # pragma: no cover
        for slot in range(SLOT_PARAMS):
            destroy_buffer(function.inputs[slot].buffer)
            free(function.inputs[slot].host)
        free(function.scratch)
        free(function)

    def statistics(self) -> HandlerStatistics:
        """Returns calls, cache hits, executions and execution seconds of each callback."""
        counters: cython.pointer[Statistics] = cython.address(self.function.statistics)
        return HandlerStatistics._make(
            CallbackStatistics(
                calls=counters.calls[index],
                hits=counters.hits[index],
                runs=counters.runs[index],
                seconds=counters.seconds[index],
            )
            for index in range(NUM_CALLBACKS)
        )

    def reset_statistics(self) -> None:
        """Clears the counters."""
        memset(cython.address(self.function.statistics), 0, cython.sizeof(Statistics))

    def set_input(self, slot: cython.int, values: cython.double[::1]) -> None:
        """Copies an array into the host memory of an input slot."""
        check_length("input", values, self.num_inputs)
        memcpy(
            self.function.inputs[slot].host,
            cython.address(values[0]),
            self.num_inputs * cython.sizeof(cython.double),
        )

    def evaluate(self, x: cython.double[::1], params: cython.double[::1]) -> float:
        """Runs the gradient program and returns `f(x, params)`."""
        return self.value_and_gradient(x, params)[0]

    def value_and_gradient(
        self, x: cython.double[::1], params: cython.double[::1]
    ) -> tuple[float, np.ndarray]:
        """Runs the gradient program and returns the value and the gradient in `x`."""
        value: cython.double = 0.0
        gradient = np.empty(self.num_inputs)
        view: cython.double[::1] = gradient
        self.set_input(SLOT_X, x)
        run_direct(
            self,
            Program.VALUE_AND_GRADIENT,
            params,
            cython.address(value),
            cython.address(view[0]),
        )
        return value, gradient

    def hessian_vector(
        self,
        x: cython.double[::1],
        direction: cython.double[::1],
        params: cython.double[::1],
    ) -> np.ndarray:
        """Runs the Hessian-vector program and returns the Hessian in `x` times `direction`."""
        product = np.empty(self.num_inputs)
        view: cython.double[::1] = product
        self.set_input(SLOT_X, x)
        self.set_input(SLOT_DIRECTION, direction)
        run_direct(
            self, Program.HESSIAN_VECTOR, params, cython.address(view[0]), cython.NULL
        )
        return product

    def estimator(
        self,
        reference: cython.double[::1],
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
        overestimate: cython.bint,
    ) -> Estimator:
        """Computes an estimator at a reference point clamped to the box, as `ESTIMATE` does."""
        check_length("reference", reference, self.num_inputs)
        self.set_input(SLOT_LOWER, lower)
        self.set_input(SLOT_UPPER, upper)
        x: cython.p_double = self.function.inputs[SLOT_X].host
        index: cython.int
        for index in range(self.num_inputs):
            x[index] = min(max(reference[index], lower[index]), upper[index])
        coefficients = np.empty(self.num_inputs)
        view: cython.double[::1] = coefficients
        constant: cython.double = 0.0
        data: cython.pointer[ExprData] = direct_data(
            self, params, switches_from(DIRECT_SWITCHES)
        )
        success: cython.bint = compute_estimator(
            data,
            1 if overestimate else 0,
            CALLBACK_ESTIMATE,
            cython.address(constant),
            cython.address(view[0]),
        )
        free_expr_data(data)
        return Estimator(
            success=bool(success), constant=constant, coefficients=coefficients
        )

    def interval(
        self,
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
    ) -> Interval:
        """Runs the interval program and returns bounds on the value over the box."""
        result: cython.double[2]
        self.set_input(SLOT_LOWER, lower)
        self.set_input(SLOT_UPPER, upper)
        run_direct(
            self,
            Program.INTERVAL,
            params,
            cython.address(result[0]),
            cython.address(result[1]),
        )
        return Interval(lower=result[0], upper=result[1])

    def gradient_interval(
        self,
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
    ) -> Interval:
        """Runs the gradient interval program and returns bounds on the gradient over the box."""
        low = np.empty(self.num_inputs)
        high = np.empty(self.num_inputs)
        low_view: cython.double[::1] = low
        high_view: cython.double[::1] = high
        self.set_input(SLOT_LOWER, lower)
        self.set_input(SLOT_UPPER, upper)
        run_direct(
            self,
            Program.GRADIENT_INTERVAL,
            params,
            cython.address(low_view[0]),
            cython.address(high_view[0]),
        )
        return Interval(lower=low, upper=high)

    def time_program(
        self,
        program: Program,
        points: cython.double[:, ::1],
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
        repeat: cython.int = 1,
    ) -> float:
        """Returns the mean seconds per execution of one program over many points, measured in C.

        Each execution writes a new point into the `x` input, as the callbacks do, and the
        direction of the Hessian-vector program is the point itself.
        """
        if points.shape[0] == 0 or repeat < 1:
            raise ValueError("Timing needs at least one point and one repetition")
        check_length("points[0]", points[0], self.num_inputs)
        self.set_input(SLOT_LOWER, lower)
        self.set_input(SLOT_UPPER, upper)
        index: cython.int = program
        if self.function.programs[index].handle == cython.NULL:
            raise ValueError(f"The function has no {program.name.lower()} program")
        data: cython.pointer[ExprData] = direct_data(
            self, params, switches_from(DIRECT_SWITCHES)
        )
        first: cython.p_double = aligned_doubles(self.num_inputs + 1)
        second: cython.p_double = aligned_doubles(self.num_inputs + 1)
        size: cython.size_t = self.num_inputs * cython.sizeof(cython.double)
        x: cython.p_double = self.function.inputs[SLOT_X].host
        direction: cython.p_double = self.function.inputs[SLOT_DIRECTION].host
        count: cython.int = points.shape[0]
        status: cython.int = 0
        round_index: cython.int
        point: cython.int
        start: cython.double
        elapsed: cython.double
        with cython.nogil:
            start = now()
            for round_index in range(repeat):
                for point in range(count):
                    memcpy(x, cython.address(points[point, 0]), size)
                    memcpy(direction, cython.address(points[point, 0]), size)
                    status |= run(data, index, first, second, CALLBACK_EVAL)
            elapsed = now() - start
        free(first)
        free(second)
        free_expr_data(data)
        if status != 0:
            raise_last_error(program.name.lower())  # pragma: no cover
        return elapsed / (repeat * count)


@cython.cfunc
def run_direct(
    handle: FunctionHandle,
    program: Program,
    params: cython.double[::1],
    first: cython.p_double,
    second: cython.p_double,
) -> cython.void:
    """Runs one program of a handle on its current host inputs, raising an error on failure."""
    index: cython.int = program
    data: cython.pointer[ExprData] = direct_data(
        handle, params, switches_from(DIRECT_SWITCHES)
    )
    status: cython.int = run(data, index, first, second, CALLBACK_EVAL)
    free_expr_data(data)
    if status != 0:
        if handle.function.programs[index].handle == cython.NULL:
            raise ValueError(f"The function has no {program.name.lower()} program")
        raise_last_error(program.name.lower())  # pragma: no cover


@cython.cfunc
def direct_data(
    handle: FunctionHandle, params: cython.double[::1], switches: Switches
) -> cython.pointer[ExprData]:
    """Returns new expression data for the direct calls of a handle, which the caller frees."""
    check_length("params", params, handle.num_params)
    return new_expr_data(handle.function, cython.address(params[0]), switches)
