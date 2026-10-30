"""SCIP expression handler for entries of block terms, executed through the PJRT runtime.

A block term is a vector function `f(x, params)` of many variables whose entries each read a few
of them. Each entry is one expression of the `jaxblock` handler, with the variables of its entry
as children. All entries of a block share one `BlockState` in a SCIP instance. It holds the host
inputs of the block's programs and the most recent runs of each program, which cover every entry
at once. A run is valid for an entry exactly when the inputs that the run read at the entry's
children equal the inputs now, because the entry depends on nothing else. The programs
return the derivatives of all entries in the order of the block's pattern.

All callbacks run without the GIL. A SCIP instance and its copies each have their own states, so
solves in different threads never share an input.
"""

import cython
import cyxla
import numpy as np
from cython.cimports.cyxla import (
    ABI_VERSION,
    MAX_OUTPUTS,
    Executable,
    Input,
    ProgramSpec,
    execute,
    free_aligned,
    now,
)
from cython.cimports.jaxcip._handler import (
    CALLBACK_BWDIFF,
    CALLBACK_BWFWDIFF,
    CALLBACK_ESTIMATE,
    CALLBACK_EVAL,
    CALLBACK_FWDIFF,
    CALLBACK_INITESTIMATES,
    CALLBACK_INTEVAL,
    CALLBACK_MONOTONICITY,
    CALLBACK_REVERSEPROP,
    NUM_CALLBACKS,
    NUM_INIT_POINTS,
    NUM_PROGRAMS,
    PROGRAM_CONCAVE,
    PROGRAM_CONVEX,
    PROGRAM_GRADIENT_INTERVAL,
    PROGRAM_HESSIAN_VECTOR,
    PROGRAM_INTERVAL,
    PROGRAM_VALUE_AND_GRADIENT,
    SLOT_DIRECTION,
    SLOT_LOWER,
    SLOT_PARAMS,
    SLOT_UPPER,
    SLOT_X,
    Statistics,
    Switches,
    aligned_doubles,
    all_finite,
    as_doubles,
    create_input,
    destroy_input,
    set_interval,
    switches_from,
    tighten_from_plane,
)
from cython.cimports.jaxcip._scip import (
    SCIP,
    SCIP_EXPR,
    SCIP_EXPR_MAXINITESTIMATES,
    SCIP_EXPRDATA,
    SCIP_EXPRHDLR,
    SCIP_EXPRHDLRDATA,
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
    SCIP_VAR,
    SYM_EXPRDATA,
    ExprIntevalVar,
    SCIP_Bool,
    SCIP_Real,
    SCIPcreateExpr,
    SCIPevalExprActivity,
    SCIPexprGetActivity,
    SCIPexprGetChildren,
    SCIPexprGetData,
    SCIPexprGetDot,
    SCIPexprGetEvalValue,
    SCIPexprGetNChildren,
    SCIPexprhdlrGetData,
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
    SCIPgetSolVal,
    SCIPgetVarExprVar,
    SCIPincludeExprhdlr,
    SCIPisExprVar,
    SCIPisInfinity,
    SCIPvarGetLbLocal,
    SCIPvarGetUbLocal,
    allocExprArray,
    allocRealArray,
    allocSymData,
)
from cython.cimports.libc.math import fabs, isfinite, isnan
from cython.cimports.libc.stdint import uintptr_t
from cython.cimports.libc.stdlib import calloc, free, malloc, realloc
from cython.cimports.libc.string import memcpy, memset
from cyxla import raise_last_error

from jaxcip._options import HandlerOptions
from jaxcip._types import (
    CallbackStatistics,
    Estimator,
    HandlerStatistics,
    Interval,
    Program,
    Programs,
)

cyxla.check_abi(ABI_VERSION)

BLOCK_HANDLER_NAME = cython.declare(
    cython.pointer[cython.const[cython.char]], b"jaxblock"
)
"""Name of the expression handler of block entries in SCIP."""

MAX_ESTIMATOR_COEFFICIENT = cython.declare(cython.double, 1e9)
"""Largest absolute estimator coefficient passed to SCIP, beyond which the LP gets unstable."""

BLOCK_SLOTS: tuple[tuple[int, ...], ...] = (
    (SLOT_X, SLOT_PARAMS),
    (SLOT_X, SLOT_DIRECTION, SLOT_PARAMS),
    (SLOT_X, SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_X, SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
    (SLOT_LOWER, SLOT_UPPER, SLOT_PARAMS),
)
"""Inputs of each program of a block in the order of `Program`."""


def block_output_sizes(program: Program, rows: int, nnz: int) -> tuple[int, ...]:
    """Returns the number of doubles in each output of a block program.

    Values and bounds have one entry per row of the block, and derivatives one per entry of
    its pattern.
    """
    match program:
        case Program.VALUE_AND_GRADIENT | Program.CONVEX | Program.CONCAVE:
            return (rows, nnz)
        case Program.HESSIAN_VECTOR:
            return (nnz,)
        case Program.INTERVAL:
            return (rows, rows)
        case Program.GRADIENT_INTERVAL:
            return (nnz, nnz)


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def key_count(program: cython.int) -> cython.int:
    """Returns the number of inputs of a program that a run depends on besides the parameters."""
    if program == PROGRAM_VALUE_AND_GRADIENT:
        return 1
    if program == PROGRAM_CONVEX or program == PROGRAM_CONCAVE:
        return 3
    return 2


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def key_slot(program: cython.int, index: cython.int) -> cython.int:
    """Returns the input slot of the `index`-th key of a program."""
    if program == PROGRAM_VALUE_AND_GRADIENT:
        return SLOT_X
    if program == PROGRAM_HESSIAN_VECTOR:
        return SLOT_X if index == 0 else SLOT_DIRECTION
    if program == PROGRAM_CONVEX or program == PROGRAM_CONCAVE:
        return SLOT_X if index == 0 else (SLOT_LOWER if index == 1 else SLOT_UPPER)
    return SLOT_LOWER if index == 0 else SLOT_UPPER


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def run_slots(program: cython.int) -> cython.int:
    """Returns how many runs of a program a block remembers."""
    if program == PROGRAM_INTERVAL:
        return 4
    if program == PROGRAM_HESSIAN_VECTOR or program == PROGRAM_GRADIENT_INTERVAL:
        return 1
    return 2


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def first_size(
    function: cython.pointer[BlockFunction], program: cython.int
) -> cython.int:
    """Returns the number of doubles in the first output of a program."""
    if program == PROGRAM_HESSIAN_VECTOR or program == PROGRAM_GRADIENT_INTERVAL:
        return function.nnz
    return function.rows


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def second_size(
    function: cython.pointer[BlockFunction], program: cython.int
) -> cython.int:
    """Returns the number of doubles in the second output of a program, which is 0 if it has none."""
    if program == PROGRAM_HESSIAN_VECTOR:
        return 0
    if program == PROGRAM_INTERVAL:
        return function.rows
    return function.nnz


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def new_block_state(
    function: cython.pointer[BlockFunction],
    params: cython.p_double,
    switches: Switches,
) -> cython.pointer[BlockState]:
    """Returns the state of a block with a copy of `params`, one reference and empty caches.

    Returns NULL on failure.
    """
    state: cython.pointer[BlockState] = cython.cast(
        cython.pointer[BlockState], calloc(1, cython.sizeof(BlockState))
    )
    if state == cython.NULL:
        return cython.NULL  # pragma: no cover
    state.function = function
    state.references = 1
    state.switches = switches
    slot: cython.int
    size: cython.int
    for slot in range(SLOT_PARAMS + 1):
        size = function.m if slot == SLOT_PARAMS else function.n
        create_input(cython.address(state.inputs[slot]), size)
    memcpy(
        as_doubles(state.inputs[SLOT_PARAMS].host),
        params,
        function.m * cython.sizeof(cython.double),
    )
    state.scratch = aligned_doubles(function.n)
    state.variables = cython.cast(
        cython.pointer[cython.pointer[SCIP_VAR]],
        calloc(function.n, cython.sizeof(cython.pointer[SCIP_VAR])),
    )
    program: cython.int
    slots: cython.int
    for program in range(NUM_PROGRAMS):
        slots = run_slots(program)
        state.caches[program].keys = aligned_doubles(
            slots * key_count(program) * function.n
        )
        state.caches[program].first = aligned_doubles(
            slots * first_size(function, program)
        )
        state.caches[program].second = aligned_doubles(
            slots * max(second_size(function, program), 1)
        )
    return state


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def release_block_state(state: cython.pointer[BlockState]) -> cython.void:
    """Drops a reference to a block state and frees it when none is left."""
    state.references -= 1
    if state.references > 0:
        return
    slot: cython.int
    program: cython.int
    for slot in range(SLOT_PARAMS + 1):
        destroy_input(cython.address(state.inputs[slot]))
    for program in range(NUM_PROGRAMS):
        free_aligned(state.caches[program].keys)
        free_aligned(state.caches[program].first)
        free_aligned(state.caches[program].second)
    free_aligned(state.scratch)
    free(state.variables)
    free(state)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def new_block_data(
    state: cython.pointer[BlockState], entry: cython.int
) -> cython.pointer[BlockData]:
    """Returns the data of an expression that is an entry of a block, or NULL on failure."""
    data: cython.pointer[BlockData] = cython.cast(
        cython.pointer[BlockData], malloc(cython.sizeof(BlockData))
    )
    if data == cython.NULL:
        return cython.NULL  # pragma: no cover
    data.state = state
    data.entry = entry
    state.references += 1
    return data


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def get_data(expr: cython.pointer[SCIP_EXPR]) -> cython.pointer[BlockData]:
    """Returns the handler's data of an expression."""
    return cython.cast(cython.pointer[BlockData], SCIPexprGetData(expr))


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def execute_program(
    state: cython.pointer[BlockState],
    program: cython.int,
    first: cython.p_double,
    second: cython.p_double,
    callback: cython.int,
) -> cython.int:
    """Executes a program on the host inputs of a block and writes its outputs.

    Returns -1 if the program does not exist and the plugin's status otherwise.
    """
    function: cython.pointer[BlockFunction] = state.function
    spec: cython.pointer[ProgramSpec] = cython.address(function.programs[program])
    if spec.handle == cython.NULL:
        return -1
    inputs: Input[MAX_INPUTS]
    outputs: cython.p_void[MAX_OUTPUTS]
    index: cython.int
    for index in range(spec.num_inputs):
        inputs[index] = state.inputs[function.slots[program][index]]
    outputs[0] = first
    outputs[1] = second
    outputs[2] = cython.NULL
    start: cython.double = now()
    status: cython.int = execute(spec, inputs, outputs, state.switches.zero_copy)
    function.statistics.runs[callback] += 1
    function.statistics.seconds[callback] += now() - start
    return status


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def entry_matches(
    state: cython.pointer[BlockState],
    entry: cython.int,
    program: cython.int,
    slot: cython.int,
) -> cython.bint:
    """Returns whether a stored run read the current inputs at the children of an entry."""
    function: cython.pointer[BlockFunction] = state.function
    count: cython.int = key_count(program)
    snapshot: cython.p_double = state.caches[program].keys + slot * count * function.n
    live: cython.p_double
    column: cython.int
    key: cython.int
    position: cython.int
    for key in range(count):
        live = as_doubles(state.inputs[key_slot(program, key)].host)
        for position in range(function.pointers[entry], function.pointers[entry + 1]):
            column = function.columns[position]
            if live[column] != snapshot[key * function.n + column]:
                return False
    return True


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def new_request(
    scip: cython.pointer[SCIP], sol: cython.pointer[SCIP_SOL], kind: cython.int
) -> Prefetch:
    """Returns a request to fill the inputs of a block from the state of SCIP."""
    request: Prefetch
    request.scip = scip
    request.sol = sol
    request.kind = kind
    request.intevalvar = cython.NULL
    request.intevalvardata = cython.NULL
    return request


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def record_variables(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    data: cython.pointer[BlockData],
) -> cython.void:
    """Remembers the variables that the children of an entry are, so that a run can fill their inputs.

    The state learns the variables of the block entry by entry, in this instance of SCIP, whose
    variables may differ from those the block was built with. A child that is not a variable
    leaves its input unknown, and a run then does not set it.
    """
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        state.variables[function.columns[function.pointers[data.entry] + index]] = (
            SCIPgetVarExprVar(children[index])
            if SCIPisExprVar(scip, children[index])
            else cython.NULL
        )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def prefetch_inputs(
    state: cython.pointer[BlockState],
    entry: cython.int,
    request: cython.pointer[Prefetch],
) -> cython.void:
    """Fills the inputs at the variables that an entry does not read from SCIP's state.

    A run covers every entry of the block, and so the entries that SCIP asks about next find
    their children at the values the run read. If they were set to something else, those
    entries run the program again. Nothing here is needed for correctness, since every entry
    checks its own children.
    """
    function: cython.pointer[BlockFunction] = state.function
    variables: cython.pointer[cython.pointer[SCIP_VAR]] = state.variables
    if request.kind == PREFETCH_NONE or not state.switches.prefetch:
        return
    x: cython.p_double = as_doubles(state.inputs[SLOT_X].host)
    lower: cython.p_double = as_doubles(state.inputs[SLOT_LOWER].host)
    upper: cython.p_double = as_doubles(state.inputs[SLOT_UPPER].host)
    position: cython.int = function.pointers[entry]
    stop: cython.int = function.pointers[entry + 1]
    column: cython.int
    variable: cython.pointer[SCIP_VAR]
    interval: SCIP_INTERVAL
    for column in range(function.n):
        if position < stop and function.columns[position] == column:
            position += 1
            continue
        variable = variables[column]
        if variable == cython.NULL:
            continue
        if request.kind == PREFETCH_POINT:
            x[column] = SCIPgetSolVal(request.scip, request.sol, variable)
        elif request.kind == PREFETCH_BOX:
            interval = request.intevalvar(
                request.scip, variable, request.intevalvardata
            )
            lower[column] = interval.inf
            upper[column] = interval.sup
        else:
            lower[column] = SCIPvarGetLbLocal(variable)
            upper[column] = SCIPvarGetUbLocal(variable)
            x[column] = min(
                max(SCIPgetSolVal(request.scip, cython.NULL, variable), lower[column]),
                upper[column],
            )


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ensure(
    state: cython.pointer[BlockState],
    entry: cython.int,
    program: cython.int,
    callback: cython.int,
    request: cython.pointer[Prefetch],
) -> cython.int:
    """Returns the stored run of a program that is valid for an entry at the current inputs.

    A run that is not stored yet is executed on the inputs of the whole block and stored with
    the inputs it read. Before the run, `request` says where the inputs of the other entries
    come from. Returns -1 if the program does not exist or fails.
    """
    function: cython.pointer[BlockFunction] = state.function
    cache: cython.pointer[RunCache] = cython.address(state.caches[program])
    if function.programs[program].handle == cython.NULL:
        return -1
    slot: cython.int
    # The caches of values, bounds and estimators can be switched off. The others avoid
    # running a program once for every child of an expression.
    use_caches: cython.bint = state.switches.caches or not (
        program == PROGRAM_CONVEX
        or program == PROGRAM_CONCAVE
        or program == PROGRAM_INTERVAL
    )
    if use_caches:
        for slot in range(cache.count):
            if entry_matches(state, entry, program, slot):
                function.statistics.hits[callback] += 1
                return slot
    if request != cython.NULL:
        prefetch_inputs(state, entry, request)
    slot = cache.next
    second_length: cython.int = second_size(function, program)
    if (
        execute_program(
            state,
            program,
            cache.first + slot * first_size(function, program),
            cache.second + slot * second_length if second_length > 0 else cython.NULL,
            callback,
        )
        != 0
    ):
        # A program fails to run only if the plugin fails.
        return -1  # pragma: no cover
    count: cython.int = key_count(program)
    key: cython.int
    for key in range(count):
        memcpy(
            cache.keys + (slot * count + key) * function.n,
            as_doubles(state.inputs[key_slot(program, key)].host),
            function.n * cython.sizeof(cython.double),
        )
    cache.next = (slot + 1) % run_slots(program)
    cache.count = min(cache.count + 1, run_slots(program))
    return slot


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_values(
    expr: cython.pointer[SCIP_EXPR],
    columns: cython.pointer[cython.int],
    destination: cython.p_double,
) -> cython.bint:
    """Copies the children's evaluation values into their columns and returns whether all are valid."""
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        destination[columns[index]] = SCIPexprGetEvalValue(children[index])
        # The children are variable expressions, whose values are never invalid.
        if destination[columns[index]] == SCIP_INVALID:
            return False  # pragma: no cover
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_dots(
    expr: cython.pointer[SCIP_EXPR],
    columns: cython.pointer[cython.int],
    destination: cython.p_double,
) -> cython.bint:
    """Copies the children's directional derivatives into their columns and returns whether all are valid."""
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    for index in range(SCIPexprGetNChildren(expr)):
        destination[columns[index]] = SCIPexprGetDot(children[index])
        # The children are variable expressions, whose derivatives are never invalid.
        if destination[columns[index]] == SCIP_INVALID:
            return False  # pragma: no cover
    return True


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def read_box(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    columns: cython.pointer[cython.int],
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
        lower[columns[index]] = activity.inf
        upper[columns[index]] = activity.sup
    return True


@cython.cfunc
@cython.inline
@cython.nogil
@cython.exceptval(check=False)
def entry_columns(
    data: cython.pointer[BlockData],
) -> cython.pointer[cython.int]:
    """Returns the columns of the children of an entry, which are variables of the block."""
    function: cython.pointer[BlockFunction] = data.state.function
    return function.columns + function.pointers[data.entry]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def eval_callback(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[SCIP_EXPR],
    val: cython.pointer[SCIP_Real],
    sol: cython.pointer[SCIP_SOL],
) -> SCIP_RETCODE:
    """Evaluates the entry at the children's values, together with its gradient."""
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    function.statistics.calls[CALLBACK_EVAL] += 1
    val[0] = SCIP_INVALID
    request: Prefetch = new_request(scip, sol, PREFETCH_POINT)
    record_variables(scip, expr, data)
    if not read_values(
        expr, entry_columns(data), as_doubles(state.inputs[SLOT_X].host)
    ):
        return SCIP_OKAY  # pragma: no cover
    slot: cython.int = ensure(
        state,
        data.entry,
        PROGRAM_VALUE_AND_GRADIENT,
        CALLBACK_EVAL,
        cython.address(request),
    )
    if slot < 0 or not entry_gradient_finite(state, data.entry, slot):
        return SCIP_OKAY  # pragma: no cover
    val[0] = state.caches[PROGRAM_VALUE_AND_GRADIENT].first[
        slot * function.rows + data.entry
    ]
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def entry_gradient_finite(
    state: cython.pointer[BlockState], entry: cython.int, slot: cython.int
) -> cython.bint:
    """Returns whether the value and the gradient of an entry in a stored run are finite."""
    function: cython.pointer[BlockFunction] = state.function
    cache: cython.pointer[RunCache] = cython.address(
        state.caches[PROGRAM_VALUE_AND_GRADIENT]
    )
    return all_finite(cache.first + slot * function.rows + entry, 1) and all_finite(
        cache.second + slot * function.nnz + function.pointers[entry],
        function.pointers[entry + 1] - function.pointers[entry],
    )


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
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    function.statistics.calls[CALLBACK_BWDIFF] += 1
    val[0] = SCIP_INVALID
    if not read_values(
        expr, entry_columns(data), as_doubles(state.inputs[SLOT_X].host)
    ):
        return SCIP_OKAY  # pragma: no cover
    slot: cython.int = ensure(
        state,
        data.entry,
        PROGRAM_VALUE_AND_GRADIENT,
        CALLBACK_BWDIFF,
        cython.NULL,
    )
    if slot < 0 or not entry_gradient_finite(state, data.entry, slot):
        return SCIP_OKAY  # pragma: no cover
    val[0] = state.caches[PROGRAM_VALUE_AND_GRADIENT].second[
        slot * function.nnz + function.pointers[data.entry] + childidx
    ]
    return SCIP_OKAY


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
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    columns: cython.pointer[cython.int] = entry_columns(data)
    dots: cython.p_double = as_doubles(state.inputs[SLOT_DIRECTION].host)
    total: cython.double = 0.0
    index: cython.int
    function.statistics.calls[CALLBACK_FWDIFF] += 1
    dot[0] = SCIP_INVALID
    if not read_values(
        expr, columns, as_doubles(state.inputs[SLOT_X].host)
    ) or not read_dots(expr, columns, dots):
        return SCIP_OKAY  # pragma: no cover
    slot: cython.int = ensure(
        state,
        data.entry,
        PROGRAM_VALUE_AND_GRADIENT,
        CALLBACK_FWDIFF,
        cython.NULL,
    )
    if slot < 0 or not entry_gradient_finite(state, data.entry, slot):
        # SCIP evaluates a point before it differentiates there, and that evaluation
        # already fails if the gradient is not finite.
        return SCIP_OKAY  # pragma: no cover
    gradient: cython.p_double = (
        state.caches[PROGRAM_VALUE_AND_GRADIENT].second
        + slot * function.nnz
        + function.pointers[data.entry]
    )
    for index in range(SCIPexprGetNChildren(expr)):
        if dots[columns[index]] != 0.0:
            total += gradient[index] * dots[columns[index]]
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
    """Returns one entry of the Hessian times the children's directional derivatives."""
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    columns: cython.pointer[cython.int] = entry_columns(data)
    function.statistics.calls[CALLBACK_BWFWDIFF] += 1
    bardot[0] = SCIP_INVALID
    if (
        not state.switches.hessian
        or not read_values(expr, columns, as_doubles(state.inputs[SLOT_X].host))
        or not read_dots(expr, columns, as_doubles(state.inputs[SLOT_DIRECTION].host))
    ):
        return SCIP_OKAY
    slot: cython.int = ensure(
        state,
        data.entry,
        PROGRAM_HESSIAN_VECTOR,
        CALLBACK_BWFWDIFF,
        cython.NULL,
    )
    if slot < 0:
        return SCIP_OKAY
    entries: cython.p_double = (
        state.caches[PROGRAM_HESSIAN_VECTOR].first
        + slot * function.nnz
        + function.pointers[data.entry]
    )
    if all_finite(entries, SCIPexprGetNChildren(expr)):
        bardot[0] = entries[childidx]
    return SCIP_OKAY


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
    """Encloses the entry's range over the children's activities."""
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    function.statistics.calls[CALLBACK_INTEVAL] += 1
    request: Prefetch = new_request(scip, cython.NULL, PREFETCH_BOX)
    request.intevalvar = intevalvar
    request.intevalvardata = intevalvardata
    record_variables(scip, expr, data)
    interval.inf = -SCIP_INTERVAL_INFINITY
    interval.sup = SCIP_INTERVAL_INFINITY
    if not read_box(
        scip,
        expr,
        entry_columns(data),
        as_doubles(state.inputs[SLOT_LOWER].host),
        as_doubles(state.inputs[SLOT_UPPER].host),
    ):
        return SCIP_OKAY
    slot: cython.int = ensure(
        state, data.entry, PROGRAM_INTERVAL, CALLBACK_INTEVAL, cython.address(request)
    )
    if slot < 0:
        return SCIP_OKAY  # pragma: no cover
    set_interval(
        interval,
        state.caches[PROGRAM_INTERVAL].first[slot * function.rows + data.entry],
        state.caches[PROGRAM_INTERVAL].second[slot * function.rows + data.entry],
    )
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def compute_estimator(
    data: cython.pointer[BlockData],
    side: cython.int,
    callback: cython.int,
    constant: cython.p_double,
    coefficients: cython.p_double,
    request: cython.pointer[Prefetch],
) -> cython.bint:
    """Computes a linear estimator of an entry at the `x` input over the box of the bound inputs.

    Side 0 is the underestimator from the convex relaxation and side 1 the overestimator from
    the concave relaxation. Returns whether the estimator is finite with coefficients in range.
    """
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    program: cython.int = PROGRAM_CONCAVE if side == 1 else PROGRAM_CONVEX
    slot: cython.int = ensure(state, data.entry, program, callback, request)
    if slot < 0:
        return False  # pragma: no cover
    x: cython.p_double = as_doubles(state.inputs[SLOT_X].host)
    columns: cython.pointer[cython.int] = entry_columns(data)
    count: cython.int = (
        function.pointers[data.entry + 1] - function.pointers[data.entry]
    )
    value: cython.double = state.caches[program].first[
        slot * function.rows + data.entry
    ]
    source: cython.p_double = (
        state.caches[program].second
        + slot * function.nnz
        + function.pointers[data.entry]
    )
    index: cython.int
    constant[0] = value
    if not isfinite(value):
        return False
    for index in range(count):
        coefficients[index] = source[index]
        if (
            not isfinite(source[index])
            or fabs(source[index]) > MAX_ESTIMATOR_COEFFICIENT
        ):
            return False
        constant[0] -= source[index] * x[columns[index]]
    return True


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
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    columns: cython.pointer[cython.int] = entry_columns(data)
    x: cython.p_double = as_doubles(state.inputs[SLOT_X].host)
    lower: cython.p_double = as_doubles(state.inputs[SLOT_LOWER].host)
    upper: cython.p_double = as_doubles(state.inputs[SLOT_UPPER].host)
    index: cython.int
    is_global: cython.bint = state.switches.global_estimators
    function.statistics.calls[CALLBACK_ESTIMATE] += 1
    success[0] = False
    islocal[0] = True
    for index in range(SCIPexprGetNChildren(expr)):
        if (
            SCIPisInfinity(scip, -localbounds[index].inf)
            or SCIPisInfinity(scip, localbounds[index].sup)
            or refpoint[index] == SCIP_INVALID
        ):
            return SCIP_OKAY
        lower[columns[index]] = localbounds[index].inf
        upper[columns[index]] = localbounds[index].sup

        # The relaxations are defined on the box, and a plane anchored anywhere in it holds on
        # all of it.
        x[columns[index]] = min(
            max(refpoint[index], lower[columns[index]]), upper[columns[index]]
        )
        if (
            localbounds[index].inf != globalbounds[index].inf
            or localbounds[index].sup != globalbounds[index].sup
        ):
            is_global = False
    request: Prefetch = new_request(scip, cython.NULL, PREFETCH_ESTIMATE)
    record_variables(scip, expr, data)
    success[0] = compute_estimator(
        data,
        1 if overestimate else 0,
        CALLBACK_ESTIMATE,
        constant,
        coefs,
        cython.address(request),
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
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    columns: cython.pointer[cython.int] = entry_columns(data)
    x: cython.p_double = as_doubles(state.inputs[SLOT_X].host)
    lower: cython.p_double = as_doubles(state.inputs[SLOT_LOWER].host)
    upper: cython.p_double = as_doubles(state.inputs[SLOT_UPPER].host)
    index: cython.int
    point: cython.int
    fractions: cython.double[NUM_INIT_POINTS] = [0.5, 0.25, 0.75]
    nreturned[0] = 0
    function.statistics.calls[CALLBACK_INITESTIMATES] += 1
    if not state.switches.init_estimates:
        return SCIP_OKAY
    for index in range(SCIPexprGetNChildren(expr)):
        if SCIPisInfinity(scip, -bounds[index].inf) or SCIPisInfinity(
            scip, bounds[index].sup
        ):
            return SCIP_OKAY
        lower[columns[index]] = bounds[index].inf
        upper[columns[index]] = bounds[index].sup
    for point in range(min(NUM_INIT_POINTS, SCIP_EXPR_MAXINITESTIMATES)):
        for index in range(SCIPexprGetNChildren(expr)):
            x[columns[index]] = lower[columns[index]] + fractions[point] * (
                upper[columns[index]] - lower[columns[index]]
            )
        if compute_estimator(
            data,
            1 if overestimate else 0,
            CALLBACK_INITESTIMATES,
            cython.address(constant[nreturned[0]]),
            coefs[nreturned[0]],
            cython.NULL,
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
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    children: cython.pointer[cython.pointer[SCIP_EXPR]] = SCIPexprGetChildren(expr)
    index: cython.int
    function.statistics.calls[CALLBACK_MONOTONICITY] += 1
    result[0] = SCIP_MONOTONE_UNKNOWN
    if (
        not state.switches.monotonicity
        or function.programs[PROGRAM_GRADIENT_INTERVAL].handle == cython.NULL
    ):
        return SCIP_OKAY
    for index in range(SCIPexprGetNChildren(expr)):
        if SCIPevalExprActivity(scip, children[index]) != SCIP_OKAY:
            return SCIP_OKAY  # pragma: no cover
    if not read_box(
        scip,
        expr,
        entry_columns(data),
        as_doubles(state.inputs[SLOT_LOWER].host),
        as_doubles(state.inputs[SLOT_UPPER].host),
    ):
        return SCIP_OKAY
    slot: cython.int = ensure(
        state,
        data.entry,
        PROGRAM_GRADIENT_INTERVAL,
        CALLBACK_MONOTONICITY,
        cython.NULL,
    )
    if slot < 0:
        return SCIP_OKAY  # pragma: no cover
    position: cython.int = (
        slot * function.nnz + function.pointers[data.entry] + childidx
    )
    low: cython.double = state.caches[PROGRAM_GRADIENT_INTERVAL].first[position]
    high: cython.double = state.caches[PROGRAM_GRADIENT_INTERVAL].second[position]
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
def center_box(
    data: cython.pointer[BlockData],
    count: cython.int,
    childrenbounds: cython.pointer[SCIP_INTERVAL],
) -> cython.void:
    """Sets the bound inputs to the children's bounds and the `x` input to the box center."""
    state: cython.pointer[BlockState] = data.state
    columns: cython.pointer[cython.int] = entry_columns(data)
    x: cython.p_double = as_doubles(state.inputs[SLOT_X].host)
    lower: cython.p_double = as_doubles(state.inputs[SLOT_LOWER].host)
    upper: cython.p_double = as_doubles(state.inputs[SLOT_UPPER].host)
    index: cython.int
    for index in range(count):
        lower[columns[index]] = childrenbounds[index].inf
        upper[columns[index]] = childrenbounds[index].sup
        x[columns[index]] = 0.5 * (lower[columns[index]] + upper[columns[index]])


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
    """Tightens the children's bounds from bounds on the entry, with estimators at the box center."""
    data: cython.pointer[BlockData] = get_data(expr)
    state: cython.pointer[BlockState] = data.state
    function: cython.pointer[BlockFunction] = state.function
    count: cython.int = SCIPexprGetNChildren(expr)
    index: cython.int
    constant: cython.double
    weights: cython.p_double = state.scratch
    tolerance: cython.double = SCIPfeastol(scip)
    infeasible[0] = False
    function.statistics.calls[CALLBACK_REVERSEPROP] += 1
    if not state.switches.reverse_propagation:
        return SCIP_OKAY
    for index in range(count):
        if (
            SCIPisInfinity(scip, -childrenbounds[index].inf)
            or SCIPisInfinity(scip, childrenbounds[index].sup)
            or childrenbounds[index].inf > childrenbounds[index].sup
        ):
            return SCIP_OKAY
    center_box(data, count, childrenbounds)
    if (
        bounds.sup < SCIP_INTERVAL_INFINITY
        and compute_estimator(
            data,
            0,
            CALLBACK_REVERSEPROP,
            cython.address(constant),
            weights,
            cython.NULL,
        )
        and tighten_from_plane(
            constant, weights, bounds.sup, count, tolerance, childrenbounds
        )
    ):
        infeasible[0] = True
        return SCIP_OKAY
    if bounds.inf > -SCIP_INTERVAL_INFINITY:
        center_box(data, count, childrenbounds)
        if compute_estimator(
            data,
            1,
            CALLBACK_REVERSEPROP,
            cython.address(constant),
            weights,
            cython.NULL,
        ):
            for index in range(count):
                weights[index] = -weights[index]
            infeasible[0] = tighten_from_plane(
                -constant, weights, -bounds.inf, count, tolerance, childrenbounds
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

    A distinct coefficient per child marks the input order. The constants identify the block and
    the entry, so that no two entries look alike, which only hides symmetries.
    """
    data: cython.pointer[BlockData] = get_data(expr)
    count: cython.int = SCIPexprGetNChildren(expr)
    index: cython.int
    if allocSymData(scip, symdata) != SCIP_OKAY:
        return SCIP_NOMEMORY  # pragma: no cover
    description: cython.pointer[SYM_EXPRDATA] = symdata[0]
    description.nconstants = 3
    description.ncoefficients = count
    if (
        allocRealArray(scip, cython.address(description.constants), 3) != SCIP_OKAY
        or allocRealArray(scip, cython.address(description.coefficients), count)
        != SCIP_OKAY
        or allocExprArray(scip, cython.address(description.children), count)
        != SCIP_OKAY
    ):
        return SCIP_NOMEMORY  # pragma: no cover
    description.constants[0] = cython.cast(
        cython.double, cython.cast(uintptr_t, data.state.function) % 1000003
    )
    description.constants[1] = cython.cast(
        cython.double, cython.cast(uintptr_t, data.state) % 1000003
    )
    description.constants[2] = data.entry
    for index in range(count):
        description.coefficients[index] = index + 1.0
        description.children[index] = SCIPexprGetChildren(expr)[index]
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def copy_state(
    source: cython.pointer[BlockState],
) -> cython.pointer[BlockState]:
    """Returns a new state for the same block, with its parameters and empty caches."""
    return new_block_state(
        source.function, as_doubles(source.inputs[SLOT_PARAMS].host), source.switches
    )


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
    """Copies an entry into another SCIP instance, where the entries of its block share a new state.

    Within one instance the copy shares the state of its source. The handler of the target
    keeps a state for every state it has copied, and a reference to it, so that the next
    entry of the same block finds it.
    """
    source: cython.pointer[BlockData] = get_data(sourceexpr)
    table: cython.pointer[CopyTable] = cython.cast(
        cython.pointer[CopyTable], SCIPexprhdlrGetData(targetexprhdlr)
    )
    state: cython.pointer[BlockState] = source.state
    index: cython.int
    found: cython.bint = False
    if targetscip != sourcescip:
        for index in range(table.count):
            if table.sources[index] == source.state:
                state = table.targets[index]
                found = True
                break
        if not found:
            state = copy_state(source.state)
            if state == cython.NULL:
                return SCIP_NOMEMORY  # pragma: no cover
            if table.count == table.capacity:
                table.capacity = max(2 * table.capacity, 8)
                table.sources = cython.cast(
                    cython.pointer[cython.pointer[BlockState]],
                    realloc(
                        table.sources,
                        table.capacity * cython.sizeof(cython.pointer[BlockState]),
                    ),
                )
                table.targets = cython.cast(
                    cython.pointer[cython.pointer[BlockState]],
                    realloc(
                        table.targets,
                        table.capacity * cython.sizeof(cython.pointer[BlockState]),
                    ),
                )
            # The table keeps the creator's reference of the new state.
            table.sources[table.count] = source.state
            table.targets[table.count] = state
            table.count += 1
    copy: cython.pointer[BlockData] = new_block_data(state, source.entry)
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
    """Frees the data of an entry and drops its reference to the state of its block."""
    data: cython.pointer[BlockData] = get_data(expr)
    release_block_state(data.state)
    free(data)
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
    """Orders entries by block and entry, which decide the children."""
    first: cython.pointer[BlockData] = get_data(expr1)
    second: cython.pointer[BlockData] = get_data(expr2)
    if first.state != second.state:
        return (
            -1
            if cython.cast(uintptr_t, first.state)
            < cython.cast(uintptr_t, second.state)
            else 1
        )
    if first.entry != second.entry:
        return -1 if first.entry < second.entry else 1
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
    """Hashes the block and the entry, which decide the children."""
    data: cython.pointer[BlockData] = get_data(expr)
    address: uintptr_t = cython.cast(uintptr_t, data.state)
    hashkey[0] = (
        cython.cast(cython.uint, address >> 4) ^ cython.cast(cython.uint, address >> 36)
    ) * 31 + cython.cast(cython.uint, data.entry)
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def copyhdlr_callback(
    scip: cython.pointer[SCIP], sourceexprhdlr: cython.pointer[SCIP_EXPRHDLR]
) -> SCIP_RETCODE:
    """Includes the handler into copies of SCIP, which some primal heuristics solve."""
    return include_block_handler(scip)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def freehdlr_callback(
    scip: cython.pointer[SCIP],
    exprhdlr: cython.pointer[SCIP_EXPRHDLR],
    exprhdlrdata: cython.pointer[cython.pointer[SCIP_EXPRHDLRDATA]],
) -> SCIP_RETCODE:
    """Releases the states that the handler copied and frees its table."""
    table: cython.pointer[CopyTable] = cython.cast(
        cython.pointer[CopyTable], exprhdlrdata[0]
    )
    index: cython.int
    for index in range(table.count):
        release_block_state(table.targets[index])
    free(table.sources)
    free(table.targets)
    free(table)
    exprhdlrdata[0] = cython.NULL
    return SCIP_OKAY


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def include_block_handler(scip: cython.pointer[SCIP]) -> SCIP_RETCODE:
    """Includes the handler of block entries into a SCIP instance unless it is already there."""
    if SCIPfindExprhdlr(scip, BLOCK_HANDLER_NAME) != cython.NULL:
        return SCIP_OKAY
    table: cython.pointer[CopyTable] = cython.cast(
        cython.pointer[CopyTable], calloc(1, cython.sizeof(CopyTable))
    )
    if table == cython.NULL:
        return SCIP_NOMEMORY  # pragma: no cover
    handler: cython.pointer[SCIP_EXPRHDLR]
    retcode: SCIP_RETCODE = SCIPincludeExprhdlr(
        scip,
        cython.address(handler),
        BLOCK_HANDLER_NAME,
        b"entry of a vector JAX function executed through PJRT",
        0,
        eval_callback,
        cython.cast(cython.pointer[SCIP_EXPRHDLRDATA], table),
    )
    if retcode != SCIP_OKAY:
        free(table)  # pragma: no cover
        return retcode  # pragma: no cover
    SCIPexprhdlrSetCopyFreeHdlr(handler, copyhdlr_callback, freehdlr_callback)
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
def create_block_expression(
    scip: cython.pointer[SCIP],
    expr: cython.pointer[cython.pointer[SCIP_EXPR]],
    state: cython.pointer[BlockState],
    entry: cython.int,
    children: cython.pointer[cython.pointer[SCIP_EXPR]],
) -> SCIP_RETCODE:
    """Creates the expression of an entry of a block, including the handler if needed."""
    retcode: SCIP_RETCODE = include_block_handler(scip)
    if retcode != SCIP_OKAY:
        return retcode  # pragma: no cover
    data: cython.pointer[BlockData] = new_block_data(state, entry)
    if data == cython.NULL:
        return SCIP_NOMEMORY  # pragma: no cover
    retcode = SCIPcreateExpr(
        scip,
        expr,
        SCIPfindExprhdlr(scip, BLOCK_HANDLER_NAME),
        cython.cast(cython.pointer[SCIP_EXPRDATA], data),
        state.function.pointers[entry + 1] - state.function.pointers[entry],
        children,
        cython.NULL,
        cython.NULL,
    )
    if retcode != SCIP_OKAY:
        release_block_state(state)  # pragma: no cover
        free(data)  # pragma: no cover
    return retcode


def check_length(name: str, values: object, expected: int) -> None:
    """Raises a `ValueError` unless an array has the expected length."""
    actual: int = len(values)
    if actual != expected:
        raise ValueError(f"{name} has {actual} entries, expected {expected}")


@cython.cfunc
def copy_input(
    state: cython.pointer[BlockState], slot: cython.int, values: cython.double[::1]
) -> cython.void:
    """Copies an array into the host memory of an input slot of a block state."""
    check_length("input", values, state.function.n)
    memcpy(
        as_doubles(state.inputs[slot].host),
        cython.address(values[0]),
        state.function.n * cython.sizeof(cython.double),
    )


DIRECT_SWITCHES = HandlerOptions(caches=False)
"""Switches of the direct calls of `BlockHandle`, which never reuse results."""


@cython.cclass
class BlockHandle:
    """The executables of one block term and its pattern, shared by every instance of SCIP.

    A SCIP problem keeps the handles of its expressions alive for as long as it exists.
    """

    def __cinit__(
        self,
        num_inputs: cython.int,
        num_params: cython.int,
        pointers: cython.int[::1],
        columns: cython.int[::1],
        programs: Programs,
    ) -> None:
        rows: cython.int = pointers.shape[0] - 1
        if num_inputs < 1 or rows < 1:
            raise ValueError("A block needs at least one input and one entry")
        if pointers[0] != 0 or pointers[rows] != columns.shape[0]:
            raise ValueError("The pointers of a block do not match its columns")
        if columns.shape[0] == 0:
            raise ValueError("Every entry of a block needs a variable")
        self.num_inputs = num_inputs
        self.num_params = max(num_params, 1)
        self.num_entries = rows
        self.nnz = columns.shape[0]
        self.programs = programs
        function: cython.pointer[BlockFunction] = cython.cast(
            cython.pointer[BlockFunction], calloc(1, cython.sizeof(BlockFunction))
        )
        self.function = function
        function.n = num_inputs
        function.m = self.num_params
        function.rows = rows
        function.nnz = self.nnz
        function.pointers = cython.cast(
            cython.pointer[cython.int], malloc((rows + 1) * cython.sizeof(cython.int))
        )
        function.columns = cython.cast(
            cython.pointer[cython.int], malloc(self.nnz * cython.sizeof(cython.int))
        )
        memcpy(
            function.pointers,
            cython.address(pointers[0]),
            (rows + 1) * cython.sizeof(cython.int),
        )
        memcpy(
            function.columns,
            cython.address(columns[0]),
            self.nnz * cython.sizeof(cython.int),
        )
        entry: cython.int
        position: cython.int
        for entry in range(rows):
            if function.pointers[entry + 1] <= function.pointers[entry]:
                raise ValueError("Every entry of a block needs a variable")
            for position in range(
                function.pointers[entry], function.pointers[entry + 1]
            ):
                if (
                    function.columns[position] < 0
                    or function.columns[position] >= num_inputs
                ):
                    raise IndexError("The columns of a block index missing inputs")

        executable: Executable
        index: cython.int
        for program in Program:
            executable = programs[program]
            if executable is None:
                if program not in (Program.HESSIAN_VECTOR, Program.GRADIENT_INTERVAL):
                    raise ValueError(f"The {program.name.lower()} program is required")
                continue
            sizes = block_output_sizes(program, rows, self.nnz)
            slots = BLOCK_SLOTS[program]
            if executable.num_outputs != len(sizes):
                raise ValueError(
                    f"The {program.name.lower()} program has {executable.num_outputs} outputs, "
                    f"expected {len(sizes)}"
                )
            index = program
            function.programs[index].handle = executable.handle
            function.programs[index].num_inputs = len(slots)
            for position, slot in enumerate(slots):
                function.slots[index][position] = slot
            function.programs[index].num_outputs = len(sizes)
            for position, size in enumerate(sizes):
                function.programs[index].output_bytes[position] = size * cython.sizeof(
                    cython.double
                )

    def __dealloc__(self) -> None:
        function: cython.pointer[BlockFunction] = self.function
        if function == cython.NULL:
            return  # pragma: no cover
        free(function.pointers)
        free(function.columns)
        free(function)

    @property
    def pointers(self) -> np.ndarray:
        """Where the children of each entry start among the columns, then the number of columns."""
        result = np.empty(self.num_entries + 1, np.intc)
        view: cython.int[::1] = result
        memcpy(
            cython.address(view[0]),
            self.function.pointers,
            (self.num_entries + 1) * cython.sizeof(cython.int),
        )
        return result

    @property
    def columns(self) -> np.ndarray:
        """The input that each child of each entry is, entry after entry."""
        result = np.empty(self.nnz, np.intc)
        view: cython.int[::1] = result
        memcpy(
            cython.address(view[0]),
            self.function.columns,
            self.nnz * cython.sizeof(cython.int),
        )
        return result

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

    def run(
        self,
        program: Program,
        x: cython.double[::1],
        params: cython.double[::1],
        direction: object = None,
        lower: object = None,
        upper: object = None,
    ) -> tuple[np.ndarray, ...]:
        """Runs one program once on whole vectors and returns its outputs."""
        check_length("params", params, self.num_params)
        index: cython.int = program
        first_length: cython.int = first_size(self.function, index)
        second_length: cython.int = second_size(self.function, index)
        first = np.empty(first_length)
        second = np.empty(max(second_length, 1))
        first_view: cython.double[::1] = first
        second_view: cython.double[::1] = second
        state: cython.pointer[BlockState] = new_block_state(
            self.function, cython.address(params[0]), switches_from(DIRECT_SWITCHES)
        )
        if state == cython.NULL:
            raise MemoryError()  # pragma: no cover
        copy_input(state, SLOT_X, x)
        if direction is not None:
            copy_input(
                state, SLOT_DIRECTION, np.ascontiguousarray(direction, np.float64)
            )
        if lower is not None:
            copy_input(state, SLOT_LOWER, np.ascontiguousarray(lower, np.float64))
        if upper is not None:
            copy_input(state, SLOT_UPPER, np.ascontiguousarray(upper, np.float64))
        status: cython.int = execute_program(
            state,
            index,
            cython.address(first_view[0]),
            cython.address(second_view[0]) if second_length > 0 else cython.NULL,
            CALLBACK_EVAL,
        )
        release_block_state(state)
        if status != 0:
            if self.function.programs[index].handle == cython.NULL:
                raise ValueError(f"The block has no {program.name.lower()} program")
            raise_last_error(program.name.lower())  # pragma: no cover
        if second_length > 0:
            return first, second
        return (first,)

    def value_and_jacobian(
        self, x: cython.double[::1], params: cython.double[::1]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the values of all entries and their derivatives in the order of the pattern."""
        first, second = self.run(Program.VALUE_AND_GRADIENT, x, params)
        return first, second

    def hessian_vector(
        self,
        x: cython.double[::1],
        direction: cython.double[::1],
        params: cython.double[::1],
    ) -> np.ndarray:
        """Returns the derivative of the Jacobian entries in `x` along `direction`."""
        return self.run(Program.HESSIAN_VECTOR, x, params, direction)[0]

    def relaxation(
        self,
        x: cython.double[::1],
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
        overestimate: cython.bint = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns the values of the convex or concave relaxations and their gradients."""
        program = Program.CONCAVE if overestimate else Program.CONVEX
        first, second = self.run(program, x, params, None, lower, upper)
        return first, second

    def estimator(
        self,
        entry: cython.int,
        reference: cython.double[::1],
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
        overestimate: cython.bint,
    ) -> Estimator:
        """Computes the estimator of one entry at a reference point clamped to the box.

        The coefficients are those of the entry's children, as `ESTIMATE` returns them.
        """
        if entry < 0 or entry >= self.num_entries:
            raise IndexError(
                f"The block has {self.num_entries} entries, not {entry + 1}"
            )
        check_length("reference", reference, self.num_inputs)
        check_length("lower", lower, self.num_inputs)
        check_length("upper", upper, self.num_inputs)
        check_length("params", params, self.num_params)
        state: cython.pointer[BlockState] = new_block_state(
            self.function, cython.address(params[0]), switches_from(DIRECT_SWITCHES)
        )
        if state == cython.NULL:
            raise MemoryError()  # pragma: no cover
        index: cython.int
        for index in range(self.num_inputs):
            as_doubles(state.inputs[SLOT_LOWER].host)[index] = lower[index]
            as_doubles(state.inputs[SLOT_UPPER].host)[index] = upper[index]
            as_doubles(state.inputs[SLOT_X].host)[index] = min(
                max(reference[index], lower[index]), upper[index]
            )
        data: BlockData
        data.state = state
        data.entry = entry
        count: cython.int = (
            self.function.pointers[entry + 1] - self.function.pointers[entry]
        )
        coefficients = np.empty(count)
        view: cython.double[::1] = coefficients
        constant: cython.double = 0.0
        success: cython.bint = compute_estimator(
            cython.address(data),
            1 if overestimate else 0,
            CALLBACK_ESTIMATE,
            cython.address(constant),
            cython.address(view[0]),
            cython.NULL,
        )
        release_block_state(state)
        return Estimator(
            success=bool(success), constant=constant, coefficients=coefficients
        )

    def interval(
        self,
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
    ) -> Interval:
        """Returns bounds on the value of each entry over the box."""
        first, second = self.run(Program.INTERVAL, lower, params, None, lower, upper)
        return Interval(lower=first, upper=second)

    def gradient_interval(
        self,
        lower: cython.double[::1],
        upper: cython.double[::1],
        params: cython.double[::1],
    ) -> Interval:
        """Returns bounds on each derivative in the pattern over the box."""
        first, second = self.run(
            Program.GRADIENT_INTERVAL, lower, params, None, lower, upper
        )
        return Interval(lower=first, upper=second)


@cython.cclass
class BlockStateRef:
    """The state of one block in one SCIP instance, which its entries share.

    The state lives for as long as this object or an expression that uses it does.
    """

    def __cinit__(
        self,
        handle: BlockHandle,
        params: cython.double[::1],
        inputs: cython.int[::1],
        options: HandlerOptions,
    ) -> None:
        check_length("params", params, handle.num_params)
        check_length("inputs", inputs, handle.num_inputs)
        self.handle = handle
        self.inputs = np.asarray(inputs, np.intc).copy()
        self.state = new_block_state(
            handle.function, cython.address(params[0]), switches_from(options)
        )
        if self.state == cython.NULL:
            raise MemoryError()  # pragma: no cover

    def __dealloc__(self) -> None:
        if self.state != cython.NULL:
            release_block_state(self.state)
