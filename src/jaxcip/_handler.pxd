from cyxla cimport Input, MAX_INPUTS, ProgramSpec
from jaxcip._scip cimport SCIP, SCIP_EXPR, SCIP_INTERVAL, SCIP_RETCODE


cdef enum:
    VALUE_CACHE_SIZE = 4
    BOUNDS_CACHE_SIZE = 4
    ESTIMATE_CACHE_SIZE = 2
    BOX_STORE_SIZE = 8
    NUM_INIT_POINTS = 3


# Inputs of the programs. Every expression of a function shares the host arrays of all slots
# but the parameters, the warm start of roots and the enclosures of roots, which each
# expression owns.
cdef enum:
    SLOT_X = 0
    SLOT_DIRECTION = 1
    SLOT_LOWER = 2
    SLOT_UPPER = 3
    SLOT_PARAMS = 4
    SLOT_STATE = 5
    SLOT_ENCLOSURE = 6
    NUM_SLOTS = 7


# The programs of a function, with the values of `jaxcip._types.Program`.
cdef enum:
    PROGRAM_VALUE_AND_GRADIENT = 0
    PROGRAM_HESSIAN_VECTOR = 1
    PROGRAM_CONVEX = 2
    PROGRAM_CONCAVE = 3
    PROGRAM_INTERVAL = 4
    PROGRAM_GRADIENT_INTERVAL = 5
    NUM_PROGRAMS = 6


# The counted callbacks, with the values of `jaxcip._types.Callback`.
cdef enum:
    CALLBACK_EVAL = 0
    CALLBACK_BWDIFF = 1
    CALLBACK_FWDIFF = 2
    CALLBACK_BWFWDIFF = 3
    CALLBACK_INTEVAL = 4
    CALLBACK_ESTIMATE = 5
    CALLBACK_INITESTIMATES = 6
    CALLBACK_MONOTONICITY = 7
    CALLBACK_REVERSEPROP = 8
    NUM_CALLBACKS = 9


cdef struct Statistics:
    long calls[NUM_CALLBACKS]
    long hits[NUM_CALLBACKS]
    long runs[NUM_CALLBACKS]
    double seconds[NUM_CALLBACKS]


cdef struct Switches:
    # The fields of `jaxcip._options.HandlerOptions`.
    bint caches
    bint init_estimates
    bint monotonicity
    bint global_estimators
    bint reverse_propagation
    bint hessian
    bint zero_copy
    bint prefetch


cdef struct Function:
    # The programs of one function structure and the host inputs its expressions share.
    int n
    int m
    int state_size
    int enclosure_size
    ProgramSpec programs[NUM_PROGRAMS]
    int slots[NUM_PROGRAMS][MAX_INPUTS]
    Input inputs[NUM_SLOTS]
    double* scratch
    Statistics statistics


cdef struct ExprData:
    # One expression's parameters, switches and caches of recent results.
    Function* function
    Switches switches
    Input params
    Input state
    double* state_next
    Input enclosure
    double* enclosure_next
    double* box_keys
    double* box_enclosures
    int box_count
    int box_next
    double* cache_block
    double* value_keys
    double* value_results
    int value_count
    int value_next
    double* gradient_key
    double* gradient
    double gradient_value
    bint gradient_valid
    double* hessian_key
    double* hessian
    bint hessian_valid
    double* bounds_keys
    double* bounds_results
    int bounds_count
    int bounds_next
    double* estimate_keys[2]
    double* estimate_results[2]
    bint estimate_success[2][ESTIMATE_CACHE_SIZE]
    int estimate_count[2]
    int estimate_next[2]
    double* monotonicity_key
    double* monotonicity_result
    bint monotonicity_valid


cdef double* as_doubles(void* memory) noexcept nogil
cdef double* aligned_doubles(int count) noexcept nogil
cdef void create_input(Input* host_input, int count) noexcept nogil
cdef void destroy_input(Input* host_input) noexcept nogil
cdef ExprData* new_expr_data(Function* function, const double* params, Switches switches) noexcept nogil
cdef void free_expr_data(ExprData* data) noexcept nogil
cdef SCIP_RETCODE include_handler(SCIP* scip) noexcept nogil
cdef SCIP_RETCODE create_expression(SCIP* scip, SCIP_EXPR** expr, ExprData* data, SCIP_EXPR** children) noexcept nogil
cdef Switches switches_from(object options)
cdef bint same(double* first, double* second, int count) noexcept nogil
cdef bint all_finite(double* values, int count) noexcept nogil
cdef void set_interval(SCIP_INTERVAL* interval, double low, double high) noexcept nogil
cdef bint tighten_from_plane(double constant, double* weights, double rhs, int n, double tolerance, SCIP_INTERVAL* childrenbounds) noexcept nogil


cdef class FunctionHandle:
    cdef Function* function
    cdef readonly int num_inputs
    cdef readonly int num_params
    cdef readonly int state_size
    cdef readonly int enclosure_size
    cdef readonly object programs
