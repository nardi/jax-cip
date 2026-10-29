from cyxla._runtime cimport Input, MAX_INPUTS, ProgramSpec
from jaxcip._handler cimport NUM_PROGRAMS, Statistics, Switches
from jaxcip._scip cimport ExprIntevalVar, SCIP, SCIP_EXPR, SCIP_RETCODE, SCIP_SOL, SCIP_VAR


cdef enum:
    # Most runs of one program that a block remembers.
    MAX_RUN_SLOTS = 4
    # The inputs of a block's programs, the first five slots of `_handler.pxd`.
    NUM_BLOCK_INPUTS = 5

cdef enum:
    # What a miss fills the inputs of all entries from.
    PREFETCH_NONE = 0
    PREFETCH_POINT = 1
    PREFETCH_BOX = 2
    PREFETCH_ESTIMATE = 3


cdef struct BlockFunction:
    # The programs of a block term and its pattern. A block function is read-only and
    # shared by every SCIP instance that has the block.
    int n
    int m
    int rows
    int nnz
    int* pointers
    int* columns
    ProgramSpec programs[NUM_PROGRAMS]
    int slots[NUM_PROGRAMS][MAX_INPUTS]
    Statistics statistics


cdef struct RunCache:
    # The most recent runs of one program, with the inputs each run read.
    int count
    int next
    double* keys
    double* first
    double* second


cdef struct BlockState:
    # The host inputs and the recent runs of one block in one SCIP instance, which all of
    # the block's expressions share.
    BlockFunction* function
    int references
    Switches switches
    Input inputs[NUM_BLOCK_INPUTS]
    RunCache caches[NUM_PROGRAMS]
    double* scratch
    # The variable of each input, or NULL where SCIP's state cannot fill the inputs.
    SCIP_VAR** variables


cdef struct Prefetch:
    # Where the other entries of a block get their inputs from when one entry misses.
    SCIP* scip
    SCIP_SOL* sol
    int kind
    ExprIntevalVar intevalvar
    void* intevalvardata


cdef struct BlockData:
    # The data of one expression: an entry of a block.
    BlockState* state
    int entry


cdef struct CopyTable:
    # The states that a handler made when SCIP copied entries into its instance, with the
    # states they are copies of.
    int count
    int capacity
    BlockState** sources
    BlockState** targets


cdef SCIP_RETCODE include_block_handler(SCIP* scip) noexcept nogil
cdef SCIP_RETCODE create_block_expression(SCIP* scip, SCIP_EXPR** expr, BlockState* state, int entry, SCIP_EXPR** children) noexcept nogil


cdef class BlockHandle:
    cdef BlockFunction* function
    cdef readonly int num_inputs
    cdef readonly int num_params
    cdef readonly int num_entries
    cdef readonly int nnz
    cdef readonly object programs


cdef class BlockStateRef:
    cdef BlockState* state
    cdef readonly BlockHandle handle
    cdef readonly object inputs
