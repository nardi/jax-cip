from libc.stdint cimport int64_t

from jaxcip._pjrt cimport (
    PJRT_Api,
    PJRT_Buffer,
    PJRT_Client,
    PJRT_Device,
    PJRT_Error,
    PJRT_Event,
    PJRT_Executable,
    PJRT_LoadedExecutable,
)


cdef enum:
    ALIGNMENT = 64
    MAX_INPUTS = 5
    MAX_OUTPUTS = 3


cdef struct ProgramSpec:
    # An executable with its number of inputs and the number of doubles in each output.
    PJRT_LoadedExecutable* handle
    int num_inputs
    int num_outputs
    int output_sizes[MAX_OUTPUTS]


cdef struct Input:
    # A host array of `size` doubles and its aliased buffer, which is NULL when the plugin
    # cannot alias host memory.
    double* host
    PJRT_Buffer* buffer
    int64_t size


ctypedef const PJRT_Api* (*GetPjrtApi)() noexcept nogil


# Runtime state shared by all functions. SCIP calls the handler from one thread at a time.
cdef const PJRT_Api* api
cdef PJRT_Client* client
cdef PJRT_Device* device
cdef bint can_alias
cdef bint can_read_inplace
cdef char error_message[1024]
cdef long error_count


cdef double now() noexcept nogil
cdef int check(PJRT_Error* error) noexcept nogil
cdef PJRT_Buffer* alias_buffer(double* data, int64_t size) noexcept nogil
cdef void destroy_buffer(PJRT_Buffer* buffer) noexcept nogil
cdef double* aligned_doubles(int count) noexcept nogil
cdef int await_event(PJRT_Event* event) noexcept nogil
cdef void destroy_event(PJRT_Event* event) noexcept nogil
cdef PJRT_Buffer* buffer_from_host(double* data, int64_t size, bint alias) noexcept nogil
cdef void destroy_executable(PJRT_Executable* executable) noexcept nogil
cdef int execute(ProgramSpec* program, Input* inputs, double** outputs, bint zero_copy) noexcept nogil


cdef class Executable:
    cdef PJRT_LoadedExecutable* handle
    cdef readonly int num_outputs

    @staticmethod
    cdef Executable wrap(PJRT_LoadedExecutable* handle)
