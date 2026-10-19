"""Declarations for forwarding OpenBLAS to libblastrampoline, through which Ipopt calls BLAS."""


cdef extern from "libblastrampoline.h" nogil:
    int lbt_forward(const char* libname, int clear, int verbose, const char* suffix_hint)


cdef extern from "dlfcn.h" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
    int dladdr(const void* address, Dl_info* info)


cdef extern from *:
    """
    void openblas_set_num_threads(int);
    """
    void openblas_set_num_threads(int count) nogil
