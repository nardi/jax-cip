"""Declarations of the part of the PJRT C API that the extension modules use.

The declarations follow `xla/pjrt/c/pjrt_c_api.h` (vendored under `third_party/xla`). Cython
only needs the fields that the module reads or writes, so each struct lists a subset of its
fields, and the C compiler uses the real layout from the header.
"""

from libc.stdint cimport int64_t, uintptr_t


cdef extern from "xla/pjrt/c/pjrt_c_api.h" nogil:
    ctypedef struct PJRT_Error
    ctypedef struct PJRT_Client
    ctypedef struct PJRT_Device
    ctypedef struct PJRT_Buffer
    ctypedef struct PJRT_Event
    ctypedef struct PJRT_LoadedExecutable
    ctypedef struct PJRT_Executable
    ctypedef struct PJRT_SerializedExecutable
    ctypedef struct PJRT_Extension_Base

    ctypedef enum PJRT_Buffer_Type:
        PJRT_Buffer_Type_F64

    ctypedef enum PJRT_HostBufferSemantics:
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall
        PJRT_HostBufferSemantics_kImmutableZeroCopy

    ctypedef enum PJRT_NamedValue_Type:
        PJRT_NamedValue_kInt64
        PJRT_NamedValue_kBool

    # Each argument struct has a matching enum constant with its size for this header version.
    enum:
        PJRT_Error_Message_Args_STRUCT_SIZE
        PJRT_Error_Destroy_Args_STRUCT_SIZE
        PJRT_Plugin_Initialize_Args_STRUCT_SIZE
        PJRT_NamedValue_STRUCT_SIZE
        PJRT_Client_Create_Args_STRUCT_SIZE
        PJRT_Client_PlatformName_Args_STRUCT_SIZE
        PJRT_Client_AddressableDevices_Args_STRUCT_SIZE
        PJRT_Program_STRUCT_SIZE
        PJRT_Client_Compile_Args_STRUCT_SIZE
        PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
        PJRT_Buffer_Destroy_Args_STRUCT_SIZE
        PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE
        PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
        PJRT_Event_Await_Args_STRUCT_SIZE
        PJRT_Event_Destroy_Args_STRUCT_SIZE
        PJRT_ExecuteOptions_STRUCT_SIZE
        PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
        PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE
        PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE
        PJRT_Executable_Destroy_Args_STRUCT_SIZE
        PJRT_Executable_NumOutputs_Args_STRUCT_SIZE
        PJRT_Executable_Serialize_Args_STRUCT_SIZE
        PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE

    ctypedef struct PJRT_Error_Message_Args:
        size_t struct_size
        const PJRT_Error* error
        const char* message
        size_t message_size

    ctypedef struct PJRT_Error_Destroy_Args:
        size_t struct_size
        PJRT_Error* error

    ctypedef struct PJRT_Plugin_Initialize_Args:
        size_t struct_size

    ctypedef struct PJRT_NamedValue:
        size_t struct_size
        const char* name
        size_t name_size
        PJRT_NamedValue_Type type
        int64_t int64_value
        bint bool_value
        size_t value_size

    ctypedef struct PJRT_Client_Create_Args:
        size_t struct_size
        const PJRT_NamedValue* create_options
        size_t num_options
        PJRT_Client* client

    ctypedef struct PJRT_Client_PlatformName_Args:
        size_t struct_size
        PJRT_Client* client
        const char* platform_name
        size_t platform_name_size

    ctypedef struct PJRT_Client_AddressableDevices_Args:
        size_t struct_size
        PJRT_Client* client
        PJRT_Device* const* addressable_devices
        size_t num_addressable_devices

    ctypedef struct PJRT_Program:
        size_t struct_size
        char* code
        size_t code_size
        const char* format
        size_t format_size

    ctypedef struct PJRT_Client_Compile_Args:
        size_t struct_size
        PJRT_Client* client
        const PJRT_Program* program
        const char* compile_options
        size_t compile_options_size
        PJRT_LoadedExecutable* executable

    ctypedef struct PJRT_Client_BufferFromHostBuffer_Args:
        size_t struct_size
        PJRT_Client* client
        const void* data
        PJRT_Buffer_Type type
        const int64_t* dims
        size_t num_dims
        PJRT_HostBufferSemantics host_buffer_semantics
        PJRT_Device* device
        PJRT_Event* done_with_host_buffer
        PJRT_Buffer* buffer

    ctypedef struct PJRT_Buffer_Destroy_Args:
        size_t struct_size
        PJRT_Buffer* buffer

    ctypedef struct PJRT_Buffer_UnsafePointer_Args:
        size_t struct_size
        PJRT_Buffer* buffer
        uintptr_t buffer_pointer

    ctypedef struct PJRT_Buffer_ToHostBuffer_Args:
        size_t struct_size
        PJRT_Buffer* src
        void* dst
        size_t dst_size
        PJRT_Event* event

    ctypedef struct PJRT_Event_Await_Args:
        size_t struct_size
        PJRT_Event* event

    ctypedef struct PJRT_Event_Destroy_Args:
        size_t struct_size
        PJRT_Event* event

    ctypedef struct PJRT_ExecuteOptions:
        size_t struct_size

    ctypedef struct PJRT_LoadedExecutable_Execute_Args:
        size_t struct_size
        PJRT_LoadedExecutable* executable
        PJRT_ExecuteOptions* options
        PJRT_Buffer* const* const* argument_lists
        size_t num_devices
        size_t num_args
        PJRT_Buffer** const* output_lists
        PJRT_Event** device_complete_events
        PJRT_Device* execute_device

    ctypedef struct PJRT_LoadedExecutable_Destroy_Args:
        size_t struct_size
        PJRT_LoadedExecutable* executable

    ctypedef struct PJRT_LoadedExecutable_GetExecutable_Args:
        size_t struct_size
        PJRT_LoadedExecutable* loaded_executable
        PJRT_Executable* executable

    ctypedef struct PJRT_Executable_Destroy_Args:
        size_t struct_size
        PJRT_Executable* executable

    ctypedef struct PJRT_Executable_NumOutputs_Args:
        size_t struct_size
        PJRT_Executable* executable
        size_t num_outputs

    ctypedef struct PJRT_Executable_Serialize_Args:
        size_t struct_size
        const PJRT_Executable* executable
        const char* serialized_bytes
        size_t serialized_bytes_size
        PJRT_SerializedExecutable* serialized_executable
        void (*serialized_executable_deleter)(PJRT_SerializedExecutable* executable) noexcept nogil

    ctypedef struct PJRT_Executable_DeserializeAndLoad_Args:
        size_t struct_size
        PJRT_Client* client
        const char* serialized_executable
        size_t serialized_executable_size
        PJRT_LoadedExecutable* loaded_executable

    ctypedef struct PJRT_Api_Version:
        int major_version
        int minor_version

    ctypedef struct PJRT_Api:
        size_t struct_size
        PJRT_Api_Version pjrt_api_version
        void (*PJRT_Error_Destroy)(PJRT_Error_Destroy_Args* args) noexcept nogil
        void (*PJRT_Error_Message)(PJRT_Error_Message_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Plugin_Initialize)(PJRT_Plugin_Initialize_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Event_Destroy)(PJRT_Event_Destroy_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Event_Await)(PJRT_Event_Await_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Client_Create)(PJRT_Client_Create_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Client_PlatformName)(PJRT_Client_PlatformName_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Client_AddressableDevices)(PJRT_Client_AddressableDevices_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Client_Compile)(PJRT_Client_Compile_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Client_BufferFromHostBuffer)(PJRT_Client_BufferFromHostBuffer_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Executable_Destroy)(PJRT_Executable_Destroy_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Executable_NumOutputs)(PJRT_Executable_NumOutputs_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Executable_Serialize)(PJRT_Executable_Serialize_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_LoadedExecutable_Destroy)(PJRT_LoadedExecutable_Destroy_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_LoadedExecutable_GetExecutable)(PJRT_LoadedExecutable_GetExecutable_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_LoadedExecutable_Execute)(PJRT_LoadedExecutable_Execute_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Executable_DeserializeAndLoad)(PJRT_Executable_DeserializeAndLoad_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Buffer_Destroy)(PJRT_Buffer_Destroy_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Buffer_ToHostBuffer)(PJRT_Buffer_ToHostBuffer_Args* args) noexcept nogil
        PJRT_Error* (*PJRT_Buffer_UnsafePointer)(PJRT_Buffer_UnsafePointer_Args* args) noexcept nogil
