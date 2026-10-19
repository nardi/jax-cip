# Vendored PJRT C API header

`xla/pjrt/c/pjrt_c_api.h` is copied unchanged from [openxla/xla](https://github.com/openxla/xla) at commit `21d3caa7e06a33565c22927a7be5c5e68b4b94a5` (17 October 2026), which declares PJRT C API version 0.116. The header has no dependencies beyond the C standard library. jaxlib 0.10 does not ship it, since `jax.ffi.include_dir()` only contains the XLA FFI headers.

Plugins accept argument structs that are larger than the version they were built for, so this header works with plugins from version 0.114 on, including the CPU plugin of [zml/pjrt-artifacts](https://github.com/zml/pjrt-artifacts) v18.0.0. `LICENSE` is XLA's Apache 2.0 license, which covers the header.
