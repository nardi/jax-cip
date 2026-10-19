project(
  'jaxcip',
  'c',
  version: '0.1.0',
  meson_version: '>=1.8',
  default_options: ['c_std=c11', 'buildtype=release', 'b_ndebug=if-release'],
)

python = import('python').find_installation(pure: false)
cython = find_program('cython')

# SCIP and all of its dependencies come from the SCIP JLL through the wraps in
# `subprojects`, which `meson-jll` generates from `meson-jll.lock`.
scip = dependency('SCIP')

# Ipopt calls BLAS and LAPACK through libblastrampoline, which has no backend until one
# is forwarded to it. OpenBLAS is linked in so the extension can forward it on import.
openblas = dependency('OpenBLAS32')
lbt = dependency('libblastrampoline')

# The PJRT C API is a single header, vendored from XLA. The plugin itself is opened
# with `dlopen` at run time.
pjrt = declare_dependency(include_directories: include_directories('third_party/xla'))
dl = meson.get_compiler('c').find_library('dl')

subdir('src/jaxcip')
