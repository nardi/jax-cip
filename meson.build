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

# The PJRT C API header and the declarations of the cyxla runtime come with the installed
# cyxla. The first directory holds the header, the second the `.pxd` files.
cyxla_include = run_command(
  python, '-c', 'import cyxla; print("\\n".join(cyxla.get_include()))', check: true,
).stdout().strip().split('\n')
# A virtual environment inside the source tree puts cyxla there too, which
# `include_directories` refuses, so the directory goes in as a compiler argument.
cyxla = declare_dependency(compile_args: ['-I' + cyxla_include[0]])

subdir('src/jaxcip')
