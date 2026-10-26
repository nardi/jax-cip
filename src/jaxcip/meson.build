python.install_sources(
  '__init__.py',
  '_analysis.py',
  '_blocks.py',
  '_cache.py',
  '_compile.py',
  '_options.py',
  '_problem_types.py',
  '_programs.py',
  '_roots.py',
  '_rows.py',
  '_scip_parameters.py',
  '_solve.py',
  '_types.py',
  '_block.pyi',
  '_handler.pyi',
  '_problem.pyi',
  '_runtime.pyi',
  'py.typed',
  subdir: 'jaxcip',
)

cython_args = ['-3', '--fast-fail']
c_args = ['-Wno-unused-function', '-Wno-unused-variable', '-Wno-unused-but-set-variable']
if get_option('coverage')
  cython_args += ['-X', 'linetrace=True']
  # Coverage measures branches through `sys.settrace` before Python 3.14, so Cython has to
  # report lines through the same hooks rather than through `sys.monitoring`.
  c_args += ['-DCYTHON_TRACE_NOGIL=1', '-DCYTHON_USE_SYS_MONITORING=0']
endif

# The C declarations every module can `cimport`. They are listed as dependencies of each
# `cython` step, because its depfile would record paths relative to the project root
# rather than the build directory.
declaration_files = files('_blas.pxd', '_block.pxd', '_handler.pxd', '_pjrt.pxd', '_runtime.pxd', '_scip.pxd')

# Meson compiles Cython from `.pyx` sources only, so the pure Python mode modules are
# translated to C by an explicit `cython` step.
foreach module : ['_runtime', '_handler', '_block', '_problem']
  module_c = custom_target(
    module + '.c',
    input: module + '.py',
    output: module + '.c',
    depend_files: declaration_files,
    command: [
      cython, cython_args,
      # Running from the project root makes Cython record the source as
      # `src/jaxcip/<module>.py`, the path coverage resolves line events against.
      '-w', meson.project_source_root(),
      '-I', 'src',
      '--module-name', 'jaxcip.' + module,
      meson.current_source_dir() / module + '.py',
      '-o', meson.current_build_dir() / module + '.c',
    ],
  )
  if get_option('coverage')
    # The `Cython.Coverage` plugin looks for the generated C file next to the source,
    # so coverage builds place a copy there.
    custom_target(
      module + '.c.copy',
      input: module_c,
      output: module + '.c.stamp',
      command: [
        python, '-c', 'import shutil, sys; shutil.copyfile(sys.argv[1], sys.argv[2]); open(sys.argv[3], "w").close()',
        '@INPUT@', meson.current_source_dir() / module + '.c', '@OUTPUT@',
      ],
      build_by_default: true,
    )
  endif
  python.extension_module(
    module,
    module_c,
    dependencies: [scip, openblas, lbt, pjrt, dl, python.dependency()],
    c_args: c_args,
    # The JLL libraries find their own dependencies through an `$ORIGIN` run path,
    # which only covers their own directory. Linking every one of them directly keeps
    # them all on the extension's run path instead.
    link_args: ['-Wl,--no-as-needed'],
    install: true,
    subdir: 'jaxcip',
  )
endforeach
