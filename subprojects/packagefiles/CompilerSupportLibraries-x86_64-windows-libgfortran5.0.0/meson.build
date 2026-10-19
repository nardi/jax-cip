project('CompilerSupportLibraries-x86_64-windows-libgfortran5.0.0', 'c')

cc = meson.get_compiler('c')
# The tarball is never downloaded ahead of time, so whether it ships headers
# is unknown until Meson has actually extracted it. Checking here, after
# extraction, avoids guessing in the generator.
fs = import('fs')
include_dirs = []
if fs.is_dir('include')
  include_dirs = [include_directories('include')]
  # Some JLLs (SuiteSparse and HiGHS both do) namespace their headers under
  # include/<lowercased-package-name>/ instead of flat under include/, to
  # avoid collisions between generically named headers from different JLLs.
  # Nothing in the JLL's own metadata says which convention it follows, so
  # both are added when present, covering `#include <SuiteSparse_config.h>`
  # and `#include <suitesparse/SuiteSparse_config.h>` alike.
  if fs.is_dir('include/compilersupportlibraries')
    include_dirs += [include_directories('include/compilersupportlibraries')]
  endif
endif


# Julia's Windows binaries are built with MinGW-w64 GCC, whose .dll.a
# import libraries are a GNU ar archive format MSVC's linker cannot read
# at all. When the active compiler turns out to be MSVC and a library has
# no native .lib of its own, dll_to_lib.py regenerates one straight from
# that DLL's own export table using dumpbin and lib.exe, both already on
# PATH as part of the same MSVC installation Meson activated to find `cc`
# in the first place. This runs at Meson build time, once per library that
# actually needs it, never while meson-jll itself is generating this file.
#
# The script lives once in packagefiles/, a directory up from every
# triplet overlay, rather than a copy inside each one: every Windows
# platform of every JLL shares this exact same file.
is_msvc = cc.get_id() == 'msvc'
python3 = find_program('python3', 'python', required: false)
dll_to_lib = meson.current_source_dir() / '..' / 'packagefiles' / 'dll_to_lib.py'


# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


if is_msvc and python3.found() and fs.is_file('bin/libatomic-1.dll') and not fs.is_file('lib/atomic-1.lib')
  libatomic_import_lib = custom_target(
    'atomic-1-msvc-import-lib',
    input: 'bin/libatomic-1.dll',
    output: 'atomic-1.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libatomic = declare_dependency(link_with: libatomic_import_lib)
else
  libatomic = cc.find_library('atomic-1', dirs: meson.current_source_dir() / fs.parent('bin/libatomic-1.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libgcc_s_seh-1.dll') and not fs.is_file('lib/gcc_s_seh-1.lib')
  libgcc_s_import_lib = custom_target(
    'gcc_s_seh-1-msvc-import-lib',
    input: 'bin/libgcc_s_seh-1.dll',
    output: 'gcc_s_seh-1.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libgcc_s = declare_dependency(link_with: libgcc_s_import_lib)
else
  libgcc_s = cc.find_library('gcc_s_seh-1', dirs: meson.current_source_dir() / fs.parent('bin/libgcc_s_seh-1.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libgfortran-5.dll') and not fs.is_file('lib/gfortran-5.lib')
  libgfortran_import_lib = custom_target(
    'gfortran-5-msvc-import-lib',
    input: 'bin/libgfortran-5.dll',
    output: 'gfortran-5.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libgfortran = declare_dependency(link_with: libgfortran_import_lib)
else
  libgfortran = cc.find_library('gfortran-5', dirs: meson.current_source_dir() / fs.parent('bin/libgfortran-5.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libgomp-1.dll') and not fs.is_file('lib/gomp-1.lib')
  libgomp_import_lib = custom_target(
    'gomp-1-msvc-import-lib',
    input: 'bin/libgomp-1.dll',
    output: 'gomp-1.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libgomp = declare_dependency(link_with: libgomp_import_lib)
else
  libgomp = cc.find_library('gomp-1', dirs: meson.current_source_dir() / fs.parent('bin/libgomp-1.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libssp-0.dll') and not fs.is_file('lib/ssp-0.lib')
  libssp_import_lib = custom_target(
    'ssp-0-msvc-import-lib',
    input: 'bin/libssp-0.dll',
    output: 'ssp-0.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libssp = declare_dependency(link_with: libssp_import_lib)
else
  libssp = cc.find_library('ssp-0', dirs: meson.current_source_dir() / fs.parent('bin/libssp-0.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libstdc++-6.dll') and not fs.is_file('lib/stdc++-6.lib')
  libstdcxx_import_lib = custom_target(
    'stdc++-6-msvc-import-lib',
    input: 'bin/libstdc++-6.dll',
    output: 'stdc++-6.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libstdcxx = declare_dependency(link_with: libstdcxx_import_lib)
else
  libstdcxx = cc.find_library('stdc++-6', dirs: meson.current_source_dir() / fs.parent('bin/libstdc++-6.dll'))
endif



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.


# Install this platform's own runtime libraries, the same as
# `library('z', ..., install: true)` in an ordinary Meson project. They go
# to libdir, which is where meson-python folds a wheel's bundled shared
# libraries from (into `.<project>.mesonpy.libs`), and where an rpath entry
# points on the platforms that have one. On Windows the whole `bin`
# directory is installed rather than only the declared products, because
# the MinGW runtime DLLs those depend on (`libwinpthread-1.dll` and friends,
# provided by CompilerSupportLibraries) ship there without ever being
# declared as products of their own, and would otherwise be left out.

install_subdir('bin', install_dir: get_option('libdir'), strip_directory: true)


compilersupportlibraries_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libatomic,

    libgcc_s,

    libgfortran,

    libgomp,

    libssp,

    libstdcxx,


  ],
)