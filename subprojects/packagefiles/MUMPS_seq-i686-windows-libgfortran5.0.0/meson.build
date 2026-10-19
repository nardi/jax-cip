project('MUMPS_seq-i686-windows-libgfortran5.0.0', 'c')

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
  if fs.is_dir('include/mumps_seq')
    include_dirs += [include_directories('include/mumps_seq')]
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


if is_msvc and python3.found() and fs.is_file('bin/libcmumps.dll') and not fs.is_file('lib/cmumps.lib')
  libcmumps_import_lib = custom_target(
    'cmumps-msvc-import-lib',
    input: 'bin/libcmumps.dll',
    output: 'cmumps.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X86'],
  )
  libcmumps = declare_dependency(link_with: libcmumps_import_lib)
else
  libcmumps = cc.find_library('cmumps', dirs: meson.current_source_dir() / fs.parent('bin/libcmumps.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libdmumps.dll') and not fs.is_file('lib/dmumps.lib')
  libdmumps_import_lib = custom_target(
    'dmumps-msvc-import-lib',
    input: 'bin/libdmumps.dll',
    output: 'dmumps.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X86'],
  )
  libdmumps = declare_dependency(link_with: libdmumps_import_lib)
else
  libdmumps = cc.find_library('dmumps', dirs: meson.current_source_dir() / fs.parent('bin/libdmumps.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libsmumps.dll') and not fs.is_file('lib/smumps.lib')
  libsmumps_import_lib = custom_target(
    'smumps-msvc-import-lib',
    input: 'bin/libsmumps.dll',
    output: 'smumps.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X86'],
  )
  libsmumps = declare_dependency(link_with: libsmumps_import_lib)
else
  libsmumps = cc.find_library('smumps', dirs: meson.current_source_dir() / fs.parent('bin/libsmumps.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libzmumps.dll') and not fs.is_file('lib/zmumps.lib')
  libzmumps_import_lib = custom_target(
    'zmumps-msvc-import-lib',
    input: 'bin/libzmumps.dll',
    output: 'zmumps.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X86'],
  )
  libzmumps = declare_dependency(link_with: libzmumps_import_lib)
else
  libzmumps = cc.find_library('zmumps', dirs: meson.current_source_dir() / fs.parent('bin/libzmumps.dll'))
endif



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.

METIS_dep = dependency('METIS')

CompilerSupportLibraries_dep = dependency('CompilerSupportLibraries')

libblastrampoline_dep = dependency('libblastrampoline')


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


mumps_seq_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libcmumps,

    libdmumps,

    libsmumps,

    libzmumps,


    METIS_dep,

    CompilerSupportLibraries_dep,

    libblastrampoline_dep,

  ],
)