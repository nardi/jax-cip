project('SPRAL-x86_64-windows-libgfortran5.0.0', 'c')

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
  if fs.is_dir('include/spral')
    include_dirs += [include_directories('include/spral')]
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


if is_msvc and python3.found() and fs.is_file('bin/libspral.dll') and not fs.is_file('lib/spral.lib')
  libspral_import_lib = custom_target(
    'spral-msvc-import-lib',
    input: 'bin/libspral.dll',
    output: 'spral.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libspral = declare_dependency(link_with: libspral_import_lib)
else
  libspral = cc.find_library('spral', dirs: meson.current_source_dir() / fs.parent('bin/libspral.dll'))
endif



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.

Hwloc_dep = dependency('Hwloc')

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


spral_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libspral,


    Hwloc_dep,

    METIS_dep,

    CompilerSupportLibraries_dep,

    libblastrampoline_dep,

  ],
)