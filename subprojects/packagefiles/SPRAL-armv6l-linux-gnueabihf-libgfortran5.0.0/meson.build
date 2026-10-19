project('SPRAL-armv6l-linux-gnueabihf-libgfortran5.0.0', 'c')

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



# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


libspral = cc.find_library('spral', dirs: meson.current_source_dir() / fs.parent('lib/libspral.so'))



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


install_data('lib/libspral.so', install_dir: get_option('libdir'))



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