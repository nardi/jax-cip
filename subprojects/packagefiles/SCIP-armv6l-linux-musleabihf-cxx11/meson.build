project('SCIP-armv6l-linux-musleabihf-cxx11', 'c')

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
  if fs.is_dir('include/scip')
    include_dirs += [include_directories('include/scip')]
  endif
endif



# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


libscip = cc.find_library('scip', dirs: meson.current_source_dir() / fs.parent('lib/libscip.so'))



libsoplex = cc.find_library('soplexshared', dirs: meson.current_source_dir() / fs.parent('lib/libsoplexshared.so'))



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.

Ipopt_dep = dependency('Ipopt')

GMP_dep = dependency('GMP')

boost_dep = dependency('boost')

CompilerSupportLibraries_dep = dependency('CompilerSupportLibraries')

Readline_dep = dependency('Readline')

Zlib_dep = dependency('Zlib')

Bzip2_dep = dependency('Bzip2')

MPFR_dep = dependency('MPFR')


# Install this platform's own runtime libraries, the same as
# `library('z', ..., install: true)` in an ordinary Meson project. They go
# to libdir, which is where meson-python folds a wheel's bundled shared
# libraries from (into `.<project>.mesonpy.libs`), and where an rpath entry
# points on the platforms that have one. On Windows the whole `bin`
# directory is installed rather than only the declared products, because
# the MinGW runtime DLLs those depend on (`libwinpthread-1.dll` and friends,
# provided by CompilerSupportLibraries) ship there without ever being
# declared as products of their own, and would otherwise be left out.


install_data('lib/libscip.so', install_dir: get_option('libdir'))

install_data('lib/libsoplexshared.so', install_dir: get_option('libdir'))



scip_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libscip,

    libsoplex,


    Ipopt_dep,

    GMP_dep,

    boost_dep,

    CompilerSupportLibraries_dep,

    Readline_dep,

    Zlib_dep,

    Bzip2_dep,

    MPFR_dep,

  ],
)