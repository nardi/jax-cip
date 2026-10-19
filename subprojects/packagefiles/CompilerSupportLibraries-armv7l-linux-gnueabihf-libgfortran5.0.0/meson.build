project('CompilerSupportLibraries-armv7l-linux-gnueabihf-libgfortran5.0.0', 'c')

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



# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


libatomic = cc.find_library('atomic', dirs: meson.current_source_dir() / fs.parent('lib/libatomic.so'))



libgcc_s = cc.find_library('gcc_s', dirs: meson.current_source_dir() / fs.parent('lib/libgcc_s.so'))



libgfortran = cc.find_library('gfortran', dirs: meson.current_source_dir() / fs.parent('lib/libgfortran.so'))



libgomp = cc.find_library('gomp', dirs: meson.current_source_dir() / fs.parent('lib/libgomp.so'))



libssp = cc.find_library('ssp', dirs: meson.current_source_dir() / fs.parent('lib/libssp.so'))



libstdcxx = cc.find_library('stdc++', dirs: meson.current_source_dir() / fs.parent('lib/libstdc++.so'))



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


install_data('lib/libatomic.so', install_dir: get_option('libdir'))

install_data('lib/libgcc_s.so', install_dir: get_option('libdir'))

install_data('lib/libgfortran.so', install_dir: get_option('libdir'))

install_data('lib/libgomp.so', install_dir: get_option('libdir'))

install_data('lib/libssp.so', install_dir: get_option('libdir'))

install_data('lib/libstdc++.so', install_dir: get_option('libdir'))



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