project('Ncurses-aarch64-linux-musl', 'c')

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
  if fs.is_dir('include/ncurses')
    include_dirs += [include_directories('include/ncurses')]
  endif
endif



# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


libform = cc.find_library('form', dirs: meson.current_source_dir() / fs.parent('lib/libform.so'))



libmenu = cc.find_library('menu', dirs: meson.current_source_dir() / fs.parent('lib/libmenu.so'))



libncurses = cc.find_library('ncurses', dirs: meson.current_source_dir() / fs.parent('lib/libncurses.so'))



libncursesw = cc.find_library('ncursesw', dirs: meson.current_source_dir() / fs.parent('lib/libncursesw.so'))



libpanel = cc.find_library('panel', dirs: meson.current_source_dir() / fs.parent('lib/libpanel.so'))



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


install_data('lib/libform.so', install_dir: get_option('libdir'))

install_data('lib/libmenu.so', install_dir: get_option('libdir'))

install_data('lib/libncurses.so', install_dir: get_option('libdir'))

install_data('lib/libncursesw.so', install_dir: get_option('libdir'))

install_data('lib/libpanel.so', install_dir: get_option('libdir'))



ncurses_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libform,

    libmenu,

    libncurses,

    libncursesw,

    libpanel,


  ],
)