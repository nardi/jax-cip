project('boost-armv7l-linux-musleabihf-cxx03', 'c')

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
  if fs.is_dir('include/boost')
    include_dirs += [include_directories('include/boost')]
  endif
endif



# One cc.find_library() per library this platform's wrapper script
# declares, using the exact path Julia itself records for it.


libboost_atomic = cc.find_library('boost_atomic', dirs: meson.current_source_dir() / fs.parent('lib/libboost_atomic.so'))



libboost_chrono = cc.find_library('boost_chrono', dirs: meson.current_source_dir() / fs.parent('lib/libboost_chrono.so'))



libboost_container = cc.find_library('boost_container', dirs: meson.current_source_dir() / fs.parent('lib/libboost_container.so'))



libboost_context = cc.find_library('boost_context', dirs: meson.current_source_dir() / fs.parent('lib/libboost_context.so'))



libboost_contract = cc.find_library('boost_contract', dirs: meson.current_source_dir() / fs.parent('lib/libboost_contract.so'))



libboost_coroutine = cc.find_library('boost_coroutine', dirs: meson.current_source_dir() / fs.parent('lib/libboost_coroutine.so'))



libboost_date_time = cc.find_library('boost_date_time', dirs: meson.current_source_dir() / fs.parent('lib/libboost_date_time.so'))



libboost_filesystem = cc.find_library('boost_filesystem', dirs: meson.current_source_dir() / fs.parent('lib/libboost_filesystem.so'))



libboost_graph = cc.find_library('boost_graph', dirs: meson.current_source_dir() / fs.parent('lib/libboost_graph.so'))



libboost_iostreams = cc.find_library('boost_iostreams', dirs: meson.current_source_dir() / fs.parent('lib/libboost_iostreams.so'))



libboost_log = cc.find_library('boost_log', dirs: meson.current_source_dir() / fs.parent('lib/libboost_log.so'))



libboost_log_setup = cc.find_library('boost_log_setup', dirs: meson.current_source_dir() / fs.parent('lib/libboost_log_setup.so'))



libboost_prg_exec_monitor = cc.find_library('boost_prg_exec_monitor', dirs: meson.current_source_dir() / fs.parent('lib/libboost_prg_exec_monitor.so'))



libboost_program_options = cc.find_library('boost_program_options', dirs: meson.current_source_dir() / fs.parent('lib/libboost_program_options.so'))



libboost_random = cc.find_library('boost_random', dirs: meson.current_source_dir() / fs.parent('lib/libboost_random.so'))



libboost_regex = cc.find_library('boost_regex', dirs: meson.current_source_dir() / fs.parent('lib/libboost_regex.so'))



libboost_serialization = cc.find_library('boost_serialization', dirs: meson.current_source_dir() / fs.parent('lib/libboost_serialization.so'))



libboost_stacktrace_basic = cc.find_library('boost_stacktrace_basic', dirs: meson.current_source_dir() / fs.parent('lib/libboost_stacktrace_basic.so'))



libboost_stacktrace_noop = cc.find_library('boost_stacktrace_noop', dirs: meson.current_source_dir() / fs.parent('lib/libboost_stacktrace_noop.so'))



libboost_system = cc.find_library('boost_system', dirs: meson.current_source_dir() / fs.parent('lib/libboost_system.so'))



libboost_thread = cc.find_library('boost_thread', dirs: meson.current_source_dir() / fs.parent('lib/libboost_thread.so'))



libboost_timer = cc.find_library('boost_timer', dirs: meson.current_source_dir() / fs.parent('lib/libboost_timer.so'))



libboost_type_erasure = cc.find_library('boost_type_erasure', dirs: meson.current_source_dir() / fs.parent('lib/libboost_type_erasure.so'))



libboost_unit_test_framework = cc.find_library('boost_unit_test_framework', dirs: meson.current_source_dir() / fs.parent('lib/libboost_unit_test_framework.so'))



libboost_wave = cc.find_library('boost_wave', dirs: meson.current_source_dir() / fs.parent('lib/libboost_wave.so'))



libboost_wserialization = cc.find_library('boost_wserialization', dirs: meson.current_source_dir() / fs.parent('lib/libboost_wserialization.so'))



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.

Bzip2_dep = dependency('Bzip2')

Zlib_dep = dependency('Zlib')


# Install this platform's own runtime libraries, the same as
# `library('z', ..., install: true)` in an ordinary Meson project. They go
# to libdir, which is where meson-python folds a wheel's bundled shared
# libraries from (into `.<project>.mesonpy.libs`), and where an rpath entry
# points on the platforms that have one. On Windows the whole `bin`
# directory is installed rather than only the declared products, because
# the MinGW runtime DLLs those depend on (`libwinpthread-1.dll` and friends,
# provided by CompilerSupportLibraries) ship there without ever being
# declared as products of their own, and would otherwise be left out.


install_data('lib/libboost_atomic.so', install_dir: get_option('libdir'))

install_data('lib/libboost_chrono.so', install_dir: get_option('libdir'))

install_data('lib/libboost_container.so', install_dir: get_option('libdir'))

install_data('lib/libboost_context.so', install_dir: get_option('libdir'))

install_data('lib/libboost_contract.so', install_dir: get_option('libdir'))

install_data('lib/libboost_coroutine.so', install_dir: get_option('libdir'))

install_data('lib/libboost_date_time.so', install_dir: get_option('libdir'))

install_data('lib/libboost_filesystem.so', install_dir: get_option('libdir'))

install_data('lib/libboost_graph.so', install_dir: get_option('libdir'))

install_data('lib/libboost_iostreams.so', install_dir: get_option('libdir'))

install_data('lib/libboost_log.so', install_dir: get_option('libdir'))

install_data('lib/libboost_log_setup.so', install_dir: get_option('libdir'))

install_data('lib/libboost_prg_exec_monitor.so', install_dir: get_option('libdir'))

install_data('lib/libboost_program_options.so', install_dir: get_option('libdir'))

install_data('lib/libboost_random.so', install_dir: get_option('libdir'))

install_data('lib/libboost_regex.so', install_dir: get_option('libdir'))

install_data('lib/libboost_serialization.so', install_dir: get_option('libdir'))

install_data('lib/libboost_stacktrace_basic.so', install_dir: get_option('libdir'))

install_data('lib/libboost_stacktrace_noop.so', install_dir: get_option('libdir'))

install_data('lib/libboost_system.so', install_dir: get_option('libdir'))

install_data('lib/libboost_thread.so', install_dir: get_option('libdir'))

install_data('lib/libboost_timer.so', install_dir: get_option('libdir'))

install_data('lib/libboost_type_erasure.so', install_dir: get_option('libdir'))

install_data('lib/libboost_unit_test_framework.so', install_dir: get_option('libdir'))

install_data('lib/libboost_wave.so', install_dir: get_option('libdir'))

install_data('lib/libboost_wserialization.so', install_dir: get_option('libdir'))



boost_dep = declare_dependency(
  include_directories: include_dirs,
  dependencies: [

    libboost_atomic,

    libboost_chrono,

    libboost_container,

    libboost_context,

    libboost_contract,

    libboost_coroutine,

    libboost_date_time,

    libboost_filesystem,

    libboost_graph,

    libboost_iostreams,

    libboost_log,

    libboost_log_setup,

    libboost_prg_exec_monitor,

    libboost_program_options,

    libboost_random,

    libboost_regex,

    libboost_serialization,

    libboost_stacktrace_basic,

    libboost_stacktrace_noop,

    libboost_system,

    libboost_thread,

    libboost_timer,

    libboost_type_erasure,

    libboost_unit_test_framework,

    libboost_wave,

    libboost_wserialization,


    Bzip2_dep,

    Zlib_dep,

  ],
)