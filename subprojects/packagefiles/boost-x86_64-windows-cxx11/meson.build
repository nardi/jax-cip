project('boost-x86_64-windows-cxx11', 'c')

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


if is_msvc and python3.found() and fs.is_file('bin/libboost_atomic.dll') and not fs.is_file('lib/boost_atomic.lib')
  libboost_atomic_import_lib = custom_target(
    'boost_atomic-msvc-import-lib',
    input: 'bin/libboost_atomic.dll',
    output: 'boost_atomic.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_atomic = declare_dependency(link_with: libboost_atomic_import_lib)
else
  libboost_atomic = cc.find_library('boost_atomic', dirs: meson.current_source_dir() / fs.parent('bin/libboost_atomic.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_chrono.dll') and not fs.is_file('lib/boost_chrono.lib')
  libboost_chrono_import_lib = custom_target(
    'boost_chrono-msvc-import-lib',
    input: 'bin/libboost_chrono.dll',
    output: 'boost_chrono.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_chrono = declare_dependency(link_with: libboost_chrono_import_lib)
else
  libboost_chrono = cc.find_library('boost_chrono', dirs: meson.current_source_dir() / fs.parent('bin/libboost_chrono.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_container.dll') and not fs.is_file('lib/boost_container.lib')
  libboost_container_import_lib = custom_target(
    'boost_container-msvc-import-lib',
    input: 'bin/libboost_container.dll',
    output: 'boost_container.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_container = declare_dependency(link_with: libboost_container_import_lib)
else
  libboost_container = cc.find_library('boost_container', dirs: meson.current_source_dir() / fs.parent('bin/libboost_container.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_context.dll') and not fs.is_file('lib/boost_context.lib')
  libboost_context_import_lib = custom_target(
    'boost_context-msvc-import-lib',
    input: 'bin/libboost_context.dll',
    output: 'boost_context.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_context = declare_dependency(link_with: libboost_context_import_lib)
else
  libboost_context = cc.find_library('boost_context', dirs: meson.current_source_dir() / fs.parent('bin/libboost_context.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_contract.dll') and not fs.is_file('lib/boost_contract.lib')
  libboost_contract_import_lib = custom_target(
    'boost_contract-msvc-import-lib',
    input: 'bin/libboost_contract.dll',
    output: 'boost_contract.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_contract = declare_dependency(link_with: libboost_contract_import_lib)
else
  libboost_contract = cc.find_library('boost_contract', dirs: meson.current_source_dir() / fs.parent('bin/libboost_contract.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_coroutine.dll') and not fs.is_file('lib/boost_coroutine.lib')
  libboost_coroutine_import_lib = custom_target(
    'boost_coroutine-msvc-import-lib',
    input: 'bin/libboost_coroutine.dll',
    output: 'boost_coroutine.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_coroutine = declare_dependency(link_with: libboost_coroutine_import_lib)
else
  libboost_coroutine = cc.find_library('boost_coroutine', dirs: meson.current_source_dir() / fs.parent('bin/libboost_coroutine.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_date_time.dll') and not fs.is_file('lib/boost_date_time.lib')
  libboost_date_time_import_lib = custom_target(
    'boost_date_time-msvc-import-lib',
    input: 'bin/libboost_date_time.dll',
    output: 'boost_date_time.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_date_time = declare_dependency(link_with: libboost_date_time_import_lib)
else
  libboost_date_time = cc.find_library('boost_date_time', dirs: meson.current_source_dir() / fs.parent('bin/libboost_date_time.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_filesystem.dll') and not fs.is_file('lib/boost_filesystem.lib')
  libboost_filesystem_import_lib = custom_target(
    'boost_filesystem-msvc-import-lib',
    input: 'bin/libboost_filesystem.dll',
    output: 'boost_filesystem.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_filesystem = declare_dependency(link_with: libboost_filesystem_import_lib)
else
  libboost_filesystem = cc.find_library('boost_filesystem', dirs: meson.current_source_dir() / fs.parent('bin/libboost_filesystem.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_graph.dll') and not fs.is_file('lib/boost_graph.lib')
  libboost_graph_import_lib = custom_target(
    'boost_graph-msvc-import-lib',
    input: 'bin/libboost_graph.dll',
    output: 'boost_graph.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_graph = declare_dependency(link_with: libboost_graph_import_lib)
else
  libboost_graph = cc.find_library('boost_graph', dirs: meson.current_source_dir() / fs.parent('bin/libboost_graph.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_iostreams.dll') and not fs.is_file('lib/boost_iostreams.lib')
  libboost_iostreams_import_lib = custom_target(
    'boost_iostreams-msvc-import-lib',
    input: 'bin/libboost_iostreams.dll',
    output: 'boost_iostreams.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_iostreams = declare_dependency(link_with: libboost_iostreams_import_lib)
else
  libboost_iostreams = cc.find_library('boost_iostreams', dirs: meson.current_source_dir() / fs.parent('bin/libboost_iostreams.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_log.dll') and not fs.is_file('lib/boost_log.lib')
  libboost_log_import_lib = custom_target(
    'boost_log-msvc-import-lib',
    input: 'bin/libboost_log.dll',
    output: 'boost_log.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_log = declare_dependency(link_with: libboost_log_import_lib)
else
  libboost_log = cc.find_library('boost_log', dirs: meson.current_source_dir() / fs.parent('bin/libboost_log.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_log_setup.dll') and not fs.is_file('lib/boost_log_setup.lib')
  libboost_log_setup_import_lib = custom_target(
    'boost_log_setup-msvc-import-lib',
    input: 'bin/libboost_log_setup.dll',
    output: 'boost_log_setup.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_log_setup = declare_dependency(link_with: libboost_log_setup_import_lib)
else
  libboost_log_setup = cc.find_library('boost_log_setup', dirs: meson.current_source_dir() / fs.parent('bin/libboost_log_setup.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_prg_exec_monitor.dll') and not fs.is_file('lib/boost_prg_exec_monitor.lib')
  libboost_prg_exec_monitor_import_lib = custom_target(
    'boost_prg_exec_monitor-msvc-import-lib',
    input: 'bin/libboost_prg_exec_monitor.dll',
    output: 'boost_prg_exec_monitor.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_prg_exec_monitor = declare_dependency(link_with: libboost_prg_exec_monitor_import_lib)
else
  libboost_prg_exec_monitor = cc.find_library('boost_prg_exec_monitor', dirs: meson.current_source_dir() / fs.parent('bin/libboost_prg_exec_monitor.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_program_options.dll') and not fs.is_file('lib/boost_program_options.lib')
  libboost_program_options_import_lib = custom_target(
    'boost_program_options-msvc-import-lib',
    input: 'bin/libboost_program_options.dll',
    output: 'boost_program_options.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_program_options = declare_dependency(link_with: libboost_program_options_import_lib)
else
  libboost_program_options = cc.find_library('boost_program_options', dirs: meson.current_source_dir() / fs.parent('bin/libboost_program_options.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_random.dll') and not fs.is_file('lib/boost_random.lib')
  libboost_random_import_lib = custom_target(
    'boost_random-msvc-import-lib',
    input: 'bin/libboost_random.dll',
    output: 'boost_random.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_random = declare_dependency(link_with: libboost_random_import_lib)
else
  libboost_random = cc.find_library('boost_random', dirs: meson.current_source_dir() / fs.parent('bin/libboost_random.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_regex.dll') and not fs.is_file('lib/boost_regex.lib')
  libboost_regex_import_lib = custom_target(
    'boost_regex-msvc-import-lib',
    input: 'bin/libboost_regex.dll',
    output: 'boost_regex.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_regex = declare_dependency(link_with: libboost_regex_import_lib)
else
  libboost_regex = cc.find_library('boost_regex', dirs: meson.current_source_dir() / fs.parent('bin/libboost_regex.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_serialization.dll') and not fs.is_file('lib/boost_serialization.lib')
  libboost_serialization_import_lib = custom_target(
    'boost_serialization-msvc-import-lib',
    input: 'bin/libboost_serialization.dll',
    output: 'boost_serialization.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_serialization = declare_dependency(link_with: libboost_serialization_import_lib)
else
  libboost_serialization = cc.find_library('boost_serialization', dirs: meson.current_source_dir() / fs.parent('bin/libboost_serialization.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_stacktrace_basic.dll') and not fs.is_file('lib/boost_stacktrace_basic.lib')
  libboost_stacktrace_basic_import_lib = custom_target(
    'boost_stacktrace_basic-msvc-import-lib',
    input: 'bin/libboost_stacktrace_basic.dll',
    output: 'boost_stacktrace_basic.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_stacktrace_basic = declare_dependency(link_with: libboost_stacktrace_basic_import_lib)
else
  libboost_stacktrace_basic = cc.find_library('boost_stacktrace_basic', dirs: meson.current_source_dir() / fs.parent('bin/libboost_stacktrace_basic.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_stacktrace_noop.dll') and not fs.is_file('lib/boost_stacktrace_noop.lib')
  libboost_stacktrace_noop_import_lib = custom_target(
    'boost_stacktrace_noop-msvc-import-lib',
    input: 'bin/libboost_stacktrace_noop.dll',
    output: 'boost_stacktrace_noop.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_stacktrace_noop = declare_dependency(link_with: libboost_stacktrace_noop_import_lib)
else
  libboost_stacktrace_noop = cc.find_library('boost_stacktrace_noop', dirs: meson.current_source_dir() / fs.parent('bin/libboost_stacktrace_noop.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_system.dll') and not fs.is_file('lib/boost_system.lib')
  libboost_system_import_lib = custom_target(
    'boost_system-msvc-import-lib',
    input: 'bin/libboost_system.dll',
    output: 'boost_system.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_system = declare_dependency(link_with: libboost_system_import_lib)
else
  libboost_system = cc.find_library('boost_system', dirs: meson.current_source_dir() / fs.parent('bin/libboost_system.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_thread.dll') and not fs.is_file('lib/boost_thread.lib')
  libboost_thread_import_lib = custom_target(
    'boost_thread-msvc-import-lib',
    input: 'bin/libboost_thread.dll',
    output: 'boost_thread.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_thread = declare_dependency(link_with: libboost_thread_import_lib)
else
  libboost_thread = cc.find_library('boost_thread', dirs: meson.current_source_dir() / fs.parent('bin/libboost_thread.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_timer.dll') and not fs.is_file('lib/boost_timer.lib')
  libboost_timer_import_lib = custom_target(
    'boost_timer-msvc-import-lib',
    input: 'bin/libboost_timer.dll',
    output: 'boost_timer.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_timer = declare_dependency(link_with: libboost_timer_import_lib)
else
  libboost_timer = cc.find_library('boost_timer', dirs: meson.current_source_dir() / fs.parent('bin/libboost_timer.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_type_erasure.dll') and not fs.is_file('lib/boost_type_erasure.lib')
  libboost_type_erasure_import_lib = custom_target(
    'boost_type_erasure-msvc-import-lib',
    input: 'bin/libboost_type_erasure.dll',
    output: 'boost_type_erasure.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_type_erasure = declare_dependency(link_with: libboost_type_erasure_import_lib)
else
  libboost_type_erasure = cc.find_library('boost_type_erasure', dirs: meson.current_source_dir() / fs.parent('bin/libboost_type_erasure.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_unit_test_framework.dll') and not fs.is_file('lib/boost_unit_test_framework.lib')
  libboost_unit_test_framework_import_lib = custom_target(
    'boost_unit_test_framework-msvc-import-lib',
    input: 'bin/libboost_unit_test_framework.dll',
    output: 'boost_unit_test_framework.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_unit_test_framework = declare_dependency(link_with: libboost_unit_test_framework_import_lib)
else
  libboost_unit_test_framework = cc.find_library('boost_unit_test_framework', dirs: meson.current_source_dir() / fs.parent('bin/libboost_unit_test_framework.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_wave.dll') and not fs.is_file('lib/boost_wave.lib')
  libboost_wave_import_lib = custom_target(
    'boost_wave-msvc-import-lib',
    input: 'bin/libboost_wave.dll',
    output: 'boost_wave.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_wave = declare_dependency(link_with: libboost_wave_import_lib)
else
  libboost_wave = cc.find_library('boost_wave', dirs: meson.current_source_dir() / fs.parent('bin/libboost_wave.dll'))
endif



if is_msvc and python3.found() and fs.is_file('bin/libboost_wserialization.dll') and not fs.is_file('lib/boost_wserialization.lib')
  libboost_wserialization_import_lib = custom_target(
    'boost_wserialization-msvc-import-lib',
    input: 'bin/libboost_wserialization.dll',
    output: 'boost_wserialization.lib',
    command: [python3, dll_to_lib, '@INPUT@', '@OUTPUT@', 'X64'],
  )
  libboost_wserialization = declare_dependency(link_with: libboost_wserialization_import_lib)
else
  libboost_wserialization = cc.find_library('boost_wserialization', dirs: meson.current_source_dir() / fs.parent('bin/libboost_wserialization.dll'))
endif



# Transitive JLL dependencies, resolved the same way this package itself
# is: through their own selector wrap's [provide] fallback.

Bzip2_dep = dependency('Bzip2')

Zlib_dep = dependency('Zlib')

CompilerSupportLibraries_dep = dependency('CompilerSupportLibraries')


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

    CompilerSupportLibraries_dep,

  ],
)