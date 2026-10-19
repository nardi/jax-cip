project('boost-jll', 'c')


# host_machine has no libc() getter, so tell glibc and musl apart by
# compiling a snippet that only glibc defines the macro for.
cc = meson.get_compiler('c')
libc = 'unknown'
if host_machine.system() == 'linux'
  if cc.compiles('#include <features.h>\nint main(void) { return __GLIBC__; }', name: 'glibc probe')
    libc = 'gnu'
  else
    libc = 'musl'
  endif
endif


# Declared in the sibling meson.options this package generated.
cxxstring_abi = get_option('cxxstring_abi')



# Which per-platform subproject matches this machine. Each branch condition
# was fully precomputed in Rust, so this is a plain if-elif chain with one
# entry per platform the JLL publishes a tarball for.
triplet = ''

if host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'darwin'

  triplet = 'aarch64-darwin'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'aarch64-linux-gnu-cxx03'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'aarch64-linux-gnu-cxx11'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx03'

  triplet = 'aarch64-linux-musl-cxx03'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx11'

  triplet = 'aarch64-linux-musl-cxx11'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'freebsd'

  triplet = 'aarch64-freebsd'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'armv6l-linux-gnueabihf-cxx03'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'armv6l-linux-gnueabihf-cxx11'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'musl' and cxxstring_abi == 'cxx03'

  triplet = 'armv6l-linux-musleabihf-cxx03'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'musl' and cxxstring_abi == 'cxx11'

  triplet = 'armv6l-linux-musleabihf-cxx11'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'armv7l-linux-gnueabihf-cxx03'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'armv7l-linux-gnueabihf-cxx11'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'musl' and cxxstring_abi == 'cxx03'

  triplet = 'armv7l-linux-musleabihf-cxx03'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'musl' and cxxstring_abi == 'cxx11'

  triplet = 'armv7l-linux-musleabihf-cxx11'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'i686-linux-gnu-cxx03'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'i686-linux-gnu-cxx11'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx03'

  triplet = 'i686-linux-musl-cxx03'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx11'

  triplet = 'i686-linux-musl-cxx11'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'windows' and cxxstring_abi == 'cxx03'

  triplet = 'i686-windows-cxx03'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'windows' and cxxstring_abi == 'cxx11'

  triplet = 'i686-windows-cxx11'

elif host_machine.cpu_family() == 'ppc64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'powerpc64le-linux-gnu-cxx03'

elif host_machine.cpu_family() == 'ppc64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'powerpc64le-linux-gnu-cxx11'

elif host_machine.cpu_family() == 'riscv64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'riscv64-linux-gnu-cxx03'

elif host_machine.cpu_family() == 'riscv64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'riscv64-linux-gnu-cxx11'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'darwin'

  triplet = 'x86_64-darwin'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx03'

  triplet = 'x86_64-linux-gnu-cxx03'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'gnu' and cxxstring_abi == 'cxx11'

  triplet = 'x86_64-linux-gnu-cxx11'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx03'

  triplet = 'x86_64-linux-musl-cxx03'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'musl' and cxxstring_abi == 'cxx11'

  triplet = 'x86_64-linux-musl-cxx11'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'freebsd'

  triplet = 'x86_64-freebsd'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'windows' and cxxstring_abi == 'cxx03'

  triplet = 'x86_64-windows-cxx03'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'windows' and cxxstring_abi == 'cxx11'

  triplet = 'x86_64-windows-cxx11'

else
  error('meson-jll: unsupported platform for boost')
endif

# Meson only fetches a subproject once something references it, so only
# the tarball for the matching triplet is ever downloaded.
sub = subproject('boost-' + triplet)
boost_dep = sub.get_variable('boost_dep')
meson.override_dependency('boost', boost_dep)