project('XML2-jll', 'c')


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




# Which per-platform subproject matches this machine. Each branch condition
# was fully precomputed in Rust, so this is a plain if-elif chain with one
# entry per platform the JLL publishes a tarball for.
triplet = ''

if host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'darwin'

  triplet = 'aarch64-darwin'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'gnu'

  triplet = 'aarch64-linux-gnu'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'musl'

  triplet = 'aarch64-linux-musl'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'freebsd'

  triplet = 'aarch64-freebsd'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'gnu'

  triplet = 'armv6l-linux-gnueabihf'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'musl'

  triplet = 'armv6l-linux-musleabihf'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'gnu'

  triplet = 'armv7l-linux-gnueabihf'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'musl'

  triplet = 'armv7l-linux-musleabihf'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'gnu'

  triplet = 'i686-linux-gnu'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'musl'

  triplet = 'i686-linux-musl'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'windows'

  triplet = 'i686-windows'

elif host_machine.cpu_family() == 'ppc64' and host_machine.system() == 'linux' and libc == 'gnu'

  triplet = 'powerpc64le-linux-gnu'

elif host_machine.cpu_family() == 'riscv64' and host_machine.system() == 'linux' and libc == 'gnu'

  triplet = 'riscv64-linux-gnu'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'darwin'

  triplet = 'x86_64-darwin'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'gnu'

  triplet = 'x86_64-linux-gnu'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'musl'

  triplet = 'x86_64-linux-musl'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'freebsd'

  triplet = 'x86_64-freebsd'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'windows'

  triplet = 'x86_64-windows'

else
  error('meson-jll: unsupported platform for XML2')
endif

# Meson only fetches a subproject once something references it, so only
# the tarball for the matching triplet is ever downloaded.
sub = subproject('XML2-' + triplet)
xml2_dep = sub.get_variable('xml2_dep')
meson.override_dependency('XML2', xml2_dep)