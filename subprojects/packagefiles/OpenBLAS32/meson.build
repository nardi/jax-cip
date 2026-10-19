project('OpenBLAS32-jll', 'c')


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
libgfortran_version = get_option('libgfortran_version')


# Which per-platform subproject matches this machine. Each branch condition
# was fully precomputed in Rust, so this is a plain if-elif chain with one
# entry per platform the JLL publishes a tarball for.
triplet = ''

if host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'i686-linux-gnu-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'x86_64-linux-gnu-libgfortran5.0.0'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'aarch64-linux-gnu-libgfortran5.0.0'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'armv6l-linux-gnueabihf-libgfortran5.0.0'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'armv7l-linux-gnueabihf-libgfortran5.0.0'

elif host_machine.cpu_family() == 'ppc64' and host_machine.system() == 'linux' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'powerpc64le-linux-gnu-libgfortran5.0.0'

elif host_machine.cpu_family() == 'riscv64' and host_machine.system() == 'linux' and libc == 'gnu' and libgfortran_version == '5.0.0'

  triplet = 'riscv64-linux-gnu-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'linux' and libc == 'musl' and libgfortran_version == '5.0.0'

  triplet = 'i686-linux-musl-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'linux' and libc == 'musl' and libgfortran_version == '5.0.0'

  triplet = 'x86_64-linux-musl-libgfortran5.0.0'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'linux' and libc == 'musl' and libgfortran_version == '5.0.0'

  triplet = 'aarch64-linux-musl-libgfortran5.0.0'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv6l' and libc == 'musl' and libgfortran_version == '5.0.0'

  triplet = 'armv6l-linux-musleabihf-libgfortran5.0.0'

elif host_machine.cpu_family() == 'arm' and host_machine.system() == 'linux' and host_machine.cpu() == 'armv7l' and libc == 'musl' and libgfortran_version == '5.0.0'

  triplet = 'armv7l-linux-musleabihf-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'darwin' and libgfortran_version == '5.0.0'

  triplet = 'x86_64-darwin-libgfortran5.0.0'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'darwin' and libgfortran_version == '5.0.0'

  triplet = 'aarch64-darwin-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'freebsd' and libgfortran_version == '5.0.0'

  triplet = 'x86_64-freebsd-libgfortran5.0.0'

elif host_machine.cpu_family() == 'aarch64' and host_machine.system() == 'freebsd' and libgfortran_version == '5.0.0'

  triplet = 'aarch64-freebsd-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86' and host_machine.system() == 'windows' and libgfortran_version == '5.0.0'

  triplet = 'i686-windows-libgfortran5.0.0'

elif host_machine.cpu_family() == 'x86_64' and host_machine.system() == 'windows' and libgfortran_version == '5.0.0'

  triplet = 'x86_64-windows-libgfortran5.0.0'

else
  error('meson-jll: unsupported platform for OpenBLAS32')
endif

# Meson only fetches a subproject once something references it, so only
# the tarball for the matching triplet is ever downloaded.
sub = subproject('OpenBLAS32-' + triplet)
openblas32_dep = sub.get_variable('openblas32_dep')
meson.override_dependency('OpenBLAS32', openblas32_dep)