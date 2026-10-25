# Installation

JAX-CIP builds from source with [Meson](https://mesonbuild.com/) and needs a PJRT plugin
at run time. CI tests it on Linux x86-64 with Python 3.12 and 3.13.

## SCIP

The extension modules link SCIP 10.1 with SoPlex and Ipopt from the
[SCIP_jll](https://github.com/JuliaBinaryWrappers/SCIP_jll.jl) binaries that Julia
publishes. [meson-jll](https://github.com/nardi/meson-jll) turns those packages into
Meson wraps, and the repository commits the wraps for every platform under
`subprojects`, pinned by `meson-jll.lock`. Meson downloads the tarballs it needs on the
first build, so you don't need SCIP installed on the system.

## PJRT plugin

JAX-CIP runs its programs through the [PJRT C
API](https://openxla.org/xla/pjrt/pjrt_integration), outside the Python interpreter, so
that SCIP's callbacks never take the GIL. jaxlib links its CPU client statically and
ships no plugin library, so the CPU plugin comes from a prebuilt release.
`scripts/fetch_pjrt_plugin.sh` downloads it and checks its hash:

```bash
scripts/fetch_pjrt_plugin.sh ~/.cache/jaxcip/pjrt
export JAXCIP_PJRT_PLUGIN=~/.cache/jaxcip/pjrt/libpjrt_cpu.so
```

[`initialize`][jaxcip.initialize] loads a plugin from a path instead of the environment
variable. Call it before compiling the first problem.

## The package

```bash
git clone https://github.com/nardi/jax-cip
cd jax-cip
uv sync
```

`uv sync` installs the package in editable mode with the development tools, and Meson
will rebuild the extension modules when you import them after a change. These
development builds trace lines in the Cython modules for coverage, which makes the
callbacks several times slower. For timing, reinstall the package without tracing. The
`--no-config` flag stops uv from adding the coverage setting in `pyproject.toml`, and
`uv run --no-sync` stops it from rebuilding the package with that setting:

```bash
uv pip install --no-config --no-deps --no-build-isolation --reinstall -e . \
    -C setup-args=-Dcoverage=false
uv run --no-sync pytest benchmarks -o addopts="" --benchmark-only
```

## Environment variables

| Variable | Meaning |
| --- | --- |
| `JAXCIP_PJRT_PLUGIN` | Path of the PJRT plugin loaded by default |
| `JAXCIP_CACHE_DIR` | Directory of the disk cache, `~/.cache/jaxcip` if unset |
| `JAXCIP_MINLPLIB_DIR` | Directory of the MINLPLib instances for the test suite |
| `JAXCIP_MINLPLIB_RESULTS` | File that the MINLPLib suite appends its results to |
