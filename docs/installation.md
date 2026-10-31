# Installation

JAX-CIP builds from source with [Meson](https://mesonbuild.com/) and runs its programs
through [cyxla](https://nardi.github.io/cyxla/). CI tests it on Linux x86-64 with Python
3.11, 3.12 and 3.13.

## Python and JAX versions

JAX-CIP works with JAX 0.10 and 0.11. Release 0.5.2 of
[asdex](https://pypi.org/project/asdex/), which detects sparsity, needs Python 3.12 and
JAX 0.11, and does not import on JAX 0.12. Python 3.11 therefore installs JAX 0.10 and
release 0.5.1 of asdex, which has the same interface for everything JAX-CIP uses.

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
that SCIP's callbacks never take the GIL. cyxla compiles the programs and declares the
functions that run them. jaxlib links its CPU client statically and ships
no plugin library, so the CPU plugin is a separate wheel, `cyxla-pjrt-cpu`. JAX-CIP
depends on `cyxla[cpu]`, which installs both, so `uv sync` needs no further step.

`cyxla.initialize` loads another plugin, or the same one with other options. Call it
before compiling the first problem. The [cyxla
documentation](https://nardi.github.io/cyxla/) describes the plugins and their options.

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

## Tests

The tests have two groups. The unit tests solve small problems and cover all the code, and
`uv run pytest -m unit` runs only them. The integration tests compare configurations with
native SCIP, run longer solves and start new processes. The code blocks of the README and
of these pages count as integration tests as well, and `uv run pytest -m integration`
runs them. The MINLPLib instances need the `--minlplib` option.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `CYXLA_PJRT_PLUGIN` | Path of a PJRT plugin that replaces the installed ones |
| `CYXLA_CACHE_DIR` | Directory of the disk cache, `~/.cache/cyxla` if unset |
| `JAXCIP_MINLPLIB_DIR` | Directory of the MINLPLib instances for the test suite |
| `JAXCIP_MINLPLIB_RESULTS` | File that the MINLPLib suite appends its results to |
