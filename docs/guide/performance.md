# Performance

A solve spends time in three places: building the programs, running them in SCIP's
callbacks, and SCIP's own work on the tree. Every [`Solution`][jaxcip.Solution] reports
the first two.

## Build time

`solution.build` is a [`BuildStatistics`][jaxcip.BuildStatistics]. JAX-CIP lowers a term
that is new to the process to StableHLO, and XLA compiles it, which takes from a tenth
of a second to several seconds per program. Both results go to the disk cache, so the
next process will load executables instead (`loaded`), and a term already compiled in
this process costs nothing (`reused`).

Lowering and compiling run on a pool of threads, one per CPU core unless
[`CompileOptions.threads`][jaxcip.CompileOptions] says otherwise. XLA compiles the
programs of a term and of different terms at the same time. Lowering is Python code that
holds the GIL, so it runs one term at a time. The `lower_seconds` and `compile_seconds` of
a build add up the time of every thread, and `wall_seconds` is the time the build took on
the clock. On four cores, 24 terms with different structure build in 37 s instead of 55 s.

<!-- name: test_performance -->
```python
import jax.numpy as jnp

import jaxcip as jcp


def rosenbrock(x, params):
    return jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


problem = jcp.Problem(
    variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(4,)),
    objective=rosenbrock,
)
first = jcp.solve(problem, options=jcp.SolveOptions(time_limit=60.0))
again = jcp.solve(problem, options=jcp.SolveOptions(time_limit=60.0))
print(first.build)
print(again.build)
```

<!-- name: test_performance -->
```python
assert again.build.lowered == again.build.compiled == again.build.loaded == 0
```

The caches key a term by its jaxpr with constants moved into the parameters and with
the entries it reads moved into its inputs. So `x[0] * x[1]` and `x[2] * x[3]` share
programs, and so do two functions that differ only in their numbers. Shapes are part of
the key, and a new shape compiles again.

The executables of a term hold about 200 memory mappings, and Linux allows 65,530 per
process by default. The process therefore keeps only the 64 most recently used terms
after their problems are gone. A single problem with hundreds of distinct terms can still
reach the limit, and XLA then fails to compile with "Failed to materialize symbols".
Raising `vm.max_map_count` with `sysctl` lifts it.
[`clear_caches`][jaxcip.clear_caches] forgets all terms of the process, which is useful
for timing the disk cache.

## Callback time

`solution.statistics` is a [`HandlerStatistics`][jaxcip.HandlerStatistics] with a
[`CallbackStatistics`][jaxcip.CallbackStatistics] for each callback: how often SCIP
called it, how many calls the caches answered, how many programs ran and the seconds
spent running them.

<!-- name: test_performance -->
```python
for name, counters in first.statistics._asdict().items():
    if counters.calls:
        print(f"{name:14} {counters.calls:6} {counters.hits:6} {counters.seconds:.3f}")
```

<!-- name: test_performance -->
```python
assert first.statistics.estimate.calls > 0
```

One program run on the CPU plugin takes a few microseconds for a small term. When the
sum of `seconds` is a large part of `solving_seconds`, the terms are large or SCIP calls
them very often. A large `estimate` count usually means many nodes, and a relaxation
option or a different split of the function may need fewer.

## Guidelines

- Pass data that changes as `params` and compile once. `compile` followed by many
  `solve` calls skips tracing, analysis and the cache lookup.
- Write rows of the same form as one [`BatchedConstraint`][jaxcip.BatchedConstraint].
  They share programs and each row gets only its own variables.
- Keep affine parts as plain sums. The analysis passes them to SCIP as linear
  coefficients, which SCIP propagates and separates exactly.
- Keep `split_sums` on unless a sum has many tiny summands. Separate terms relax over
  smaller boxes and give SCIP auxiliary variables to branch on.
- Measure with a build without coverage tracing, as described in the
  [installation](../installation.md#the-package). Line tracing in the Cython callbacks
  makes them several times slower.
