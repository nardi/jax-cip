# Options

Options are `NamedTuple`s with defaults, so you change a setting with `_replace` or by
passing only the fields that differ. Every choice between named alternatives is an
`Enum`.

## Solve options

[`SolveOptions`][jaxcip.SolveOptions] contains the limits of one solve, the
[`Emphasis`][jaxcip.Emphasis] of SCIP's predefined settings and the
[`Verbosity`][jaxcip.Verbosity] of its log. A [`ScipSetting`][jaxcip.ScipSetting] sets
any other SCIP parameter. [`ScipParameter`][jaxcip.ScipParameter] has a member for every
parameter of SCIP and its default plugins, named after the parameter's path, so
`numerics/feastol` is `ScipParameter.NUMERICS_FEASTOL`. A value of the wrong type raises
a `TypeError` before it is passed to SCIP.

<!-- name: test_options -->
```python
import jax.numpy as jnp

import jaxcip as jcp

problem = jcp.Problem(
    variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(3,)),
    objective=lambda x, params: jnp.sum(jnp.cos(2 * x) * x),
)
options = jcp.SolveOptions(
    time_limit=30.0,
    gap_limit=1e-6,
    emphasis=jcp.Emphasis.OPTIMALITY,
    settings=(
        jcp.ScipSetting(jcp.ScipParameter.NUMERICS_FEASTOL, 1e-7),
        jcp.ScipSetting(jcp.ScipParameter.LIMITS_SOLUTIONS, 500),
    ),
)
solution = jcp.solve(problem, options=options)
```

<!-- name: test_options -->
```python
assert solution.status is jcp.Status.OPTIMAL
```

## Compile options

[`CompileOptions`][jaxcip.CompileOptions] contains the settings for converting functions
into SCIP expressions.

- `split_sums` splits a function at its largest nonlinear values, such as each summand
  of a sum or each entry of a vector inside a `jnp.sum`, and makes each of these *cuts*
  its own term. Without it the whole output is one term. [JAX functions in
  SCIP](../theory/jax.md#jaxpr-analysis) explains the trade-off.
- `gradient_interval_limit` is the size, in jaxpr equations, above which a term has no
  gradient interval program and so no monotonicity.
- With `disk_cache`, lowered programs and executables are stored in `CYXLA_CACHE_DIR`.
- `threads` is the number of threads that lower and compile programs. It defaults to the
  number of CPU cores, and with 1, everything is built on the calling thread.
- `relaxation` and `handler` are the two groups below.

### Relaxation options

[`RelaxationOptions`][jaxcip.RelaxationOptions] contains the options that are passed to
[JAXormick](https://nardi.github.io/jaxormick/). With `alpha_bb`, the McCormick
relaxations are intersected with αBB relaxations, which are tighter on small boxes for
functions with bounded curvature. With `tighten_bounds`, the intervals of intermediate
values are narrowed. Both options make every estimator more expensive to compute.
[`RelaxationMode`][jaxcip.RelaxationMode] selects forward or reverse mode for the
subgradients. Reverse mode is faster for terms with many inputs.

<!-- name: test_options -->
```python
tight = jcp.CompileOptions(
    relaxation=jcp.RelaxationOptions(alpha_bb=True, mode=jcp.RelaxationMode.REVERSE)
)
solution = jcp.solve(problem, compile_options=tight)
assert solution.status is jcp.Status.OPTIMAL
```

### Handler options

[`HandlerOptions`][jaxcip.HandlerOptions] has a flag for each part of the expression
handler that can be switched off. None of them changes the results. They are for
measuring the effect of each part. Among them, `prefetch` controls whether the handler
sets the inputs of the other entries of a vector function from SCIP's state before it
runs a program for one entry. With those inputs set, one run usually computes the
results that the other entries need as well. The [performance](performance.md) page
shows how to read the counters that are included in every solution.

## Native SCIP

[`solve_file`][jaxcip.solve_file] reads a problem file with SCIP's own readers, in any
format SCIP reads, such as `.cip`, `.lp`, `.mps` or `.osil`, and solves it with the same
`SolveOptions`. The [comparison with native SCIP](comparison.md) uses it to check
JAX-CIP against SCIP on the same models. The `x` of its result is in SCIP's own order of
the variables, which has binary variables first, then integers, then continuous ones.
