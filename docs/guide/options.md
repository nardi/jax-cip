# Options

Options are `NamedTuple`s with defaults, so you change a setting with `_replace` or by
passing only the fields that differ. Every choice between named alternatives is an
`Enum`.

## Solve options

[`SolveOptions`][jaxcip.SolveOptions] holds the limits of one solve, the
[`Emphasis`][jaxcip.Emphasis] of SCIP's predefined settings and the
[`Verbosity`][jaxcip.Verbosity] of its log. A [`ScipSetting`][jaxcip.ScipSetting] sets
any other SCIP parameter. [`ScipParameter`][jaxcip.ScipParameter] has a member for every
parameter of SCIP and its default plugins, named after the parameter's path, so
`numerics/feastol` is `ScipParameter.NUMERICS_FEASTOL`. A value of the wrong type raises
a `TypeError` before SCIP sees it.

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

[`CompileOptions`][jaxcip.CompileOptions] decides how functions become SCIP expressions.

- `split_sums` cuts a function at the largest nonlinear values, such as each summand of
  a sum or each entry of a vector inside a `jnp.sum`, and makes each cut its own term.
  Without it the whole output is one term. [JAX functions in
  SCIP](../theory/jax.md#jaxpr-analysis) explains the trade-off.
- `gradient_interval_limit` is the size, in jaxpr equations, above which a term has no
  gradient interval program and so no monotonicity.
- `disk_cache` stores lowered programs and executables in `CYXLA_CACHE_DIR`.
- `threads` is the number of threads that lower and compile programs. It defaults to the
  number of CPU cores, and 1 builds everything on the calling thread.
- `relaxation` and `handler` are the two groups below.

### Relaxation options

[`RelaxationOptions`][jaxcip.RelaxationOptions] passes options to
[JAXormick](https://nardi.github.io/jaxormick/). `alpha_bb` intersects the McCormick
relaxations with αBB relaxations, which are tighter on small boxes for functions with
bounded curvature. `tighten_bounds` narrows the intervals of intermediate values. Both
make every estimator more expensive. [`RelaxationMode`][jaxcip.RelaxationMode] picks
forward or reverse mode for the subgradients. Reverse mode is faster for terms with many
inputs.

<!-- name: test_options -->
```python
tight = jcp.CompileOptions(
    relaxation=jcp.RelaxationOptions(alpha_bb=True, mode=jcp.RelaxationMode.REVERSE)
)
solution = jcp.solve(problem, compile_options=tight)
assert solution.status is jcp.Status.OPTIMAL
```

### Handler options

[`HandlerOptions`][jaxcip.HandlerOptions] switches off parts of the expression handler.
None of them changes the results. They exist to measure what each part gains. Among them,
`prefetch` lets the first entry of a vector function that needs a program run fill the
inputs of all its entries from SCIP's state, so that one run answers them all. The
[performance](performance.md) page shows how to read the counters that come with every
solution.

## Native SCIP

[`solve_file`][jaxcip.solve_file] reads a problem file with SCIP's own readers, in any
format SCIP reads, such as `.cip`, `.lp`, `.mps` or `.osil`, and solves it with the same
`SolveOptions`. The [comparison with native SCIP](comparison.md) uses it to check
JAX-CIP against SCIP on the same models. Its `x` follows SCIP's own order of the
variables, which puts binary variables first, then integers, then continuous ones.
