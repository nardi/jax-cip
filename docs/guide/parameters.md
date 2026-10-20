# Parameters and batched constraints

## Parameters

The second argument of every function is `params`, any pytree of arrays. Data that
changes between solves belongs there instead of in a closure. [`compile`][jaxcip.compile]
traces the functions for the structure and shapes of `params`, and the result then solves
for any values of that layout without tracing or compiling again:

<!-- name: test_parameters -->
```python
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import jaxcip as jcp


class Site(NamedTuple):
    centre: jax.Array
    radius: jax.Array


def distance(x, params):
    return jnp.sum((x - params.centre) ** 2)


problem = jcp.Problem(
    variables=jcp.Variable(lower=-2.0, upper=2.0, shape=(2,)),
    objective=lambda x, params: jnp.sum(jnp.sin(3 * x) + x**2),
    constraints=(
        jcp.Constraint(
            lambda x, params: distance(x, params) - params.radius**2, upper=0.0
        ),
    ),
)
layout = Site(
    centre=jax.ShapeDtypeStruct((2,), jnp.float64),
    radius=jax.ShapeDtypeStruct((), jnp.float64),
)
compiled = jcp.compile(problem, layout)

for radius in (0.5, 1.0, 1.5):
    site = Site(centre=jnp.array([0.5, -0.5]), radius=jnp.array(radius))
    solution = jcp.solve(compiled, site)
    print(radius, solution.objective, solution.x)
```

<!-- name: test_parameters -->
```python
assert solution.status is jcp.Status.OPTIMAL
assert solution.build.compiled == 0
```

`compile` accepts arrays or `jax.ShapeDtypeStruct`s, since it only looks at shapes and
types. Calling [`solve`][jaxcip.solve] with a `Problem` compiles it on the way. That is
cheap after the first time, because compiled terms are cached in the process and on
disk. `solution.build` says what a solve had to build.

## Batched constraints

A model often has many rows of the same form over different variables. A
[`BatchedConstraint`][jaxcip.BatchedConstraint] states them with one function of a row.
`inputs` holds the indices of each row's variables in the flat vector of all variables,
the leaves of the variables pytree concatenated in the order of `jax.tree.leaves`.
`params` holds one entry per row along the leading axis:

<!-- name: test_parameters -->
```python
rng = np.random.default_rng(0)
rows = 100
inputs = rng.integers(0, 20, size=(rows, 2))
weights = rng.uniform(0.5, 2.0, size=(rows, 2))


def row(x, weight):
    return weight[0] * x[0] ** 2 + jnp.exp(weight[1] * x[1])


batched = jcp.Problem(
    variables=jcp.Variable(lower=0.0, upper=3.0, shape=(20,)),
    objective=lambda x, params: -jnp.sum(x),
    constraints=(
        jcp.BatchedConstraint(row, inputs=inputs, params=weights, upper=10.0),
    ),
)
solution = jcp.solve(batched, options=jcp.SolveOptions(time_limit=60.0))
```

<!-- name: test_parameters -->
```python
assert solution.status is jcp.Status.OPTIMAL
```

All 100 rows share the programs compiled for `row`, so compiling costs the same as for
one row. With the default `split_sums`, `row` itself becomes two terms, one for each
summand. The same model written as a `Constraint` with a vector output would also work,
but every row would then depend on every variable that some row reads, because the
analysis tracks dependencies per leaf and per static slice. Prefer a `BatchedConstraint`
for rows over gathered variables.

The parameters of a batched constraint are fixed when it is built. Parameters that
change between solves go in the `params` of `solve`, which the batched function does not
see.
