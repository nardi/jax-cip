# Parameters and rows

## Parameters

The second argument of every function is `params`, any pytree of arrays. Data that
changes between solves belongs there instead of in a closure.
[`compile`][jaxcip.compile] traces the functions for the structure and shapes of
`params`, and the result then solves for any values of that layout without tracing or
compiling again:

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
types. Calling [`solve`][jaxcip.solve] with a `Problem` will compile it first. After the
first time that is cheap, because the process and the disk cache keep the compiled
terms. `solution.build` counts what the solve had to build.

## Rows of one form

A model often has many rows of the same form over different variables. [`rows`][jaxcip.rows]
states them with one function of a row. Its `inputs` hold, for each row, the positions in a
vector of the entries that the row reads, and its `params` hold one entry per row along the
leading axis. The result is a function of the vector that returns one value per row:

<!-- name: test_parameters -->
```python
rng = np.random.default_rng(0)
count = 100
inputs = rng.integers(0, 20, size=(count, 2))
weights = rng.uniform(0.5, 2.0, size=(count, 2))


def row(x, weight):
    return weight[0] * x[0] ** 2 + jnp.exp(weight[1] * x[1])


evaluate = jcp.rows(row, inputs, weights)
many = jcp.Problem(
    variables=jcp.Variable(lower=0.0, upper=3.0, shape=(20,)),
    objective=lambda x, params: -jnp.sum(x),
    constraints=(jcp.Constraint(lambda x, params: evaluate(x), upper=10.0),),
)
solution = jcp.solve(many, options=jcp.SolveOptions(time_limit=60.0))
```

<!-- name: test_parameters -->
```python
assert solution.status is jcp.Status.OPTIMAL
```

All 100 rows share the programs compiled for `row`, so compiling costs the same as for one
row. `row` itself is split like any function: it becomes two terms, one for each summand,
and an affine part of a row would become linear coefficients. A summand reads only the
entries it uses. Each row gives SCIP only its own variables, as long as the vector is made of variables
alone. A reshape, a slice, a transpose or a concatenation of variables is fine, and so is
`jnp.stack` of scalar variables. A vector that is computed from the variables, such as
`jnp.exp(x)`, still works but makes every row depend on all the variables that some row
reads. The same holds when the rows are only part of the output, as in
`jnp.sum(evaluate(x))`. Write the rows as the whole value of a constraint, or add them to
other terms with `+` and `-`.

The `params` of `rows` may be built from the `params` of the problem, so rows can change
between solves. Their values enter the programs as inputs, and the compiled terms stay the
same:

<!-- name: test_parameters -->
```python
def scaled(x, params):
    return jcp.rows(row, inputs, params["scale"] * weights)(x)


tunable = jcp.Problem(
    variables=jcp.Variable(lower=0.0, upper=3.0, shape=(20,)),
    objective=lambda x, params: -jnp.sum(x),
    constraints=(jcp.Constraint(scaled, upper=10.0),),
)
compiled_rows = jcp.compile(tunable, {"scale": jnp.array(1.0)})
tighter = jcp.solve(compiled_rows, {"scale": jnp.array(1.5)})
```

<!-- name: test_parameters -->
```python
assert tighter.status is jcp.Status.OPTIMAL
assert tighter.build.compiled == 0
```
