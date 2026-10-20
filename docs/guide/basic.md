# Basic usage

## Variables

A [`Variable`][jaxcip.Variable] is an array of decision variables with bounds and a
[`Domain`][jaxcip.Domain]. Its shape is the broadcast of `shape` and the shapes of the
bounds. The variables of a problem are any pytree with a `Variable` at every leaf, and
the functions of the problem receive the same pytree with an array at every leaf. A
`NamedTuple` gives the leaves names:

<!-- name: test_basic -->
```python
from typing import NamedTuple

import jax
import jax.numpy as jnp

import jaxcip as jcp


class Plant(NamedTuple):
    flow: jcp.Variable | jax.Array
    units: jcp.Variable | jax.Array
    open: jcp.Variable | jax.Array


variables = Plant(
    flow=jcp.Variable(lower=0.1, upper=4.0, shape=(2,)),
    units=jcp.Variable(lower=0, upper=3, domain=jcp.Domain.INTEGER, shape=(2,)),
    open=jcp.Variable(domain=jcp.Domain.BINARY),
)
```

SCIP needs finite bounds on the inputs of a nonlinear function to build its
relaxations. Without them it can only branch until propagation bounds the variables.

## Objective and constraints

Every function of a problem has the signature `f(x, params)`. The objective returns a
scalar, and [`Sense`][jaxcip.Sense] picks the direction. A
[`Constraint`][jaxcip.Constraint] bounds every entry of its function's output:

<!-- name: test_basic -->
```python
def cost(x, params):
    return (
        (x.flow[0] - 2) ** 2
        + jnp.exp(0.5 * x.flow[1])
        - 3 * jnp.log(x.flow[0] + x.flow[1])
        + x.units[0] * x.flow[1]
        - 1.5 * x.units[1]
        + 2 * x.open
    )


problem = jcp.Problem(
    variables=variables,
    objective=cost,
    sense=jcp.Sense.MINIMIZE,
    constraints=(
        jcp.Constraint(
            lambda x, params: x.flow[0] * x.flow[1] + x.units[0] - x.units[1],
            lower=1.0,
        ),
        jcp.Constraint(
            lambda x, params: jnp.exp(x.flow[0] / 2) + x.open * x.flow[1],
            upper=4.0,
        ),
        jcp.Constraint(
            lambda x, params: jnp.stack([jnp.sum(x.units) + x.open, jnp.sum(x.flow)]),
            upper=jnp.array([4.0, 5.0]),
        ),
    ),
)
```

The last constraint has a vector output, so it gives two rows with their own bounds.
Both rows here are affine, so SCIP gets them as linear rows. A problem without an
objective is a feasibility problem.

## Solving

[`solve`][jaxcip.solve] compiles the problem and solves it. The result is a
[`Solution`][jaxcip.Solution], whose `x` has the structure of the variables:

<!-- name: test_basic -->
```python
solution = jcp.solve(problem, options=jcp.SolveOptions(time_limit=60.0))
print(solution.status, solution.objective, solution.dual_bound)
print(solution.x.flow, solution.x.units, solution.x.open)
```

<!-- name: test_basic -->
```python
assert solution.status is jcp.Status.OPTIMAL
assert isinstance(solution.x, Plant)
```

`status` is a [`Status`][jaxcip.Status] and says why SCIP stopped. With a time or node
limit, `x` is the best solution found so far, `dual_bound` the best proven bound, and
`gap` the relative distance between them. `x` is `None` when SCIP found no solution.

## What a function may contain

A function may use anything that `jax.make_jaxpr` traces and JAXormick relaxes. That
covers `jax.numpy` arithmetic and elementary functions, reductions, indexing,
`lax.fori_loop`, `lax.scan`, `lax.cond` and linear solves. The [supported operations of
JAXormick](https://nardi.github.io/jaxormick/guide/supported/) list the rest. Integer and
binary variables arrive as floats, so `x.units[0] * x.flow[1]` is a product of two
floats.
