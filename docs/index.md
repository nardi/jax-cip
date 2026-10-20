# JAX-CIP

JAX functions as nonlinear expressions in the [SCIP](https://www.scipopt.org/)
optimization solver.

SCIP finds global optima of mixed-integer nonlinear problems, but it needs each
nonlinear function as an expression tree of its own operators. JAX-CIP gives SCIP a JAX
function instead. It compiles the function into programs for its value, gradient,
Hessian-vector products, interval bounds and the convex and concave relaxations from
[JAXormick](https://nardi.github.io/jaxormick/), and SCIP calls them through an
expression handler. A problem can then contain anything that JAX traces and JAXormick
relaxes, including `lax.fori_loop`, `lax.scan`, `jnp.linalg` and special functions.

A problem is an immutable description over a pytree of variables:

- [`Problem`][jaxcip.Problem] holds the variables, the objective and the constraints,
- [`compile`][jaxcip.compile] turns its functions into SCIP expressions once, and
- [`solve`][jaxcip.solve] solves it for any parameter values with the same shapes.

[`solve_file`][jaxcip.solve_file] solves a problem file with SCIP's own readers, which
gives a reference for the same problem written natively.

## Quick example

This example minimizes the six-hump camel function over the part of a box that lies
inside the unit disc.

<!-- name: test_index -->
```python
import jax.numpy as jnp

import jaxcip as jcp


def camel(x, params):
    first, second = x[0], x[1]
    return (
        (4 - 2.1 * first**2 + first**4 / 3) * first**2
        + first * second
        + (-4 + 4 * second**2) * second**2
    )


problem = jcp.Problem(
    variables=jcp.Variable(lower=jnp.array([-3.0, -2.0]), upper=jnp.array([3.0, 2.0])),
    objective=camel,
    constraints=(jcp.Constraint(lambda x, params: jnp.sum(x**2), upper=1.0),),
)
solution = jcp.solve(problem)
print(solution.status, solution.objective, solution.x)
```

<!-- name: test_index -->
```python
assert solution.status is jcp.Status.OPTIMAL
assert abs(solution.objective - (-1.0316)) < 1e-3
```

## Comparison with native SCIP

JAX-CIP solves 331 of the 427 small [MINLPLib](https://www.minlplib.org/) instances
that native SCIP solves within a minute, and every optimum it reports is the known one.
It needs more nodes and more time, because SCIP sees each JAX term as one opaque function and loses the
structure it detects in its own expressions. The [comparison page](guide/comparison.md)
lists the numbers for every instance.

## License

JAX-CIP is licensed under the [Hippocratic License
3.0](https://firstdonoharm.dev/version/3/0/core.html).
