# JAX-CIP

JAX functions as nonlinear expressions in the [SCIP](https://www.scipopt.org/)
optimization solver.

JAX-CIP compiles a JAX function into programs for its value, derivatives, interval
bounds and the McCormick relaxations of [JAXormick](https://nardi.github.io/jaxormick/).
SCIP calls them through an expression handler while it searches for a global optimum, so
a mixed-integer nonlinear problem can contain anything that JAX traces and JAXormick
relaxes.

## Installation

JAX-CIP builds from source with Meson and links SCIP 10.1 from the SCIP_jll binaries. It
runs its programs through a PJRT plugin, which a script downloads:

```bash
git clone https://github.com/nardi/jax-cip
cd jax-cip
uv sync
scripts/fetch_pjrt_plugin.sh ~/.cache/jaxcip/pjrt
export JAXCIP_PJRT_PLUGIN=~/.cache/jaxcip/pjrt/libpjrt_cpu.so
```

## Example

<!-- name: test_readme -->
```python
import jax.numpy as jnp

import jaxcip as jcp


def objective(x, params):
    return jnp.sum(x**4 - 16 * x**2 + 5 * x) / 2


problem = jcp.Problem(
    variables=jcp.Variable(lower=-5.0, upper=5.0, shape=(3,)),
    objective=objective,
    constraints=(jcp.Constraint(lambda x, params: x[0] * x[1] + x[2], lower=1.0),),
)
solution = jcp.solve(problem, options=jcp.SolveOptions(time_limit=60.0))
print(solution.status, solution.objective, solution.x)
```

<!-- name: test_readme -->
```python
assert solution.status is jcp.Status.OPTIMAL
```

## Documentation

The [documentation](https://nardi.github.io/jax-cip/) explains how SCIP uses JAX
functions, and has a user guide, the API reference and a comparison with native SCIP on
MINLPLib.

## License

[Hippocratic License 3.0](LICENSE.md).
