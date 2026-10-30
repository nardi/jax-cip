# Saving and loading

Compiling a problem traces its functions, finds the sparsity of block terms, lowers the
programs of every term and compiles them. The [disk cache](performance.md) shortens this
on a machine that has done it before. A file shortens it everywhere. `save` writes a
compiled problem to a file, and `load` reads it in another process, on another machine,
or later, without tracing, detecting sparsity or compiling anything.

## Saving

<!-- name: test_docs_saving -->
```python
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxcip as jcp

problem = jcp.Problem(
    variables={"position": jcp.Variable(lower=-2.0, upper=2.0, shape=(2,))},
    objective=lambda x, params: jnp.sum(x["position"] ** 4 - 2 * x["position"] ** 2)
    + jnp.dot(x["position"], params["target"]),
    constraints=(jcp.Constraint(lambda x, params: jnp.sum(x["position"]), upper=1.0),),
)
params = {"target": np.array([1.0, 1.0])}
compiled = jcp.compile(problem, params)

path = Path(tempfile.mkdtemp()) / "problem.jcp"
jcp.save(compiled, path)
```

The file holds the programs of every term and of the affine remainders with their
StableHLO, the sparsity patterns of block terms, the bounds and the compile options.
`.jcp` is the usual extension. `dumps` returns the same data as bytes, for sending it to a
worker process. Terms that appear more than once in a problem share their programs in the
file, just as they do in memory.

## Loading

The file does not hold the functions of the problem or the trees of its variables and
parameters. `load` takes the trees, so that a loaded problem works like a compiled one:
the solution uses the structure of `variables`, and `solve` takes parameters in the
structure of `params`.

<!-- name: test_docs_saving -->
```python
loaded = jcp.load(path, problem.variables, params)
solution = jcp.solve(loaded, params, options=jcp.SolveOptions(time_limit=60.0))
print(solution.x, solution.objective)
```

<!-- name: test_docs_saving -->
```python
assert loaded.problem is None
assert solution.status is jcp.Status.OPTIMAL
assert loaded.statistics.compiled == 0
```

The leaves of `variables` are `Variable`s or arrays, and the leaves of `params` are arrays
or `jax.ShapeDtypeStruct`s, as for `compile`. Only their paths and shapes matter. They
are checked against the file, and a tree with another leaf or shape raises an error that
names the first difference. The bounds and domains come from the file, so the
`Variable`s of the tree may carry any bounds, or the tree may hold plain arrays. Any
pytree that JAX can flatten works, including named tuples, dataclasses and Equinox
modules. The class of a node only has to exist in the loading process, since nothing
about it is stored.

## Other machines

A file stores one executable per program, the one of the machine that saved it, together
with the StableHLO of every program. A machine with the same operating system,
architecture and plugin version loads the executables directly, on as many threads as
`threads` says, or one per CPU core. On any other machine the plugin cannot use them, and
`load` compiles the StableHLO instead. That costs about as long as the compilation of a
problem without the lowering and the sparsity detection. Pass `on_mismatch="raise"` to
get a `cyxla.TargetMismatchError` in that case.

The programs always run on the CPU, so a file made on a machine without a GPU loads on a
machine with one and still runs on the CPU.

## What a loaded problem skips

Loading does not import asdex or numba, does not call the functions of the problem and
does not trace anything. Terms that the process compiled or loaded before are reused, so
loading a second file with the same terms is faster. With `warm_up=True`, which is the
default, each program runs once, so that the first solve is as fast as the later ones.
Importing JAX-CIP and JAX is the remaining cost of a new process.

A loaded problem has no jaxprs, so `problem` is `None`, and it cannot be recompiled with
other options. Compile the problem again for that.

## Trust

A file contains machine code, which the plugin runs when it loads the executables. Load
only files from a source that you trust. The file protects against damage with
checksums, and it does not protect against someone who writes one on purpose.

A compiled problem cannot be pickled, because the trees are not part of the file. Send
the bytes of `dumps` to a worker and call `loads` there with the trees.
