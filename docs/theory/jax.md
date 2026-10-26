# JAX functions in SCIP

This page follows a JAX function from `Problem` to the callbacks of SCIP's expression
handler. [Bradbury et al. (2018)](references.md) describe JAX itself, and [Frostig et
al. (2018)](references.md) its tracing design. The relaxations come from
[JAXormick](https://nardi.github.io/jaxormick/), whose documentation covers McCormick
relaxations ([McCormick 1976](references.md), [Mitsos et al. 2009](references.md)).

## Jaxpr analysis

[`compile`][jaxcip.compile] traces every function of a problem to a jaxpr, with the
leaves of the variables and of the parameters as inputs. The analysis then splits the
output at additions, subtractions and negations into summands and classifies each
summand as constant, affine or nonlinear in the variables.

- The affine summands go to SCIP as linear coefficients. SCIP propagates and separates
  linear terms exactly, and inside an opaque function they would only weaken the
  relaxation. A constraint with no nonlinear summand becomes a plain linear row.
- Each nonlinear summand becomes a term, a jaxpr of its own inputs. A term keeps only
  the variable entries it reads. Static indexing such as `x[3]` or `x[2:5]` selects
  entries, and any other use of a leaf reads all of it.

With `split_sums=False` in [`CompileOptions`][jaxcip.CompileOptions] the nonlinear
summands stay one term. Separate terms give SCIP more auxiliary variables and tighter
relaxations, since JAXormick relaxes each term over its own smaller box. They also cost
more LP rows, and splitting is the default.

Every entry of a term's output becomes one SCIP expression, whose children are the
term's variables. The constraint `lower ≤ affine + Σ terms ≤ upper` then becomes one
SCIP nonlinear constraint per output entry.

## The six programs

A term is a scalar function \(f(x, p)\) of two flat vectors, its variables \(x\) and its
parameters \(p\). JAX-CIP moves the constants that the function closes over into \(p\),
so two terms that differ only in their numbers have the same code. JAX-CIP builds six
programs of \(f\):

| Program | Arguments | Result | Callbacks |
| --- | --- | --- | --- |
| value and gradient | \(x, p\) | \(f\), \(\nabla_x f\) | `EVAL`, `BWDIFF`, `FWDIFF` |
| Hessian-vector | \(x, d, p\) | \(\nabla_x^2 f \, d\) | `BWFWDIFF` |
| convex | \(x, \underline{x}, \overline{x}, p\) | convex relaxation and subgradient | `ESTIMATE`, `INITESTIMATES`, `REVERSEPROP` |
| concave | \(x, \underline{x}, \overline{x}, p\) | concave relaxation and supergradient | `ESTIMATE`, `INITESTIMATES`, `REVERSEPROP` |
| interval | \(\underline{x}, \overline{x}, p\) | bounds on \(f\) over the box | `INTEVAL` |
| gradient interval | \(\underline{x}, \overline{x}, p\) | bounds on \(\nabla_x f\) over the box | `MONOTONICITY` |

The Hessian-vector product is `jax.jvp` of `jax.grad`. The relaxations come from
`jaxormick.convex_side` and `jaxormick.concave_side`, and the intervals from
`jaxormick.interval`, with the options of
[`RelaxationOptions`][jaxcip.RelaxationOptions]. The gradient interval program is the
interval extension of `jax.grad`. XLA takes a long time to compile it for large
functions, so a term with more than `gradient_interval_limit` jaxpr equations has no
gradient interval program, and SCIP treats its monotonicity as unknown.

## Estimators from relaxations

`ESTIMATE` receives a reference point and the local box. The handler clamps the point
into the box and runs the convex program for an underestimator or the concave program
for an overestimator. A convex relaxation \(f^{cv}\) with subgradient \(s\) at \(x_0\)
gives the plane

\[
f(x) \;\ge\; f^{cv}(x) \;\ge\; f^{cv}(x_0) + s^\top (x - x_0)
\quad \text{for all } x \in [\underline{x}, \overline{x}],
\]

which is a valid underestimator on the whole box. When the local box equals the global
box, the handler marks the estimator as globally valid, so SCIP keeps the cut for the
whole tree. `INITESTIMATES` returns planes at the box center and at the points a quarter
and three quarters along its diagonal. `REVERSEPROP` uses the planes at the box center.
Given an upper bound \(u\) on \(f\), an underestimator gives the linear inequality
\(a^\top x + b \le u\), which bounds each variable. An overestimator does the same with
the lower bound of \(f\).

On a node where an input is unbounded, the handler returns no estimator, because the
relaxations need a finite box, and SCIP branches instead.

## Execution through PJRT

JAX-CIP exports each program with `jax.export` to
[StableHLO](https://openxla.org/stablehlo), and the PJRT plugin compiles it into an
executable. The callbacks are C functions that run the executables through the [PJRT C
API](https://openxla.org/xla/pjrt/pjrt_integration) without taking the GIL. The input
buffers are 64-byte aligned host memory that the plugin reads directly, and the handler
reads outputs from the plugin's buffers when it allows that, so a call copies nothing.

All expressions of one term share its executables and differ only in their parameter
vectors and children. A [`BatchedConstraint`][jaxcip.BatchedConstraint] of a thousand
rows therefore compiles once. JAX-CIP caches terms in the process, keyed by their jaxpr,
signature and options. The disk cache stores the StableHLO and the executables, so a
later process will skip lowering and compiling.

## Caches in the handler

SCIP often asks for the same result twice, since `BWDIFF` follows `EVAL` at the same
point and propagation repeats interval evaluations on unchanged boxes. Each expression
keeps its last few points, boxes and estimators, and answers from them when the inputs
match bit for bit. `EVAL` also runs the value and gradient program, so the gradient that
SCIP asks for next is already there. You can switch each of these off in
[`HandlerOptions`][jaxcip.HandlerOptions] to measure what it gains.

## Roots and warm starts

A term can contain `lax.custom_root`, which defines a value \(y(p)\) by the equation
\(r(p, y) = 0\) and a solver. JAXormick relaxes the root by proving, with the Krawczyk
operator, that the equation has exactly one solution on the box. The value and derivative
programs run the solver itself.

Newton's method needs a step or two when it starts close to the root, so the handler
passes the root of the previous run as the next guess. Each expression keeps its own
roots, and the gradient program returns them as a third output. The Hessian-vector program
takes them as an input. Entries that no run has set use the guess written in the term. For
a system of 12 unknowns, a run takes 68 µs when \(x\) repeats and 125 µs when \(x\) moves a
little, against 200 µs from the written guess.

- A Newton step after the solver gives the same root to machine precision from every
  guess, so values that SCIP compares do not depend on the order of the callbacks. That
  step has no derivative, because `lax.custom_root` takes derivatives from the implicit
  function theorem.
- Only roots at the top level of a term take a warm start. A root inside `jax.jit`, a loop
  or a branch keeps its own guess.
- The root must be unique. JAXormick proves that on every box for the relaxations, but the
  value programs do not check it. With several roots, the guess picks one, and the value
  and the relaxations can disagree.
- The PJRT plugin has no LAPACK, so a solver that calls `jnp.linalg.solve` or
  `jnp.linalg.inv` does not compile. A solver can write its linear algebra with
  elementwise operations and `lax.fori_loop`.
- JAXormick has no rule for the gradient of a root, so the term has no gradient interval
  program and SCIP treats its monotonicity as unknown.
- The Krawczyk operator only verifies a root on a box that is small compared with the
  nonlinearity of the equation. On a wider box the relaxations are infinite, which is
  valid but gives SCIP nothing until branching shrinks the box.

## Cost of an opaque term

SCIP's own expressions expose structure that a JAX term hides. SCIP's nonlinear handlers
detect a product of binaries, a quadratic form or a convex sum, and relax it exactly or
linearize it. A JAX term gets McCormick relaxations instead, which are weaker on such
structure, and branching has to make up the difference. The analysis above recovers the
affine parts and the separable sums. For other structure, write that part of the
function in a form the analysis splits off, or accept more nodes.
