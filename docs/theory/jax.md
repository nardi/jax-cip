# JAX functions in SCIP

This page follows a JAX function from `Problem` to the callbacks of SCIP's expression
handler. [Bradbury et al. (2018)](references.md) describe JAX itself, and [Frostig et
al. (2018)](references.md) its tracing design. The relaxations come from
[JAXormick](https://nardi.github.io/jaxormick/), whose documentation covers McCormick
relaxations ([McCormick 1976](references.md), [Mitsos et al. 2009](references.md)).

## Jaxpr analysis

[`compile`][jaxcip.compile] traces every function of a problem to a jaxpr, with the
leaves of the variables and of the parameters as inputs. The analysis walks back from the
output through operations that are linear in their nonlinear operands: additions,
subtractions, sums, products and matrix products with constants or parameters, reshapes,
slices, concatenations and the like. It stops at the largest nonlinear values, the
*cuts*. For `jnp.sum(x**4 - 16*x**2 + 5*x)` the cuts are `x**4` and `x**2`, each with one
entry per variable.

A constant factor stays inside the cut when the cut has a single entry, as in
`3.0*x[0]**2`. SCIP checks an auxiliary variable against its cut with an absolute
tolerance, so a factor outside the cut changes which solutions pass that check. On
`min 0.2x⁵ − 1.7x⁴ + x³ − 0.02x² + 10⁻⁴x` over `[0, 10]`, SCIP needs about 60 nodes with the
factors inside the cuts and between 80 and 40000, depending on the coefficients, with them
outside. The factors of a vector cut stay outside, in \(A\), where they keep the matrix
sparse.

Everything between the cuts and the output is affine in the cut entries \(F\), the
variables \(x\) and the constants. A row of the problem is therefore

\[
\text{lower} \le A\,F(x, p) + B\,x + c \le \text{upper},
\]

and ASDEX finds \(A\), \(B\) and \(c\) as the sparse Jacobian of that affine remainder
at zero. The sparsity pattern and its coloring are fixed when the problem is compiled.
Only their values depend on the parameters, so they are evaluated again at every solve:
a matrix \(W\) in the parameters, as in \(W\,f(x)\), gives \(A = W\) without
compiling again.

- The linear part \(B x + c\) goes to SCIP as linear coefficients. SCIP propagates and
  separates linear terms exactly, and inside an opaque function they would only weaken the
  relaxation. A constraint with no cut becomes a plain linear row.
- Each cut becomes a term, a jaxpr of its own inputs. A term keeps only the variable
  entries it reads. Static indexing such as `x[3]` or `x[2:5]` selects entries, and any
  other use of a leaf reads all of it.
- A cut whose entries all read most of its variables is not separated from the linear
  operation that follows it. Otherwise `jnp.sum` over \(n\) such entries would give SCIP
  \(n\) expressions with \(n\) children each, where the sum has one. The same holds for a
  cut with at most 64 entries times inputs, which would only add callbacks.

With `split_sums=False` in [`CompileOptions`][jaxcip.CompileOptions] the output is the
only cut. Separate cuts give SCIP more auxiliary variables and tighter relaxations, since
JAXormick relaxes each cut over its own smaller box. They also cost more LP rows, and
splitting is the default.

Every entry of a cut becomes one SCIP expression, whose children are the variables the
entry reads. A row is a sum of such expressions with the coefficients of \(A\), so one
SCIP nonlinear constraint per row.

## The six programs

A cut with one entry is a scalar function \(f(x, p)\) of two flat vectors, its variables
\(x\) and its parameters \(p\). JAX-CIP moves the constants that the function closes over
into \(p\), so two terms that differ only in their numbers have the same code. JAX-CIP
builds six programs of \(f\):

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

## Blocks

A cut with several entries is a *block*. The programs of a block take the same inputs as
above and give all entries at once, with their derivatives in the order of a fixed
sparsity pattern. ASDEX finds the pattern from the jaxpr, so the entries of `x[:-1] * x[1:]`
read two variables each, and an entry that reads no variable still gets one child with
derivative zero. A coloring of the pattern compresses every derivative: a program takes
one forward-mode derivative per color, not one per variable. The Hessian-vector program
is the derivative of the compressed Jacobian, the relaxation programs differentiate the
relaxation, and the gradient interval program is the interval extension of the compressed
Jacobian. Each entry's expression gets the entry's slice of those outputs.

A block of 64 entries therefore compiles in the time of one scalar cut, and a solve has no
dense Jacobian. Blocks have no warm start of roots, which would need one for each entry.
A term of [`rows`][jaxcip.rows] is the scalar function of one row, with a different
gather of the variables in each, and is not a block.

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
[StableHLO](https://openxla.org/stablehlo), and [cyxla](https://nardi.github.io/cyxla/)
compiles it with the PJRT plugin into an executable. The callbacks are C functions that
run the executables through the runtime of cyxla and the [PJRT C
API](https://openxla.org/xla/pjrt/pjrt_integration) without taking the GIL. The input
buffers are 64-byte aligned host memory that the plugin reads directly, and the handler
reads outputs from the plugin's buffers when it allows that, so a call copies nothing.

All expressions of one term share its executables and differ only in their parameter
vectors and children. A thousand [`rows`][jaxcip.rows] of one form therefore compile once.
JAX-CIP caches terms in the process, keyed by their jaxpr, signature and options. The disk cache stores the StableHLO and the executables, so a
later process will skip lowering and compiling.

## Caches in the handler

SCIP often asks for the same result twice, since `BWDIFF` follows `EVAL` at the same
point and propagation repeats interval evaluations on unchanged boxes. Each expression
keeps its last few points, boxes and estimators, and answers from them when the inputs
match bit for bit. `EVAL` also runs the value and gradient program, so the gradient that
SCIP asks for next is already there. You can switch each of these off in
[`HandlerOptions`][jaxcip.HandlerOptions] to measure what it gains.

The entries of a block share their runs. Every program run reads the inputs of the whole
block and keeps all entries' outputs, and a run is valid for an entry exactly when the
inputs the run read at the entry's children are the current ones, because the entry
depends on nothing else. When an entry needs a run, the handler first sets the inputs of
the other entries from SCIP's state: the point of the evaluation for `EVAL`, the bounds
that SCIP's interval evaluation gives for `INTEVAL`, and the LP solution with the local
bounds for `ESTIMATE`. The run then answers the next entries' callbacks too, one run per
callback type and round instead of one per entry. A wrong guess costs a run and never a
wrong result, since each entry checks its own children. The `prefetch` switch of
[`HandlerOptions`][jaxcip.HandlerOptions] turns this off.

Each SCIP instance keeps its own state for every block, and so does a copy that SCIP
makes for a heuristic, so solves in several threads share no input buffers.

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

The relaxation programs reuse the enclosures of roots in the same way. An enclosure that
the Krawczyk operator verified on a box holds the unique root for every point of that box,
so it also holds on every box inside it. Each expression stores the enclosures that its
runs verified together with their boxes. A relaxation program starts from the enclosure of
the smallest stored box that contains its own, and skips the solver and the search for an
enclosure. The program returns the enclosure it verified, which the handler stores in turn.
Branching shrinks boxes, so most boxes have a stored parent. The store holds the eight most
recent boxes, and a sibling box never uses an enclosure of its sibling, since the check is
that the stored box contains the current one. On a system of 6 unknowns, the convex and
interval programs take 100 to 150 µs on a box with a stored parent, against 250 to 280 µs without.

- A Newton step after the solver gives the same root to machine precision from every
  guess, so values that SCIP compares do not depend on the order of the callbacks. That
  step has no derivative, because `lax.custom_root` takes derivatives from the implicit
  function theorem.
- Only roots at the top level of a term take a warm start. A root inside `jax.jit`, a loop
  or a branch keeps its own guess.
- A block, a cut with several entries, takes no warm start. ASDEX cannot analyse a root, so
  every entry of such a cut reads every variable. The cut then joins the sum or product
  that follows it, and the root stays in a term with one entry.
- The root must be unique. JAXormick proves that on every box for the relaxations, but the
  value programs do not check it. With several roots, the guess picks one, and the value
  and the relaxations can disagree.
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
