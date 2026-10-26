# Nonlinear solving in SCIP

This page covers the parts of SCIP that JAX-CIP plugs into. The full design is in
[Vigerske and Gleixner (2018)](references.md) and in the [SCIP
documentation](https://www.scipopt.org/doc/html/), and the [SCIP Optimization Suite 10.0
report](https://arxiv.org/abs/2511.18580) describes the version in use.

## Spatial branch-and-bound

SCIP solves a mixed-integer nonlinear problem

\[
\min_{x} \; c^\top x \quad \text{subject to} \quad
\ell_i \le g_i(x) \le u_i, \quad x \in [\underline{x}, \overline{x}], \quad
x_j \in \mathbb{Z} \text{ for } j \in J
\]

with a branch-and-bound tree over boxes. A nonlinear objective becomes a constraint on
an extra variable, which JAX-CIP adds itself. At each node SCIP solves an LP relaxation.
In that LP, linear inequalities that hold for all \(x\) in the node's box take the place
of every nonlinear constraint. The LP optimum is a lower bound for the node. If its
solution violates a constraint, SCIP either adds a cut or splits the box on a variable,
which is called spatial branching. Smaller boxes give tighter relaxations, so the bounds
converge to the global optimum. Primal heuristics, among them local NLP solves with
[Ipopt](https://coin-or.github.io/Ipopt/) ([Wächter and Biegler 2006](references.md)),
find feasible points that close the tree from above.

SCIP also tightens variable bounds at every node by propagation. Forward propagation
encloses the range of each expression from the bounds of its children, and reverse
propagation tightens the children from bounds on the expression ([Belotti et al.
2009](references.md)). Tighter variable bounds give tighter LP relaxations and prune
nodes without solving them.

## Expressions and their handlers

SCIP stores each nonlinear function as an expression graph. Every node has an expression
handler that knows its operator, such as `sum`, `pow` or `exp`. The nonlinear constraint
handler calls the handlers of a graph for the values, bounds and estimators that
branch-and-bound needs, and [nonlinear
handlers](https://www.scipopt.org/doc/html/NLHDLR.php) detect structure across several
nodes, such as quadratic forms, convex subexpressions and products of binaries.

An [expression handler](https://www.scipopt.org/doc/html/EXPRHDLR.php) implements
callbacks. JAX-CIP has two handlers that implement the same ones. The `jax` handler stands
for a scalar function. The `jaxblock` handler stands for one entry of a vector function,
with the variables that entry reads as children, and the entries of one vector function
share their program runs ([Blocks](jax.md#blocks)). The callbacks are:

| Callback | What SCIP uses it for |
| --- | --- |
| `EVAL` | The value at a point, for feasibility checks and heuristics |
| `BWDIFF` | One partial derivative, for gradients in cuts and the NLP |
| `FWDIFF` | A directional derivative, which SCIP's Ipopt interface uses for gradients |
| `BWFWDIFF` | Hessian-vector products, through `SCIPevalExprHessianDir` |
| `INTEVAL` | An interval that encloses the range over the children's bounds, for forward propagation |
| `REVERSEPROP` | Tighter bounds on the children from bounds on the value |
| `ESTIMATE` | A linear under- or overestimator at a point over a box, the cuts of the LP relaxation |
| `INITESTIMATES` | Estimators for the first LP, before any point is known |
| `MONOTONICITY` | Whether the function increases or decreases in one child |

An estimator is a linear function \(a^\top x + b\) with \(a^\top x + b \le f(x)\) on the
whole box for an underestimator, and the reverse for an overestimator. SCIP turns it
into the cut \(a^\top x + b \le z\) for the auxiliary variable \(z\) of the expression.
An estimator that touches \(f\) at the LP solution cuts it off whenever the solution
violates the constraint. A convex underestimator of \(f\) has a tangent plane at every
point that is such an estimator. This is why JAX-CIP builds convex and concave
relaxations with JAXormick and returns their tangent planes.

SCIP's own handlers give tight, often exact, answers for their single operator, and SCIP
combines them across the graph. A JAX term is one node with many children, so SCIP
cannot look inside it. [JAX functions in SCIP](jax.md) describes how JAX-CIP keeps that
loss small, by cutting a function into the largest nonlinear values and giving each
entry of a cut only the variables it reads.

## Hessians and Ipopt

SCIP passes nonlinear problems to Ipopt through its NLP interface, which evaluates
expression graphs with [CppAD](https://cppad.readthedocs.io/). CppAD handles an
expression of a user handler as an atomic function that has values and first derivatives
only. SCIP 10.1 calls `BWFWDIFF` from `SCIPevalExprHessianDir` alone, so for a problem
with JAX terms Ipopt runs with a limited-memory quasi-Newton approximation of the
Hessian. Its local solves then take more iterations than for the same problem written
with SCIP's operators.
