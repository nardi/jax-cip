"""Solves a small problem with an installed JAX-CIP, to check that a wheel works."""

import jax.numpy as jnp

import jaxcip as jcp


def main() -> None:
    """Solves a nonconvex problem with a known global optimum."""
    problem = jcp.Problem(
        variables=jcp.Variable(lower=-5.0, upper=5.0, shape=(3,)),
        objective=lambda x, params: jnp.sum(x**4 - 16 * x**2 + 5 * x) / 2,
        constraints=(jcp.Constraint(lambda x, params: x[0] * x[1] + x[2], lower=1.0),),
    )
    solution = jcp.solve(
        problem,
        options=jcp.SolveOptions(time_limit=120.0, verbosity=jcp.Verbosity.QUIET),
    )
    assert solution.status is jcp.Status.OPTIMAL, solution.status
    assert abs(solution.objective - -117.4985) < 0.05, solution.objective
    print(f"SCIP {jcp.scip_version()} solved the problem in {solution.nodes} nodes.")


if __name__ == "__main__":
    main()
