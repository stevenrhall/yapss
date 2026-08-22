"""

Scalar entries in user-supplied derivative callbacks.

User callbacks treat states and controls as scalars, so a derivative that is constant
is naturally written as a scalar -- ``hessian[key] = 2.0`` -- and that must be accepted
wherever an array over the time grid would be. Broadcasting by hand is never required.

This problem is built so that a constant derivative occurs in every place the assembly
reads one: a Jacobian entry used directly, a Jacobian entry of a defect and of an
integrand read by the Hessian chain-rule blocks (``df/dt``, ``dg/dt``), and Hessian
entries for a variable pair, a mixed variable/time pair, and the time/time pair. The
user-supplied derivatives are checked against automatic differentiation, which is
exact, so agreement must be to roundoff.

"""

import numpy as np
import pytest

from yapss import Problem
from yapss._private.auto import make_auto_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss._private.user import make_user_functions


def build_problem(method):
    problem = Problem(name="scalar-entries", nx=[2], nu=[1], nq=[1], nh=[1])

    def objective(arg):
        arg.objective = arg.phase[0].integral[0]

    def objective_gradient(arg):
        arg.gradient[0, "q", 0] = 1.0

    def objective_hessian(arg):
        pass

    def continuous(arg):
        x = arg.phase[0].state
        (u,) = arg.phase[0].control
        t = arg.phase[0].time
        arg.phase[0].dynamics[:] = [3.0 * t + x[0] * u, t**2 + x[1]]
        arg.phase[0].integrand[:] = [2.0 * t + x[0] ** 2]
        arg.phase[0].path[:] = [t * u]

    def continuous_jacobian(arg):
        x = arg.phase[0].state
        (u,) = arg.phase[0].control
        t = arg.phase[0].time
        jacobian = arg.phase[0].jacobian
        jacobian[("f", 0), ("x", 0)] = u
        jacobian[("f", 0), ("u", 0)] = x[0]
        jacobian[("f", 0), ("t", 0)] = 3.0  # constant df/dt, read by a chain-rule block
        jacobian[("f", 1), ("x", 1)] = 1.0  # constant, read directly
        jacobian[("f", 1), ("t", 0)] = 2.0 * t
        jacobian[("g", 0), ("x", 0)] = 2.0 * x[0]
        jacobian[("g", 0), ("t", 0)] = 2.0  # constant dg/dt, read by a chain-rule block
        jacobian[("h", 0), ("u", 0)] = t
        jacobian[("h", 0), ("t", 0)] = u

    def continuous_hessian(arg):
        hessian = arg.phase[0].hessian
        hessian[("f", 0), ("x", 0), ("u", 0)] = 1.0  # variable pair
        hessian[("f", 1), ("t", 0), ("t", 0)] = 2.0  # time/time pair
        hessian[("g", 0), ("x", 0), ("x", 0)] = 2.0  # integrand, variable pair
        hessian[("h", 0), ("t", 0), ("u", 0)] = 1.0  # path, mixed variable/time pair

    functions = problem.functions
    functions.objective = objective
    functions.objective_gradient = objective_gradient
    functions.objective_hessian = objective_hessian
    functions.continuous = continuous
    functions.continuous_jacobian = continuous_jacobian
    functions.continuous_hessian = continuous_hessian

    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower, bounds.initial_time.upper = 0.0, 0.5
    bounds.final_time.lower, bounds.final_time.upper = 1.0, 2.0
    bounds.state.lower[:], bounds.state.upper[:] = -5.0, 5.0
    bounds.control.lower[:], bounds.control.upper[:] = -2.0, 2.0
    bounds.path.lower[:], bounds.path.upper[:] = -50.0, 50.0
    guess = problem.guess.phase[0]
    guess.time = [0.2, 1.5]
    guess.state = [[1.0, 2.0], [-1.0, 1.0]]
    guess.control = [[0.5, -0.5]]
    guess.integral = [1.0]

    problem.derivatives.method = method
    problem.derivatives.order = "second"
    problem.validate()
    return problem


def build_nlp(method, spectral_method):
    problem = build_problem(method)
    problem.spectral_method = spectral_method
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)
    if method == "auto":
        functions = make_auto_functions(problem)
    else:
        functions = make_user_functions(problem, z0, mesh.tau_u)
    nlp = NLP(problem, functions, mesh)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(len(z0)))
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(len(nlp.constraints(z))))
    return nlp, z, lam


def dense(structure, values, shape):
    out = np.zeros(shape)
    for r, c, v in zip(*structure, values, strict=True):
        out[r, c] += v
    return out


@pytest.mark.parametrize("spectral_method", ["lg", "lgr", "lgl"])
def test_scalar_entries_match_auto(spectral_method):
    """Scalar derivative entries assemble to the same Jacobian and Hessian as exact AD."""
    nlp_user, z, lam = build_nlp("user", spectral_method)
    nlp_auto, _, _ = build_nlp("auto", spectral_method)
    nz = len(z)
    sigma = np.float64(1.3)

    nrow = max(nlp_auto.jacobianstructure()[0]) + 1
    jac_user = dense(nlp_user.jacobianstructure(), nlp_user.jacobian(z), (nrow, nz))
    jac_auto = dense(nlp_auto.jacobianstructure(), nlp_auto.jacobian(z), (nrow, nz))
    np.testing.assert_allclose(jac_user, jac_auto, rtol=1e-11, atol=1e-12)

    def symmetric(nlp):
        row, col = nlp.hessianstructure()
        lower = np.zeros((nz, nz))
        for r, c, v in zip(row, col, nlp.hessian(z, lam, sigma), strict=True):
            i, j = (r, c) if r >= c else (c, r)
            lower[i, j] += v
        return lower + lower.T - np.diag(np.diag(lower))

    np.testing.assert_allclose(symmetric(nlp_user), symmetric(nlp_auto), rtol=1e-11, atol=1e-12)


def test_scalar_entries_solve():
    """End to end: the problem solves with scalar entries, through Problem.solve()."""
    problem = build_problem("user")
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    assert solution.nlp_info.ipopt_status in (0, 1)
