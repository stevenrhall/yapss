"""

Test the derivatives of a small nonlinear optimal control problem.

"""

# standard library imports
import re

# third party imports
import pytest

#  package imports
from yapss import Problem


def optimal_control_problem():
    """Small nonlinear optimal control problem."""
    ocp = Problem(name="Test", nx=[1], nq=[1], nh=[1], nu=[1], ns=1)

    def continuous(arg):
        (s,) = arg.parameter
        for p in arg.phase_list:
            (x,) = arg.phase[p].state
            (u,) = arg.phase[p].control
            t = arg.phase[p].time
            arg.phase[p].dynamics[:] = (t * t * x * x * u * u * s * s,)
            arg.phase[p].integrand[:] = (t * t * x * x * u * u * s * s,)
            arg.phase[p].path[:] = (t * t * x * x * u * u * s * s,)

    def objective(arg):
        arg.objective = 0

    ocp.functions.objective = objective
    ocp.functions.objective_gradient = lambda _: None
    ocp.functions.continuous = continuous
    ocp.functions.continuous_jacobian = lambda _: None
    ocp.functions.discrete = lambda _: None
    ocp.functions.discrete_jacobian = lambda _: None
    ocp.functions.discrete_hessian = lambda _: None

    for q in range(1):
        m, n = 1, 3
        ocp.mesh.phase[q].collocation_points = m * (n,)
        ocp.mesh.phase[q].fraction = m * (1.0 / m,)

    ocp.guess.phase[0].integral = (1,)
    ocp.guess.parameter = (5,)
    ocp.guess.phase[0].time = (0, 1)
    ocp.guess.phase[0].state = ((2, 1),)
    ocp.guess.phase[0].control = ((-1, 1),)
    ocp.bounds.phase[0].initial_time.lower = ocp.bounds.phase[0].initial_time.upper = 0
    ocp.bounds.phase[0].final_time.lower = ocp.bounds.phase[0].final_time.upper = 1
    ocp.bounds.phase[0].state.lower = (-1,)
    ocp.bounds.phase[0].state.upper = (+1,)
    ocp.bounds.phase[0].control.lower = (-1,)
    ocp.bounds.phase[0].control.upper = (+1,)

    ocp.ipopt_options.max_iter = 10
    ocp.ipopt_options.derivative_test_perturbation = 1e-6
    ocp.ipopt_options.derivative_test_tol = 1e-3

    return ocp


parameters = [
    (derivative_method, spectral_method)
    for spectral_method in ("lg", "lgr", "lgl")
    for derivative_method in ("auto", "central-difference")
]


@pytest.mark.parametrize(("derivative_method", "spectral_method"), parameters)
def test_derivatives(capfd, derivative_method, spectral_method):
    """Test the derivatives of the small nonlinear optimal control problem."""
    ocp = optimal_control_problem()
    ocp.derivatives.order = "second"
    ocp.ipopt_options.derivative_test = "second-order"
    ocp.ipopt_options.derivative_test_perturbation = 1e-7
    ocp.ipopt_options.derivative_test_tol = 3e-3
    ocp.derivatives.method = derivative_method
    ocp.spectral_method = spectral_method
    ocp.solve()
    captured = capfd.readouterr().out
    print(captured)
    assert len(re.findall("No errors detected by derivative checker", captured)) == 1


if __name__ == "__main__":
    pytest.main([__file__])
