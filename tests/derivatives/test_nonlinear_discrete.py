# ruff: noqa: D100, D103
# standard library imports
import re

# third party imports
import pytest

#  package imports
from yapss import Problem


def optimal_control_problem():
    ocp = Problem(name="Test_Discrete_Derivatives", nx=[2, 2], nq=[1, 2], ns=2, nd=3)

    def discrete(arg):
        x00 = arg.phase[0].initial_state
        x01 = arg.phase[1].initial_state
        xf0 = arg.phase[0].final_state
        xf1 = arg.phase[1].final_state
        t00 = arg.phase[0].initial_time
        t01 = arg.phase[1].initial_time
        tf0 = arg.phase[0].final_time
        tf1 = arg.phase[1].final_time
        q0 = arg.phase[0].integral
        q1 = arg.phase[1].integral
        s = arg.parameter

        variables = (x00, x01, xf0, xf1, q0, q1, s, [t00, t01, tf0, tf1])
        variables = tuple(var for var1 in variables for var in var1)

        d1 = 0.0
        d2 = 0.0
        d3 = 0.0
        for i, item1 in enumerate(variables):
            for item2 in variables[: i + 1]:
                d1 += (item1 + 3) * (item2 - 2)
                d2 += item1 + 2

        # TODO: Fix that line below gives a warning
        # Below gives a dprecation warning, as it should. The reason is that
        # arg.phase[0].final_time is a 1 element array, and should be a scalar
        arg.discrete[:] = d1, d2, d3

    def objective(arg):
        arg.objective = 0

    def continuous(arg):
        for p in range(2):
            t = arg.phase[p].time
            arg.phase[p].dynamics = [0 * t, 0 * t]

    def continuous_jacobian(_):
        return

    def discrete_jacobian(_):
        return

    ocp.functions.discrete_jacobian = discrete_jacobian
    ocp.functions.objective = objective
    ocp.functions.continuous = continuous
    ocp.functions.discrete = discrete
    ocp.functions.continuous_jacobian = continuous_jacobian

    for q in range(2):
        m, n = 1, 5
        ocp.mesh.phase[q].collocation_points = m * (n,)
        ocp.mesh.phase[q].fraction = m * (1.0 / m,)

    ocp.guess.phase[0].integral = (2,)
    ocp.guess.phase[1].integral = 3, 4
    ocp.guess.parameter = 1, 1
    ocp.guess.phase[0].time = (0, 1)
    ocp.guess.phase[1].time = (0, 1)
    ocp.guess.phase[0].state = ((0, 1), (0, 1))
    ocp.guess.phase[1].state = ((0, 1), (0, 1))
    ocp.guess.parameter = 2, 3

    ocp.derivatives.order = "second"
    ocp.ipopt_options.derivative_test = "second-order"
    ocp.ipopt_options.max_iter = 0
    ocp.ipopt_options.derivative_test_perturbation = 1e-7
    ocp.ipopt_options.derivative_test_tol = 1e-3

    return ocp


parameters = [
    (derivative_method, spectral_method)
    for spectral_method in ("lg", "lgr", "lgl")
    for derivative_method in ("auto", "central-difference")
]


@pytest.mark.parametrize(("derivative_method", "spectral_method"), parameters)
def test_derivatives(capfd, derivative_method, spectral_method):
    ocp = optimal_control_problem()
    ocp.derivatives.method = derivative_method

    # IPOPT doesn't work well enough to test central difference hessians
    ocp.ipopt_options.derivative_test = "first-order"
    # for mode in ("lgl", "lgr"):
    ocp.spectral_method = spectral_method
    ocp.solve()
    captured = capfd.readouterr().out
    assert len(re.findall("No errors detected by derivative checker", captured)) == 1


if __name__ == "__main__":
    pytest.main([__file__])
