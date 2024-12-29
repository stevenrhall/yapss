# standard library imports
import re

# third party imports
import pytest

#  package imports
from yapss import Problem


def optimal_control_problem(order):
    ocp = Problem(name="Test_Objective_Derivatives", nx=[3, 3], nq=[1, 3], ns=2)

    def objective(arg):
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

        variables = (x00, x01, xf0, xf1, [t00], [t01], [tf0], [tf1], q0, q1, s)
        variables = tuple(var for var1 in variables for var in var1)

        obj = 0.0
        if order == 2:
            for i, item1 in enumerate(variables):
                for item2 in variables[i:]:
                    obj += (item1 + 3) * (item2 - 2)

        elif order == 1:
            for item1 in variables:
                obj += item1 + 3

        else:
            obj = 0.0

        arg.objective = obj
        return

    def continuous(arg):
        if 0 in arg.phase_list:
            arg.phase[0].dynamics[:] = 0, 0, 0
            arg.phase[0].integrand[:] = (0,)
        if 1 in arg.phase_list:
            arg.phase[1].dynamics[:] = 0, 0, 0
            arg.phase[1].integrand[:] = 0, 0, 0
        return

    def discrete(_):
        return

    ocp.functions.objective = objective
    ocp.functions.continuous = continuous
    ocp.functions.discrete = discrete
    ocp.functions.objective_gradient = lambda _: None
    ocp.functions.objective_hessian = lambda _: None
    ocp.functions.continuous_jacobian = lambda _: None
    ocp.functions.continuous_hessian = lambda _: None
    ocp.functions.discrete_jacobian = lambda _: None
    ocp.functions.discrete_hessian = lambda _: None

    for p in range(2):
        m, n = 1, 5
        ocp.mesh.phase[p].collocation_points = m * (n,)
        ocp.mesh.phase[p].fraction = m * (1.0 / m,)

    ocp.guess.phase[0].integral = (1,)
    ocp.guess.phase[1].integral = 1, 1, 1
    ocp.guess.parameter = 1, 1
    ocp.guess.phase[0].time = (0, 1)
    ocp.guess.phase[1].time = (0, 1)
    ocp.guess.phase[0].state = ((0, 1), (0, 1), (0, 1))
    ocp.guess.phase[1].state = ((0, 1), (0, 1), (0, 1))

    ocp.derivatives.order = "second"
    ocp.ipopt_options.derivative_test = "second-order"
    ocp.ipopt_options.max_iter = 0

    return ocp


parameters = [
    (derivative_method, spectral_method)
    for spectral_method in ("lg", "lgr", "lgl")
    for derivative_method in ("auto", "central-difference")
]


@pytest.mark.parametrize(("derivative_method", "spectral_method"), parameters)
def test_derivatives(capfd, derivative_method, spectral_method):
    ocp = optimal_control_problem(order=2)
    ocp.derivatives.method = derivative_method
    ocp.spectral_method = spectral_method
    ocp.derivatives.order = "second"
    ocp.ipopt_options.derivative_test = "second-order"
    ocp.ipopt_options.derivative_test_perturbation = 3e-5
    ocp.ipopt_options.derivative_test_tol = 3e-3
    ocp.solve()
    captured = capfd.readouterr().out
    assert len(re.findall("No errors detected by derivative checker", captured)) == 1


if __name__ == "__main__":
    pytest.main([__file__])
