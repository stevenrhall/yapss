# ruff: noqa: D100, D103
# standard library imports
import re

# third party imports
import pytest

#  package imports
from yapss import Problem


def optimal_control_problem():
    ocp = Problem(
        name="Test_Continuous_Derivatives",
        nx=[3, 3],
        nq=[1, 3],
        nh=[2, 2],
        nu=[2, 2],
        ns=2,
    )

    def continuous(arg):
        s = arg.parameter
        for p in arg.phase_list:
            i = 1 + p

            x = arg.phase[p].state
            u = arg.phase[p].control
            t = arg.phase[p].time
            variables = (x, u, s, [t])
            variables = [z for var in variables for z in var]
            # key_list = [("x", i) for i in range(3)]
            # key_list += [("u", i) for i in range(2)]
            # key_list += [("s", i) for i in range(2)]
            # key_list.append(("t", 0))

            arg.phase[p].dynamics[:] = 0, 0, 0
            if p == 0:
                arg.phase[p].integrand[:] = (0,)
            else:
                arg.phase[p].integrand[:] = 0, 0, 0

            arg.phase[p].path[:] = 0, 0

            for z1 in variables:
                for z2 in variables:
                    # for j in range(3):
                    #     arg.phase[p].dynamics[j] = (
                    #         arg.phase[p].dynamics[j] + (z1 + j) ** 2 * (z2 - i) ** 2
                    #     )
                    for j in range(2):
                        pass
                        arg.phase[p].path[j] = arg.phase[p].path[j] + (z1 + j) * (z2 - i)
                    nq = 1 if p == 0 else 3
                    for j in range(nq):
                        pass
                        # if j==1:
                        #     arg.phase[p].integrand[j] = (
                        #         arg.phase[p].integrand[j] + (z1 + j)* (z2 - i)
                        #     )
                        # else:
                        #     arg.phase[p].integrand[j] = (
                        #             arg.phase[p].integrand[j] + (z1 + j)
                        #     )
                    i += 0.1

    def objective(arg):
        arg.objective = 0

    ocp.functions.objective = objective
    ocp.functions.objective_gradient = lambda _: None
    ocp.functions.continuous = continuous
    ocp.functions.continuous_jacobian = lambda _: None
    ocp.functions.discrete = lambda _: None
    ocp.functions.discrete_jacobian = lambda _: None
    ocp.functions.discrete_hessian = lambda _: None

    for q in range(2):
        m, n = 1, 3
        ocp.mesh.phase[q].collocation_points = m * (n,)
        ocp.mesh.phase[q].fraction = m * (1.0 / m,)

    ocp.guess.phase[0].integral = (1,)
    ocp.guess.phase[1].integral = 1, 2, 3
    ocp.guess.parameter = 5, 4
    ocp.guess.phase[0].time = (0, 1)
    ocp.guess.phase[1].time = (1, 3)
    ocp.guess.phase[0].state = ((2, 1), (3, 1), (4, 1))
    ocp.guess.phase[1].state = ((4, 1), (5, 1), (6, 1))
    ocp.guess.phase[0].control = ((-1, 1), (-3, 1))
    ocp.guess.phase[1].control = ((1, 1), (3, 1))

    ocp.ipopt_options.max_iter = 0
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
    ocp = optimal_control_problem()
    ocp.derivatives.method = derivative_method
    ocp.derivatives.order = "first"
    ocp.ipopt_options.derivative_test = "first-order"
    ocp.spectral_method = spectral_method
    ocp.solve()
    captured = capfd.readouterr().out
    print(captured)
    assert len(re.findall("No errors detected by derivative checker", captured)) == 1


if __name__ == "__main__":
    pytest.main([__file__])
