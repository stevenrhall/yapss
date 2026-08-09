"""

Test that yapss.example.isoperimetric works properly.

"""

# standard library imports
from math import pi

# third party imports
import pytest

# package imports
from yapss.examples import isoperimetric as optimal_control_problem

J = 1 / 4 / pi
tol = 1e-6

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference")
    for order in ("first", "second")
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the isoperimetric example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    # The example asks for 1e-14 to demonstrate how accurate the method is, which is
    # only reachable with automatic differentiation, second derivatives and LGL --
    # what `main()` runs. This sweep also covers first-order Hessian approximations
    # and central differences, where a KKT residual of 1e-14 cannot be reached at all,
    # so the example's value is replaced here rather than inherited.
    ocp.ipopt_options.tol = 1e-8
    solution = ocp.solve()
    print(solution.objective / J - 1)
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()
