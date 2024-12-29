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
    solution = ocp.solve()
    print(solution.objective / J - 1)
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()
