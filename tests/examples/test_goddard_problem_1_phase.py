"""

Test that yapss.examples.goddard_problem_1_phase works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import goddard_problem_1_phase as optimal_control_problem

J = 18550.8718638
tol = 3e-3

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference", "user")
    for order in ("second",)
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the goddard_problem_1_phase example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()
