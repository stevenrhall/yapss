"""

Test that yapss.example.newton works properly for the alternate setup.

"""

# third party imports
import pytest

# package imports
from yapss.examples import newton as optimal_control_problem

J = 1.499263915138482
tol = 1e-7

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference", "user")
    for order in ["second"]
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the alternate Newton example."""
    ocp = optimal_control_problem.setup2()
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "first":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
