"""

Test that yapss.examples.delta_iii_ascent works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import delta_iii_ascent as optimal_control_problem

J = 7529.712287
tol = 1e-8

parameters = [
    ("auto", "lg", "second"),
    ("auto", "lgr", "second"),
    ("auto", "lgl", "second"),
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the Delta III ascent example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 5
    ocp.ipopt_options.print_user_options = "yes"
    ocp.ipopt_options.mu_strategy = "adaptive"
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
