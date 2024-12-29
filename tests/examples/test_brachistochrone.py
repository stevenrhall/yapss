"""

Test that yapss.examples.brachistochrone works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import brachistochrone as optimal_control_problem

J = 0.312480130708665
Jw = 0.323331164025862

tol = 1e-8

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference", "user")
    for order in ("first", "second")
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the brachistochrone example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.ipopt_options.print_user_options = "yes"
    ocp.ipopt_options.tol = 1e-20
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plt.ion()
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem_wall(method: str, mode: str, order: str) -> None:
    """Test the brachistochrone example with a wall."""
    ocp = optimal_control_problem.setup(wall=True)
    ocp.ipopt_options.print_level = 3
    ocp.ipopt_options.print_user_options = "yes"
    ocp.ipopt_options.tol = 1e-20
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    solution = ocp.solve()
    assert solution.objective == pytest.approx(Jw, rel=tol)
    if method == "auto" and mode == "lgl":
        optimal_control_problem.plot_solution(solution, wall=True)
        optimal_control_problem.plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
