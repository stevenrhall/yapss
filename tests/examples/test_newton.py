"""

Test that yapss.example.newton works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import newton as optimal_control_problem

J = 1.499263915138482
# solution is poor due to discontinuity in the solution, hence the large tolerance
tol = 1e-2

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference", "user")
    for order in ["second"]
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the Newton example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.tol = 1e-8
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode

    m, n = 50, 4
    ocp.mesh.phase[0].collocation_points = m * (n,)
    ocp.mesh.phase[0].fraction = m * (1.0 / m,)

    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
