"""

Test that yapss.examples.orbit_raising works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import orbit_raising as optimal_control_problem

J = -1.5252777
tol = 3e-8

parameters = [
    (method, mode, order)
    for mode in ("lg", "lgr", "lgl")
    for method in ("auto", "central-difference")
    for order in ("first", "second")
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test that yapss.examples.orbit_raising gives the expected optimal cost."""
    ocp = optimal_control_problem.setup()
    # one case won't converge with default mu_strategy
    ocp.ipopt_options.mu_strategy = "adaptive"
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "auto" and mode == "lgr" and order == "second":
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
