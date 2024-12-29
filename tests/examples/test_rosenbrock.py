"""

Test that yapss.example.rosenbrock works properly.

"""

import pytest

# third party imports
from matplotlib import pyplot as plt

# package imports
from yapss.examples import rosenbrock as optimal_control_problem

J = 0
tol = 1e-10

parameters = [
    (method, mode, order)
    for method in ("auto", "central-difference")
    for mode in ("lgr", "lgl")
    for order in ("first", "second")
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the Rosenbrock example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, abs=tol)
    if method == "auto" and mode == "lgl" and order == "second":
        plt.ion()
        optimal_control_problem.plot_rosenbrock()
        plt.show()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
