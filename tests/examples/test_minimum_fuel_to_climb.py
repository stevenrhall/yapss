"""

Test that yapss.examples.minimum_time_to_climb (minimum fuel case) works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import minimum_time_to_climb as optimal_control_problem

J = -1177.67
tol = 1e-5

# parameters = [
#     (method, mode, order)
#     for mode in ("lg", "lgr", "lgl")
#     for method in ("central-difference",)
#     for order in ("second",)
# ]

parameters = [
    ("central-difference", "lg", "second"),
    pytest.param(
        "central-difference",
        "lgr",
        "second",
        marks=pytest.mark.xfail(reason="Known issue with minimum_fuel_to_climb example."),
    ),
    ("central-difference", "lgl", "second"),
]


@pytest.mark.parametrize(("method", "mode", "order"), parameters)
def test_optimal_control_problem(method: str, mode: str, order: str) -> None:
    """Test the minimum fuel to climb example."""
    ocp = optimal_control_problem.setup(min_fuel=True)
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)
    if method == "central-difference" and mode == "lgl" and order == "second":
        optimal_control_problem.plt.ion()
        optimal_control_problem.plot_solution(solution)
        optimal_control_problem.plot_density()
        optimal_control_problem.plot_speed_of_sound()
        optimal_control_problem.plot_lift_curve_slope()
        optimal_control_problem.plot_cd0()
        optimal_control_problem.plot_eta()
        optimal_control_problem.plt.show()
