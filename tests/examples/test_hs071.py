"""

Test that yapss.example.hs071 works properly.

"""

# third party imports
import pytest

# package imports
from yapss.examples import hs071 as optimal_control_problem

J = 17.014017140224176
tol = 1e-8

parameters = [
    (method, order) for method in ("auto", "central-difference") for order in ("first", "second")
]


@pytest.mark.parametrize(("method", "order"), parameters)
def test_optimal_control_problem(method: str, order: str) -> None:
    """Test the hs071 example."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.derivatives.method = method
    ocp.derivatives.order = order
    ocp.ipopt_options.linear_solver = "mumps"
    solution = ocp.solve()
    optimal_control_problem.print_solution(solution)
    assert solution.objective == pytest.approx(J, rel=tol)


if __name__ == "__main__":
    pytest.main([__file__])
