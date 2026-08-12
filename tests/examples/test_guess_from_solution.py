"""

Test that Guess.from_solution produces a usable warm-start guess for every spectral method.

This exercises the fix for the bug where `from_solution` assigned a solution's `control`
array (defined on `time_c`) directly onto the guess's `time` grid, which only happens to
have the same length as `time_c` for the lgl spectral method.

"""

# third party imports
import pytest

# package imports
from yapss.examples import brachistochrone_minimal as optimal_control_problem

J = 0.312480130
tol = 1e-8


@pytest.mark.parametrize("mode", ["lgl", "lgr", "lg"])
def test_from_solution_warm_start(mode: str) -> None:
    """Solve once, build a guess from the solution, and re-solve as a warm start."""
    ocp = optimal_control_problem.setup()
    ocp.ipopt_options.print_level = 3
    ocp.spectral_method = mode
    ocp.ipopt_options.linear_solver = "mumps"

    solution = ocp.solve()
    assert solution.objective == pytest.approx(J, rel=tol)

    # this used to raise for mode in ("lgr", "lg"), since control.shape[1] != len(time)
    ocp.guess.from_solution(solution)
    ocp.guess.validate()

    warm_solution = ocp.solve()
    assert warm_solution.objective == pytest.approx(J, rel=tol)
