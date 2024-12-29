"""

Test the yapss._private.auto module.

"""

# third party imports
import pytest

# package imports
from yapss import Problem


def test_no_hessian():
    """Test edge case where there are no Hessian terms."""
    problem = Problem(name="test", nx=[], ns=1)

    def objective(arg):
        arg.objective = arg.parameter[0]

    problem.functions.objective = objective
    problem.bounds.parameter.lower = [-1]
    problem.bounds.parameter.upper = [1]
    problem.derivatives.method = "auto"
    solution = problem.solve()
    assert pytest.approx(solution.objective) == -1


def test_no_dynamics():
    """Test edge case where there is a phase with no dynamics."""
    problem = Problem(name="test", nx=[0], nu=[1], nq=[1])

    def objective(arg):
        arg.objective = arg.phase[0].integral[0]

    def continuous(arg):
        (u,) = arg.phase[0].control
        t = arg.phase[0].time
        arg.phase[0].integrand[0] = t + (u - t) ** 2

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    problem.guess.phase[0].time = [0.0, 1.0]
    problem.derivatives.method = "auto"
    solution = problem.solve()
    assert pytest.approx(solution.objective) == 0.5
