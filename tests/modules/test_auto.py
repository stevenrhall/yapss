"""

Test the yapss._private.auto module.

"""

# third party imports
import numpy as np
import pytest

# package imports
from yapss._legacy import Problem


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


def test_auto_leaves_auxdata_alone():
    """The trace writes nothing into the user's namespace, so a re-solve sees it intact.

    Through 0.2.2 the objective trace stored an SXW and a casadi Function in
    ``auxdata.objective_out`` and ``auxdata.objective_function``; a user helper of the
    latter name was replaced after the first solve, and the second solve then called the
    casadi Function with the user's arguments.
    """
    problem = Problem(name="auxdata", nx=[], ns=1)
    problem.auxdata.objective_function = lambda s: (s - 3.0) ** 2
    problem.auxdata.marker = "mine"

    def objective(arg):
        arg.objective = arg.auxdata.objective_function(arg.parameter[0])

    problem.functions.objective = objective
    problem.derivatives.method = "auto"
    problem.guess.parameter = [0.0]
    problem.ipopt_options.print_level = 0

    first = problem.solve()
    assert vars(problem.auxdata).keys() == {"objective_function", "marker"}
    second = problem.solve()
    np.testing.assert_allclose(first.parameter, [3.0], atol=1e-6)
    np.testing.assert_allclose(second.parameter, [3.0], atol=1e-6)
