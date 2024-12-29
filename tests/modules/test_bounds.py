"""

Test the yapss._private.bounds module.

"""

# ruff: noqa: D103 (missing docstring)
# standard library imports
import re

# third party imports
import numpy as np
import pytest

# package imports
from yapss import Problem
from yapss.examples import dynamic_soaring, goddard_problem_3_phase


def test_array_bound_class():
    """Test the ArrayBound class."""
    ocp = goddard_problem_3_phase.setup()
    # can't set dummy attribute
    with pytest.raises(AttributeError):
        ocp.bounds.parameter.lower.dummy = 1
    # private variable exists
    assert hasattr(ocp.bounds.parameter, "_lower")
    # bound attribute returns a numpy array
    assert isinstance(ocp.bounds.parameter.lower, np.ndarray)
    # can't delete an ArrayBounds attribute
    msg = "cannot delete 'ArrayBounds' attribute 'lower'"
    with pytest.raises(AttributeError, match=msg):
        del ocp.bounds.parameter.lower


def test_change_bounds_attribute():
    """Test that the bounds attribute cannot be changed."""
    ocp = goddard_problem_3_phase.setup()
    msg = "cannot set 'Problem' attribute 'bounds'"
    with pytest.raises(AttributeError, match=msg):
        ocp.bounds = 1
    msg = "cannot delete 'Problem' attribute 'bounds'"
    with pytest.raises(AttributeError, match=msg):
        del ocp.bounds


def test_reset_bounds():
    """Test that the bounds can be reset properly."""
    ocp = dynamic_soaring.setup()
    ocp.bounds.reset()
    assert np.all(ocp.bounds.phase[0].state.lower == -np.inf)
    assert np.all(ocp.bounds.phase[0].state.upper == np.inf)
    assert np.all(ocp.bounds.parameter.upper == np.inf)
    assert np.all(ocp.bounds.parameter.lower == -np.inf)
    assert np.all(ocp.bounds.discrete.lower == -np.inf)
    assert np.all(ocp.bounds.discrete.upper == np.inf)


def test_delete_bounds_phase_attribute():
    """Test that attributes cannot be deleted."""
    ocp = dynamic_soaring.setup()
    msg = "cannot delete 'Bounds' attribute 'parameter'"
    with pytest.raises(AttributeError, match=msg):
        del ocp.bounds.parameter
    msg = "cannot delete 'ArrayBounds' attribute 'lower'"
    with pytest.raises(AttributeError, match=msg):
        del ocp.bounds.phase[0].state.lower
    with pytest.raises(AttributeError, match="can't delete attribute"):
        del ocp.bounds.phase[0].initial_time.lower


def test_change_bounds_discrete_attribute():
    ocp = goddard_problem_3_phase.setup()
    with pytest.raises(AttributeError, match="cannot set 'Bounds' attribute 'discrete'"):
        ocp.bounds.discrete = 1


def test_change_bounds_parameter_attribute():
    ocp = goddard_problem_3_phase.setup()
    with pytest.raises(AttributeError, match="cannot set 'Bounds' attribute 'parameter'"):
        ocp.bounds.parameter = 1


def test_change_bounds_new_attribute():
    ocp = goddard_problem_3_phase.setup()
    with pytest.raises(AttributeError, match="cannot set 'Bounds' attribute 'new'"):
        ocp.bounds.new = 1


def test_change_bounds_shape():
    ocp = goddard_problem_3_phase.setup()
    with pytest.raises(ValueError, match="ArrayBound must be a sequence of floats of length 3."):
        ocp.bounds.phase[0].state.lower = [1, 2, 3, 4]
    ocp.bounds.phase[0].state.lower = [1, 2, 3]
    ocp.bounds.phase[0].state.upper = [-1, 2, -3]
    msg = (
        "bounds.phase[0].state.lower[i] is greater than bounds.phase[0].state.upper[i] "
        "for indices i in [0 2]"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    ocp = dynamic_soaring.setup()
    ocp.bounds.parameter.lower = [1]
    ocp.bounds.parameter.upper = [-1]
    msg = (
        "bounds.parameter.lower[i] is greater than bounds.parameter.upper[i] for "
        "indices i in [0]"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    ocp.bounds.phase[0].initial_time.lower = 1
    ocp.bounds.phase[0].initial_time.upper = -1
    msg = "bounds.phase[0].initial_time.lower is greater than bounds.phase[0].initial_time.upper"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()


def test_duration_bound_errors():
    """Test that errors are raised when duration bounds are infeasible."""
    # duration.lower > duration.upper
    ocp = goddard_problem_3_phase.setup()
    ocp.bounds.phase[0].duration.lower = 1
    ocp.bounds.phase[0].duration.upper = -1
    msg = "bounds.phase[0].duration.lower is greater than bounds.phase[0].duration.upper"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()

    # duration.upper < final_time.lower - initial_time.upper
    ocp.bounds.reset()
    ocp.bounds.phase[1].duration.upper = 10
    ocp.bounds.phase[1].initial_time.upper = 0
    ocp.bounds.phase[1].final_time.lower = 20
    msg = (
        "Time bounds are infeasible:\n"
        "bounds.phase[1].final_time.lower - bounds.phase[1].initial_time.upper > "
        "bounds.phase[1].duration.upper."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()

    # duration.lower > final_time.upper - initial_time.lower
    ocp.bounds.reset()
    ocp.bounds.phase[1].duration.lower = 10
    ocp.bounds.phase[1].initial_time.lower = 0
    ocp.bounds.phase[1].final_time.upper = 5
    msg = (
        "Time bounds are infeasible:\n"
        "bounds.phase[1].final_time.upper - bounds.phase[1].initial_time.lower < "
        "bounds.phase[1].duration.lower."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()

    # duration.upper < 0
    ocp.bounds.reset()
    ocp.bounds.phase[1].duration.upper = -1
    ocp.bounds.phase[1].duration.lower = -10
    msg = "bounds.phase[1].duration.upper cannot be less than zero"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    # duration is a float
    msg = "attribute 'upper must be a float, not <class 'str'>"
    with pytest.raises(TypeError, match=re.escape(msg)):
        ocp.bounds.phase[1].duration.upper = "string"


def test_duration_bounds():
    """Test that the duration bounds work.

    The objective will be to minimize a single state that is linear in time t, but with a
    minimum duration.
    """
    # instantiation
    problem = Problem(name="test", nx=[1])

    # functions
    def continuous(arg):
        arg.phase[0].dynamics[0] = 1

    def objective(arg):
        arg.objective = arg.phase[0].final_state[0] * factor

    problem.functions.continuous = continuous
    problem.functions.objective = objective

    # bounds
    problem.bounds.phase[0].duration.lower = 1
    problem.bounds.phase[0].duration.upper = 2
    problem.bounds.phase[0].initial_state.lower = [0.0]
    problem.bounds.phase[0].initial_state.upper = [0.0]
    problem.bounds.phase[0].initial_time.lower = 0.0
    problem.bounds.phase[0].initial_time.upper = 0.0
    problem.bounds.phase[0].final_time.lower = 0.1
    problem.bounds.phase[0].final_time.upper = 10.0

    # guess
    problem.guess.phase[0].time = [0.0, 5.0]
    problem.guess.phase[0].state = [[0.0, 0.0]]

    # mesh
    m, n = 1, 5
    problem.mesh.phase[0].collocation_points = m * [n]
    problem.mesh.phase[0].fraction = m * [1 / m]

    # derivatives
    problem.derivatives.method = "auto"
    problem.derivatives.order = "first"

    problem.ipopt_options.dual_inf_tol = 1e-8
    problem.ipopt_options.acceptable_dual_inf_tol = 1e-8

    # solve
    factor = 1.0
    solution = problem.solve()
    assert solution.objective == pytest.approx(1.0)
    factor = -1.0
    solution = problem.solve()
    assert solution.objective == pytest.approx(-2.0)
