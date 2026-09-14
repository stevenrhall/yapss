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


def test_boundary_state_bounds_must_overlap_state_bounds():
    """Individually consistent state and boundary-state bounds must still intersect.

    The NLP bounds on the boundary states are the intersection of the state bounds with
    the initial (final) state bounds, so a disjoint pair would reach Ipopt as an
    infeasible lower > upper bound with no diagnostic.
    """
    ocp = goddard_problem_3_phase.setup()
    ocp.bounds.phase[0].state.lower = [5, 0, 0]
    ocp.bounds.phase[0].state.upper = [10, 1, 1]
    ocp.bounds.phase[0].initial_state.lower = [-1, 0, 0]
    ocp.bounds.phase[0].initial_state.upper = [0, 1, 1]
    msg = "bounds.phase[0].initial_state and bounds.phase[0].state do not overlap for indices i in [0]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()

    ocp.bounds.reset()
    ocp.bounds.phase[2].state.lower = [0, 0, 0]
    ocp.bounds.phase[2].state.upper = [1, 1, 1]
    ocp.bounds.phase[2].final_state.lower = [0, 2, 3]
    msg = "bounds.phase[2].final_state and bounds.phase[2].state do not overlap for indices i in [1 2]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()

    # touching at a single point is a valid (fixed) boundary state
    ocp.bounds.reset()
    ocp.bounds.phase[2].state.lower = [0, 0, 0]
    ocp.bounds.phase[2].final_state.lower = [0, 0, 0]
    ocp.bounds.phase[2].final_state.upper = [0, 0, 0]
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
    msg = "attribute 'upper' must be a float, not <class 'str'>"
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


def test_scalar_bounds_accept_numpy_scalars():
    """A scalar bound takes any real number, NumPy scalars included, as array bounds do.

    Through 0.2.2 only Python int and float were accepted (np.float64 by subclassing), so
    an element pulled from a float32 or integer array raised TypeError.
    """
    ocp = Problem(name="Test", nx=[1])
    bounds = ocp.bounds.phase[0]
    bounds.final_time.upper = np.float32(10.0)
    bounds.final_time.lower = np.int64(2)
    bounds.duration.upper = np.float64(8.0)
    assert bounds.final_time.upper == 10.0
    assert bounds.final_time.lower == 2.0
    assert isinstance(bounds.final_time.lower, float)
    with pytest.raises(TypeError, match="must be a float"):
        bounds.final_time.upper = True


def _bounds_problem() -> Problem:
    """A two-phase problem with every kind of array bound, all at their defaults."""
    return Problem(name="bounds", nx=[2, 1], nu=[1, 1], nq=[1, 1], nh=[1, 1], ns=2, nd=2)


_ARRAY_BOUNDS = [
    (lambda b: b.phase[0].state, "bounds.phase[0].state"),
    (lambda b: b.phase[1].control, "bounds.phase[1].control"),
    (lambda b: b.phase[0].initial_state, "bounds.phase[0].initial_state"),
    (lambda b: b.phase[1].path, "bounds.phase[1].path"),
    (lambda b: b.discrete, "bounds.discrete"),
    (lambda b: b.parameter, "bounds.parameter"),
]


@pytest.mark.parametrize(("get", "path"), _ARRAY_BOUNDS)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_nan_array_bound_is_rejected(get, path, side):
    """NaN passes every comparison, so it is checked first and named."""
    ocp = _bounds_problem()
    values = getattr(get(ocp.bounds), side)
    values[-1] = np.nan
    msg = f"{path}.{side}[i] is NaN for indices i in [{len(values) - 1}]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()


def test_none_in_a_bound_list_is_reported_as_nan():
    ocp = _bounds_problem()
    ocp.bounds.phase[0].state.upper = [10.0, None]
    msg = "bounds.phase[0].state.upper[i] is NaN for indices i in [1]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()


@pytest.mark.parametrize(("get", "path"), _ARRAY_BOUNDS)
def test_infinite_bound_on_the_wrong_side_is_rejected(get, path, recwarn):
    """Equal infinite bounds leave no feasible value and are not a crossing."""
    ocp = _bounds_problem()
    bound = get(ocp.bounds)
    bound.lower[0] = bound.upper[0] = np.inf
    msg = f"{path}.lower[i] is +inf for indices i in [0]; a lower bound must be less than +inf"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    bound.lower[0] = bound.upper[0] = -np.inf
    msg = f"{path}.upper[i] is -inf for indices i in [0]; an upper bound must be greater than -inf"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    # comparing directly rather than subtracting: no "invalid value in subtract" warning
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


@pytest.mark.parametrize("name", ["initial_time", "final_time", "duration"])
def test_scalar_bound_nan_and_wrong_side_infinity_are_rejected(name):
    ocp = _bounds_problem()
    scalar = getattr(ocp.bounds.phase[1], name)
    scalar.upper = float("nan")
    with pytest.raises(ValueError, match=re.escape(f"bounds.phase[1].{name}.upper is NaN")):
        ocp.bounds.validate()
    ocp.bounds.reset()
    scalar.upper = np.inf
    scalar.lower = np.inf
    msg = f"bounds.phase[1].{name}.lower is +inf; a lower bound must be less than +inf"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()


def test_negative_duration_lower_bound_is_rejected():
    ocp = _bounds_problem()
    ocp.bounds.phase[0].duration.lower = -5.0
    msg = "bounds.phase[0].duration.lower cannot be less than zero"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
    ocp.bounds.phase[0].duration.lower = 0.0
    ocp.bounds.validate()


def test_crossing_message_is_unchanged():
    ocp = _bounds_problem()
    ocp.bounds.phase[0].state.lower[:] = [1.0, 5.0]
    ocp.bounds.phase[0].state.upper[:] = [2.0, 3.0]
    msg = "bounds.phase[0].state.lower[i] is greater than bounds.phase[0].state.upper[i]"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.bounds.validate()
