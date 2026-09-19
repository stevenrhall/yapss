"""

Test the yapss._private.guess module.

"""

import re

# must run pytest before yapss modules are imported
import pytest

if __name__ == "__main__":
    pytest.main(["--cov=yapss", "--cov-report=html", __file__])

# standard library imports
import re
from types import SimpleNamespace

# third party imports
import numpy as np
from scipy.interpolate import interp1d

# package imports
from yapss._legacy import Problem


def test_guess_parameter():
    problem = Problem(name="Test", nx=[], ns=2)
    assert problem.guess.parameter.shape == (2,)
    assert np.array_equal(problem.guess.parameter, np.zeros(2))
    problem.guess.parameter = [-2.0, 2.0]
    assert np.array_equal(problem.guess.parameter, np.array([-2.0, 2.0], dtype=float))
    msg = "guess.parameter must have length 2"
    with pytest.raises(ValueError, match=msg):
        problem.guess.parameter = [-2.0, 2.0, 3.0]


def test_guess_parameter_element_assignment():
    problem = Problem(name="Test", nx=[], ns=2)
    problem.guess.parameter[0] = -2.0
    problem.guess.parameter[1] = 2.0
    assert np.array_equal(problem.guess.parameter, np.array([-2.0, 2.0]))


def test_guess_integral():
    problem = Problem(name="Test", nx=[2], nu=[2], nq=[3], nh=[1], nd=4)
    problem.guess.phase[0].integral = [0.0, 0.0, 0.0]
    assert np.array_equal(problem.guess.phase[0].integral, np.zeros(3))


def test_validate_init():
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 5], nh=[1, 6], nd=4)

    # test that validate() raises an error if the guess for the time vector is not set
    with pytest.raises(ValueError, match=re.escape("guess.phase[0].time has not been set.")):
        problem.guess.validate()
    problem.guess.phase[0].time = np.linspace(0, 1, num=10, dtype=float)
    with pytest.raises(ValueError, match=re.escape("guess.phase[1].time has not been set.")):
        problem.guess.validate()

    # set both time vectors and the state vector for phase 1
    problem.guess.phase[1].time = np.linspace(0, 1, num=5, dtype=float)
    problem.guess.phase[1].state = np.zeros([3, 5])
    problem.guess.validate()

    # if state or control vector is inconsistent with time vector, validate() should raise an error
    problem.guess.phase[1].state = np.zeros([3, 10])
    msg = re.escape("guess.phase[1].state must be a 2-dimensional array of shape (3, 5).")
    with pytest.raises(ValueError, match=msg):
        problem.guess.validate()
    problem.guess.phase[1].state = np.zeros([3, 5])

    # if control vector is inconsistent with time vector, validate() should raise an error
    problem.guess.phase[1].control = np.zeros([4, 4])
    with pytest.raises(
        ValueError,
        match=re.escape("guess.phase[1].control must be a 2-dimensional array of shape (4, 5)."),
    ):
        problem.guess.validate()
    problem.guess.phase[1].control = np.zeros([4, 5])

    # check that unititialized control and state guesses are set to arrays of zeros of the
    # correct shape
    assert np.array_equal(problem.guess.phase[0].control, np.zeros([2, 10], dtype=float))
    assert np.array_equal(problem.guess.phase[0].state, np.zeros([2, 10], dtype=float))


def test_control_assignment():
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 5], nh=[1, 6], nd=4)
    problem.guess.phase[1].state = np.zeros([3, 5])
    with pytest.raises(
        ValueError,
        match=re.escape("'guess.phase[0].control' must be a 2-dimensional array."),
    ):
        problem.guess.phase[0].control = np.ones([10], dtype=float)
    with pytest.raises(
        ValueError,
        match=re.escape("Expected 'control' in 'guess.phase[0]' to have 2 rows, but got 10."),
    ):
        problem.guess.phase[0].control = np.ones([10, 10], dtype=float)
    with pytest.raises(
        ValueError,
        match=re.escape("'guess.phase[0].state' must have at least 2 columns."),
    ):
        problem.guess.phase[0].state = np.ones([2, 1], dtype=float)


def test_parameter():
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 5], nh=[1, 6], nd=4, ns=3)
    assert np.array_equal(
        problem.guess.parameter,
        np.zeros(
            [
                3,
            ],
            dtype=float,
        ),
    )
    # test that we can assign a new parameter guess
    problem.guess.parameter = 1, 2, 3
    assert np.array_equal(problem.guess.parameter, np.array([1, 2, 3], dtype=float))
    # test that assigning a parameter guess of the wrong length raises an error
    with pytest.raises(
        ValueError,
        match=re.escape("guess.parameter must have length 3"),
    ):
        problem.guess.parameter = 1, 2, 3, 4
    # test that we can assign a new element of the parameter guess
    problem.guess.parameter[0] = 4
    assert np.array_equal(problem.guess.parameter, np.array([4, 2, 3], dtype=float))


def test_integral():
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 5], nh=[1, 6], nd=4, ns=3)
    assert np.array_equal(problem.guess.phase[0].integral, np.zeros([3], dtype=float))
    # test that we can assign a new integral guess
    problem.guess.phase[0].integral = 1, 2, 3
    assert np.array_equal(problem.guess.phase[0].integral, np.array([1, 2, 3], dtype=float))
    # test that assigning an integral guess of the wrong length raises an error
    with pytest.raises(
        ValueError,
        match=re.escape("guess.phase[0].integral must have length 3, got 4."),
    ):
        problem.guess.phase[0].integral = 1, 2, 3, 4
    # test that we can assign a new element of the integral guess
    problem.guess.phase[0].integral[0] = 4
    assert np.array_equal(problem.guess.phase[0].integral, np.array([4, 2, 3], dtype=float))


def test_invalid_data():
    """A string is the wrong type, named as such; NumPy used to raise ValueError here."""
    problem = Problem(name="Test", nx=[], ns=2)
    with pytest.raises(TypeError, match="guess.parameter must be a real number"):
        problem.guess.parameter = [-2.0, "invalid"]

    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 4], nh=[1, 2], nd=4)
    with pytest.raises(TypeError, match=re.escape("guess.phase[0].integral must be a real")):
        problem.guess.phase[0].integral = [0.0, 0.0, "invalid"]


def test_guess_time():
    """Test setting the time vector for each phase."""
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 4], nh=[1, 2], nd=4)
    assert problem.guess.phase[0].time is None
    with pytest.raises(TypeError, match=re.escape("guess.phase[0].time must be a real")):
        problem.guess.phase[0].time = "invalid"
    msg = re.escape(
        "Expected 'guess.phase[1].time' to be a strictly increasing, 1-dimensional array with "
        "at least 2 elements, received shape ()."
    )
    with pytest.raises(ValueError, match=msg):
        problem.guess.phase[1].time = 10.0
    msg = re.escape(
        "Expected 'guess.phase[1].time' to be a strictly increasing, 1-dimensional array with "
        "at least 2 elements, received shape (1,)."
    )
    with pytest.raises(ValueError, match=msg):
        problem.guess.phase[1].time = (10.0,)
    problem.guess.phase[1].time = 10, 20, 30
    assert np.array_equal(problem.guess.phase[1].time, np.array([10, 20, 30]))
    msg = re.escape(
        "Expected 'guess.phase[1].time' to be a strictly increasing, 1-dimensional array with "
        "at least 2 elements, but the values were not strictly increasing."
    )
    with pytest.raises(ValueError, match=msg):
        problem.guess.phase[1].time = 20, 10, 30


def _make_solution_phase(*, time, time_c, state, control, integral):
    """Build a duck-typed stand-in for a `SolutionPhase`.

    `Guess.from_solution` only reads `.time`, `.time_c`, `.state`, `.control`, and
    `.integral` off each phase (and `.parameter` off the solution itself), so a
    `SimpleNamespace` is enough -- no need to run the solver to exercise it.
    """
    return SimpleNamespace(
        time=np.asarray(time, dtype=float),
        time_c=np.asarray(time_c, dtype=float),
        state=np.asarray(state, dtype=float),
        control=np.asarray(control, dtype=float),
        integral=np.asarray(integral, dtype=float),
    )


def test_from_solution_identity_when_time_c_equals_time():
    """For lgl, time_c == time, so the resampled control should equal the input exactly."""
    problem = Problem(name="Test", nx=[2], nu=[1], nq=[1])
    time = np.linspace(0.0, 1.0, 5)
    state = np.vstack([time, time**2])
    control = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    phase = _make_solution_phase(
        time=time,
        time_c=time,
        state=state,
        control=control,
        integral=[0.5],
    )
    solution = SimpleNamespace(phase=[phase], parameter=np.zeros(0))

    problem.guess.from_solution(solution)
    problem.guess.validate()

    assert np.array_equal(problem.guess.phase[0].time, time)
    assert np.array_equal(problem.guess.phase[0].state, state)
    assert np.allclose(problem.guess.phase[0].control, control)
    assert np.array_equal(problem.guess.phase[0].integral, np.array([0.5]))


def test_from_solution_resamples_control_when_time_c_differs():
    """For lgr/lg, time_c != time (different length); control must land on time."""
    problem = Problem(name="Test", nx=[1], nu=[1], nq=[0])
    time = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    time_c = np.array([0.0, 0.5, 1.0])
    state = np.array([time**2])
    control = np.array([[10.0, 20.0, 30.0]])
    phase = _make_solution_phase(
        time=time,
        time_c=time_c,
        state=state,
        control=control,
        integral=[],
    )
    solution = SimpleNamespace(phase=[phase], parameter=np.zeros(0))

    # this used to raise, since 'control' has a different length than 'time'
    problem.guess.from_solution(solution)
    problem.guess.validate()

    guess_control = problem.guess.phase[0].control
    assert guess_control.shape == (1, len(time))

    # cross-check against an independent interp1d call rather than hard-coded values,
    # to catch axis/transposition mistakes rather than just shape mistakes
    expected = interp1d(time_c, control, axis=1, fill_value="extrapolate")(time)
    assert np.allclose(guess_control, expected)

    # the shared nodes (0.0, 0.5, 1.0) should match the original control exactly
    assert np.allclose(guess_control[0, [0, 2, 4]], control[0])


def test_from_solution_multi_phase_independent():
    """Each phase's control should be resampled onto its own time grid, not mixed up."""
    problem = Problem(name="Test", nx=[1, 2], nu=[1, 2], nq=[0, 1])

    time0 = np.array([0.0, 1.0, 2.0])
    time_c0 = np.array([0.0, 2.0])
    phase0 = _make_solution_phase(
        time=time0,
        time_c=time_c0,
        state=[time0],
        control=[[5.0, 15.0]],
        integral=[],
    )

    time1 = np.array([0.0, 0.5, 1.0, 1.5])
    time_c1 = np.array([0.0, 0.75, 1.5])
    phase1 = _make_solution_phase(
        time=time1,
        time_c=time_c1,
        state=[time1, time1**2],
        control=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        integral=[2.0],
    )

    solution = SimpleNamespace(phase=[phase0, phase1], parameter=np.zeros(0))

    problem.guess.from_solution(solution)
    problem.guess.validate()

    assert problem.guess.phase[0].control.shape == (1, len(time0))
    assert problem.guess.phase[1].control.shape == (2, len(time1))
    assert np.array_equal(problem.guess.phase[1].integral, np.array([2.0]))


def test_from_solution_zero_controls():
    """A phase with no controls (nu == 0) shouldn't blow up the interpolation."""
    problem = Problem(name="Test", nx=[1], nu=[0], nq=[0])
    time = np.array([0.0, 1.0, 2.0])
    time_c = np.array([0.0, 2.0])
    phase = _make_solution_phase(
        time=time,
        time_c=time_c,
        state=[time],
        control=np.zeros((0, 2)),
        integral=[],
    )
    solution = SimpleNamespace(phase=[phase], parameter=np.zeros(0))

    problem.guess.from_solution(solution)
    problem.guess.validate()

    assert problem.guess.phase[0].control.shape == (0, len(time))


@pytest.mark.parametrize("spectral_method", ["lg", "lgr", "lgl"])
def test_initial_guess_nlp_state_is_in_time_order(spectral_method):
    """The NLP state guess, read back the way solution.py reads it, is the interpolant.

    For LG the state layout is collocation points first, then segment-start and final
    values, so the write must go through the layout's ``time_order`` just as the read does.
    """
    from yapss._private.guess import make_initial_guess_nlp
    from yapss._private.layout import problem_layout
    from yapss._private.mesh import Mesh
    from yapss._private.structure import get_nlp_dv_structure

    problem = Problem(name="Test", nx=[1], nu=[0])
    problem.spectral_method = spectral_method
    problem.mesh.phase[0].collocation_points = (4, 3)
    problem.mesh.phase[0].fraction = (0.5, 0.5)
    problem.guess.phase[0].time = np.array([0.0, 1.0])
    problem.guess.phase[0].state = np.array([[0.0, 1.0]])
    problem.guess.validate()

    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(spectral_method)
    z0 = make_initial_guess_nlp(problem._to_spec(), mesh)
    dv = get_nlp_dv_structure(problem._to_spec(), np.float64)
    dv.z[:] = z0

    phase = dv.phase[0]
    x = phase.x[0][problem_layout(problem._to_spec())[0].time_order]
    t_x = (mesh.tau_x[0] + 1) / 2
    np.testing.assert_allclose(x, t_x, atol=1e-14)
    assert phase.x0[0] == 0.0
    assert phase.xf[0] == 1.0


def test_unset_guess_follows_the_time_array():
    """An unset state/control guess is zeros at the current time length, never stored.

    Through 0.2.2 validate() stored the zeros, so after one solve a change to the time
    array alone made the next validate() reject a state array the user never set.
    """
    problem = Problem(name="Test", nx=[2], nu=[1])
    phase = problem.guess.phase[0]
    with pytest.raises(ValueError, match=re.escape("cannot be read before guess.phase[0].time")):
        phase.state
    with pytest.raises(ValueError, match="time is set"):
        phase.control[0, :] = 1.0

    phase.time = [0.0, 1.0]
    assert np.array_equal(phase.state, np.zeros((2, 2)))
    problem.guess.validate()

    # refine only the time grid: the default must follow it
    phase.time = np.linspace(0.0, 1.0, 5)
    problem.guess.validate()
    assert np.array_equal(phase.state, np.zeros((2, 5)))
    assert np.array_equal(phase.control, np.zeros((1, 5)))

    # a guess with values in it is kept and checked against the new time array
    phase.state = np.ones((2, 5))
    phase.time = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match=re.escape("shape (2, 3)")):
        problem.guess.validate()


def test_guess_copies_what_it_is_set_from():
    """The guess owns its arrays: editing it in place touches neither a Solution nor a user array.

    Through 0.2.2 the setters used np.asarray, which returns the caller's array unchanged
    when the dtype already matches, so problem.guess(solution) aliased the solution.
    """
    problem = Problem(name="Test", nx=[1], nu=[1], nq=[1], ns=1)
    time = np.array([0.0, 1.0, 2.0])
    state = np.array([[0.0, 1.0, 2.0]])
    control = np.array([[1.0, 1.0, 1.0]])
    phase = _make_solution_phase(
        time=time, time_c=time, state=state, control=control, integral=[0.5]
    )
    parameter = np.array([3.0])
    solution = SimpleNamespace(phase=[phase], parameter=parameter)

    problem.guess.from_solution(solution)
    guess_phase = problem.guess.phase[0]
    assert not np.shares_memory(guess_phase.state, state)
    assert not np.shares_memory(guess_phase.control, control)
    assert not np.shares_memory(guess_phase.time, time)
    assert not np.shares_memory(problem.guess.parameter, parameter)

    guess_phase.state[0, 0] = 99.0
    guess_phase.time[0] = -1.0
    problem.guess.parameter[0] = 99.0
    assert state[0, 0] == 0.0 and time[0] == 0.0 and parameter[0] == 3.0

    # and the other direction: a user array edited after assignment
    user_state = np.zeros((1, 3))
    guess_phase.state = user_state
    user_state[0, 1] = 5.0
    assert guess_phase.state[0, 1] == 0.0


def test_slice_assignment_into_the_default_guess_sticks():
    """Indexing and slicing assign into the stored default, so a guess can be built up."""
    problem = Problem(name="Test", nx=[2], nu=[1])
    phase = problem.guess.phase[0]
    phase.time = np.linspace(0.0, 1.0, 4)

    phase.state[0, :] = [1.0, 2.0, 3.0, 4.0]  # slice
    phase.state[1][2] = 9.0  # a view of a row, then an element
    phase.control += 0.5  # in-place operator on the whole array
    np.testing.assert_array_equal(phase.state, [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 9.0, 0.0]])
    np.testing.assert_array_equal(phase.control, [[0.5, 0.5, 0.5, 0.5]])
    problem.guess.validate()

    # a written guess is kept across a time change and validate() reports the mismatch
    phase.time = [0.0, 1.0]
    assert phase.state.shape == (2, 4)
    with pytest.raises(ValueError, match=re.escape("shape (2, 2)")):
        problem.guess.validate()
    # an all-zero guess, assigned or default, follows the new length
    phase.state = np.zeros((2, 4))
    phase.control = np.zeros((1, 4))
    phase.time = [0.0, 0.5, 1.0]
    assert phase.state.shape == (2, 3) and phase.control.shape == (1, 3)
    problem.guess.validate()


def test_guess_rejects_misspelled_attribute():
    """A misspelled attribute raises instead of being stored and silently ignored."""
    problem = Problem(name="Test", nx=[1], ns=2)
    with pytest.raises(AttributeError, match="cannot set 'Guess' attribute 'parmeter'"):
        problem.guess.parmeter = [-2.0, 2.0]
    with pytest.raises(AttributeError, match="cannot set 'Guess' attribute 'phase'"):
        problem.guess.phase = ()
    # the real attribute is unaffected
    problem.guess.parameter = [-2.0, 2.0]
    assert np.array_equal(problem.guess.parameter, [-2.0, 2.0])


def test_protected_guess_survives_deepcopy():
    """`Solution` deep-copies the problem; the copy keeps its own guess values."""
    from copy import deepcopy

    problem = Problem(name="Test", nx=[1], ns=2)
    problem.guess.parameter = [1.0, 2.0]
    copy = deepcopy(problem)
    copy.guess.parameter = [3.0, 4.0]
    assert np.array_equal(problem.guess.parameter, [1.0, 2.0])
    assert np.array_equal(copy.guess.parameter, [3.0, 4.0])
    with pytest.raises(AttributeError, match="cannot set 'Guess' attribute"):
        copy.guess.parmeter = [0.0, 0.0]
