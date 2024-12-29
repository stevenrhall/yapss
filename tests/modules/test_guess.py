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

# third party imports
import numpy as np

# package imports
from yapss import Problem


def test_guess_parameter():
    problem = Problem(name="Test", nx=[], ns=2)
    assert problem.guess.parameter.shape == (2,)
    assert np.array_equal(problem.guess.parameter, np.zeros(2))
    problem.guess.parameter = [-2.0, 2.0]
    assert np.array_equal(problem.guess.parameter, np.array([-2.0, 2.0], dtype=float))
    msg = "'guess.parameter' must be a 1-dimensional array of length 2."
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
        match="'guess.parameter' must be a 1-dimensional array of length 3.",
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
        match=re.escape("could not broadcast input array from shape (4,) into shape (3,)"),
    ):
        problem.guess.phase[0].integral = 1, 2, 3, 4
    # test that we can assign a new element of the integral guess
    problem.guess.phase[0].integral[0] = 4
    assert np.array_equal(problem.guess.phase[0].integral, np.array([4, 2, 3], dtype=float))


def test_invalid_data():
    problem = Problem(name="Test", nx=[], ns=2)
    with pytest.raises(ValueError, match="could not convert string to float: 'invalid'"):
        problem.guess.parameter = [-2.0, "invalid"]

    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 4], nh=[1, 2], nd=4)
    with pytest.raises(ValueError, match="could not convert string to float: 'invalid'"):
        problem.guess.phase[0].integral = [0.0, 0.0, "invalid"]


def test_guess_time():
    """Test setting the time vector for each phase."""
    problem = Problem(name="Test", nx=[2, 3], nu=[2, 4], nq=[3, 4], nh=[1, 2], nd=4)
    assert problem.guess.phase[0].time is None
    with pytest.raises(ValueError, match="could not convert string to float: 'invalid'"):
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
