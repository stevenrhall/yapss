"""Pickling and copying a solution.

A solution is data, except for its `problem`, which holds the user's callbacks. It is pickled
with the problem when the problem can be pickled, and without it otherwise; either way every
solved quantity survives. Copying never goes through pickling, so a copy keeps its problem.
"""

import copy
import pickle

import numpy as np
import pytest

import yapss
from yapss.examples.goddard_problem_3_phase import setup as goddard
from yapss.math import cos, sin


def objective(arg):
    arg.objective = arg.phase[0].final_time


def continuous(arg):
    _, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics = [v * cos(u), v * sin(u), 32.174 * sin(u)]


def brachistochrone():
    """Return a problem whose callbacks are module-level functions, so that it pickles."""
    problem = yapss.Problem(name="brachistochrone", nx=[3], nu=[1])
    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0
    bounds.initial_state.lower = bounds.initial_state.upper = [0, 0, 0]
    bounds.final_state.lower[0] = bounds.final_state.upper[0] = 1
    problem.guess.phase[0].time = [0, 1]
    problem.guess.phase[0].state = [[0, 1], [0, 1], [0, 5]]
    problem.guess.phase[0].control = [[0, 0]]
    problem.ipopt_options.print_level = 0
    return problem


@pytest.fixture(scope="module")
def picklable():
    return brachistochrone().solve()


@pytest.fixture(scope="module")
def unpicklable():
    """A solution whose problem has callbacks defined inside `setup`, as the examples do."""
    problem = goddard()
    problem.ipopt_options.print_level = 0
    return problem.solve()


def same_data(a, b):
    assert a.name == b.name
    assert a.objective == b.objective
    assert a.status == b.status
    assert a.converged == b.converged
    np.testing.assert_array_equal(a.parameter, b.parameter)
    np.testing.assert_array_equal(a.nlp_info.x, b.nlp_info.x)
    assert type(a.phase) is type(b.phase)
    assert len(a.phase) == len(b.phase)
    for pa, pb in zip(a.phase, b.phase, strict=True):
        for name in ("time", "time_c", "state", "control", "costate", "hamiltonian"):
            np.testing.assert_array_equal(getattr(pa, name), getattr(pb, name))


def test_a_solution_pickles_with_a_problem_that_pickles(picklable):
    loaded = pickle.loads(pickle.dumps(picklable))
    same_data(loaded, picklable)
    assert isinstance(loaded.problem, yapss.Problem)
    assert loaded.problem.functions.continuous is continuous


def test_a_solution_pickles_without_a_problem_that_does_not(unpicklable):
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(unpicklable.problem)
    loaded = pickle.loads(pickle.dumps(unpicklable))
    same_data(loaded, unpicklable)


def test_reading_the_dropped_problem_says_why(unpicklable):
    loaded = pickle.loads(pickle.dumps(unpicklable))
    with pytest.raises(AttributeError, match="pickled without its problem") as info:
        _ = loaded.problem
    assert "top level of a module" in str(info.value)
    assert "<locals>" in str(info.value)  # Python's own reason, naming the local function
    assert not hasattr(loaded, "problem")


def test_a_lambda_in_auxdata_drops_the_problem_too():
    problem = brachistochrone()
    problem.auxdata.table = lambda h: h
    loaded = pickle.loads(pickle.dumps(problem.solve()))
    with pytest.raises(AttributeError, match="pickled without its problem"):
        _ = loaded.problem


def test_pickling_leaves_the_solution_itself_unchanged(unpicklable):
    pickle.dumps(unpicklable)
    assert isinstance(unpicklable.problem, yapss.Problem)


def test_a_solution_loaded_without_its_problem_pickles_again(unpicklable):
    once = pickle.loads(pickle.dumps(unpicklable))
    twice = pickle.loads(pickle.dumps(once))
    same_data(twice, unpicklable)
    with pytest.raises(AttributeError, match="pickled without its problem"):
        _ = twice.problem


def test_another_missing_name_is_an_ordinary_attribute_error(unpicklable):
    loaded = pickle.loads(pickle.dumps(unpicklable))
    for solution in (unpicklable, loaded):
        with pytest.raises(AttributeError, match="'Solution' object has no attribute 'objectiv'"):
            _ = solution.objectiv


def test_a_deep_copy_keeps_the_problem_and_shares_nothing(unpicklable):
    duplicate = copy.deepcopy(unpicklable)
    same_data(duplicate, unpicklable)
    assert isinstance(duplicate.problem, yapss.Problem)
    assert duplicate.problem is not unpicklable.problem
    assert duplicate.phase[0].state is not unpicklable.phase[0].state


def test_a_shallow_copy_shares_the_problem_and_the_arrays(unpicklable):
    duplicate = copy.copy(unpicklable)
    assert duplicate is not unpicklable
    assert duplicate.problem is unpicklable.problem
    assert duplicate.phase is unpicklable.phase


def test_the_phases_of_a_copy_or_a_pickle_are_the_phases(unpicklable):
    """A copied or pickled tuple of phases used to come back as one tuple inside another."""
    for phases in (
        copy.copy(unpicklable.phase),
        copy.deepcopy(unpicklable.phase),
        pickle.loads(pickle.dumps(unpicklable.phase)),
    ):
        assert type(phases) is type(unpicklable.phase)
        assert len(phases) == 3
        assert [phase.index for phase in phases] == [0, 1, 2]
