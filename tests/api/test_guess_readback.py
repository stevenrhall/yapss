"""A guess reads back in the form its setter accepts, so writing it back changes nothing.

A pair reads back as the pair, a guess never set as ``(0.0, 0.0)``, a `yapss.interp` guess as
the object that was assigned, and a block field as a tuple of those, one per row. Which kind a
guess is, is carried by the value's own type; nothing about how YAPSS stores it shows.
"""

import numpy as np
import pytest

import yapss
from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot


class State(yapss.State):
    x = yapss.scalar()
    r = yapss.vector(3)


class Control(yapss.Control):
    u = yapss.scalar()
    w = yapss.vector(2)


class Integral(yapss.Integral):
    q = yapss.scalar()
    b = yapss.vector(2)


class Parameter(yapss.Parameter):
    s = yapss.scalar()
    t = yapss.vector(2)


class Shape(yapss.Phase):
    state: State
    control: Control
    integral: Integral


class Phases(yapss.Phases):
    phase: Shape


class Guessed(yapss.Problem):
    phases: Phases
    parameter: Parameter


T = np.array([0.0, 0.5, 1.0])
ONE_ROW = yapss.interp(T, [0.0, 1.0, 0.5])
THREE_ROWS = yapss.interp(T, [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])


def setups():
    """Every form of guess a setter accepts, each as a function writing it."""

    def unset(problem):
        pass

    def pairs(problem):
        ph = problem.phases.phase
        ph.state.x.guess = (1, 2)
        ph.control.u.guess = (0.5, 0.5)
        ph.state.r.guess[:] = (0.0, 1.0)
        ph.control.w.guess[:] = [(0.0, 1.0), (2.0, 3.0)]

    def one_row(problem):
        ph = problem.phases.phase
        ph.state.x.guess = ONE_ROW
        ph.state.r.guess[:] = ONE_ROW
        ph.control.w.guess[1] = ONE_ROW

    def every_row(problem):
        problem.phases.phase.state.r.guess[:] = THREE_ROWS

    def mixed(problem):
        ph = problem.phases.phase
        ph.state.r.guess[:] = [(0.0, 1.0), ONE_ROW, (2.0, 2.0)]

    def numbers(problem):
        ph = problem.phases.phase
        ph.integral.q.guess = 3.0
        ph.integral.b.guess[:] = [1.0, 2.0]
        problem.parameter.s.guess = 4.0
        problem.parameter.t.guess[0] = 5.0

    return [unset, pairs, one_row, every_row, mixed, numbers]


def build(write):
    problem = Guessed("guessed")
    problem.phases.phase.time.guess = (0.0, 1.0)
    write(problem)
    return problem


def settings(problem):
    """Every guess setting of `problem`, each with whether its field is a block."""
    ph = problem.phases.phase
    return [
        (ph.state.x, False),
        (ph.state.r, True),
        (ph.control.u, False),
        (ph.control.w, True),
        (ph.integral.q, False),
        (ph.integral.b, True),
        (problem.parameter.s, False),
        (problem.parameter.t, True),
    ]


def guesses(problem):
    return [tuple(field.guess) if block else field.guess for field, block in settings(problem)]


def test_an_unset_guess_reads_as_zero_at_both_ends():
    problem = build(lambda problem: None)
    ph = problem.phases.phase
    assert ph.state.x.guess == (0.0, 0.0)
    assert ph.state.r.guess == ((0.0, 0.0),) * 3
    assert ph.integral.q.guess == 0.0


def test_a_pair_reads_back_as_the_pair_of_floats():
    problem = build(lambda problem: None)
    ph = problem.phases.phase
    ph.state.x.guess = (1, 2)
    assert ph.state.x.guess == (1.0, 2.0)
    assert all(isinstance(side, float) for side in ph.state.x.guess)


def test_samples_read_back_as_the_object_assigned():
    problem = build(lambda problem: None)
    ph = problem.phases.phase
    ph.state.x.guess = ONE_ROW
    ph.state.r.guess[:] = THREE_ROWS
    assert ph.state.x.guess is ONE_ROW
    assert all(row is THREE_ROWS for row in ph.state.r.guess)


def test_a_block_field_reads_back_as_a_tuple_of_its_rows():
    problem = build(lambda problem: None)
    ph = problem.phases.phase
    ph.state.r.guess[:] = [(0.0, 1.0), ONE_ROW, (2.0, 2.0)]
    assert ph.state.r.guess == ((0.0, 1.0), ONE_ROW, (2.0, 2.0))


@pytest.mark.parametrize("write", setups(), ids=lambda write: write.__name__)
def test_writing_a_guess_back_changes_nothing(write):
    """``h.guess = h.guess`` for a scalar field, ``r.guess[:] = r.guess`` for a block."""
    problem = build(write)
    before = guesses(problem)
    spec = to_transcription_spec(snapshot_with_callbacks(problem))
    for field, block in settings(problem):
        if block:
            field.guess[:] = field.guess
        else:
            field.guess = field.guess
    assert guesses(problem) == before
    again = to_transcription_spec(snapshot_with_callbacks(problem))
    for name in ("guess_time", "guess_state", "guess_control", "guess_integral"):
        np.testing.assert_array_equal(getattr(again.phases[0], name), getattr(spec.phases[0], name))
    np.testing.assert_array_equal(again.guess_parameter, spec.guess_parameter)


def snapshot_with_callbacks(problem):
    """A snapshot needs an objective, and the guess it records needs nothing else."""
    problem.register.objective(lambda arg: 0.0)
    return snapshot(problem)
