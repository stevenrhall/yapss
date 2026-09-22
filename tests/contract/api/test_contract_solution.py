"""What a solution holds, how it is read, and what it refuses.

A solution is a record of what was solved, read under the names the problem declared. It is
data: every quantity reads back, nothing about it can be assigned, and a name that was never
declared is refused rather than answered with something plausible.
"""

from __future__ import annotations

import numpy as np

from ._api import not_yet, raises, solvable


def solution():
    """Return a solved problem and its solution."""
    problem = solvable()
    return problem, problem.solve()


# ---------------------------------------------------------------------- reading by name


def test_a_phase_is_reached_by_its_handle() -> None:
    """The same rule as a callback's `arg[ph]`, so one way of naming a phase serves both."""
    problem, result = solution()
    assert result[problem.phases.slide] is not None


def test_a_phase_is_also_reached_by_its_name() -> None:
    """A solution is data, readable with no problem at hand -- unpickled in a worker, say."""
    problem, result = solution()
    assert result["slide"] is result[problem.phases.slide]


def test_a_misspelled_phase_name_is_refused_with_a_suggestion() -> None:
    """The name is checked against the phases the problem declared."""
    _, result = solution()
    with raises(KeyError, "has no phase 'slid'", "Did you mean 'slide'?", at="result["):
        result["slid"]


def test_anything_else_is_refused_naming_both_forms() -> None:
    """Neither a handle nor a name: the message shows the two forms that work."""
    _, result = solution()
    with raises(KeyError, "takes a phase handle", "or a phase's name", at="result["):
        result[0]


def test_every_declared_quantity_reads_back_by_name() -> None:
    """The state, the control, the path, the integral and the costate, under the user's names."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    assert np.asarray(ps.state.x).shape == np.asarray(ps.time).shape
    assert np.asarray(ps.control.theta).size > 0
    assert np.asarray(ps.path.speed).size > 0
    assert np.asarray(ps.costate.v).size > 0
    assert float(ps.integral.effort) >= 0.0


def test_the_ends_of_a_phase_are_reported_as_a_callback_sees_them() -> None:
    """`initial` and `final` hold the state there and the independent variable there."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    assert ps.initial.time == 0.0
    assert abs(ps.final.x - 1.0) < 1e-6
    assert ps.duration == ps.final.time - ps.initial.time


def test_the_problem_level_values_read_back_too() -> None:
    """The objective, the status, and the discrete constraints with their multipliers."""
    _, result = solution()
    assert result.converged
    assert result.status is not None
    assert abs(result.discrete.drop - 0.5) < 1e-6
    assert result.multiplier.discrete.drop is not None


# --------------------------------------------------------------------------- what is refused


def test_a_solution_has_no_name_that_was_not_declared() -> None:
    """Checked against what the record holds, with a suggestion."""
    _, result = solution()
    with raises(AttributeError, "the solution has no 'nope'", at="result.nope"):
        _ = result.nope


def test_a_phase_solution_has_no_name_that_was_not_declared() -> None:
    """The same for a phase's own record."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "the phase solution has no 'nope'", at="ps.nope"):
        _ = ps.nope


def test_a_misspelled_field_names_the_vector() -> None:
    """Inside a vector, the message says which vector of which phase was asked."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "phase 'slide' solution state has no field 'nope'", at="state."):
        _ = ps.state.nope


def test_a_solution_is_read_only() -> None:
    """A record of what was solved cannot be edited into a record of something else."""
    _, result = solution()
    with raises(AttributeError, "a solution is read-only", at="result.objective"):
        result.objective = 1.0


def test_a_phase_solution_is_read_only() -> None:
    """The same for a phase's record."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "a solution is read-only", at="ps.state"):
        ps.state = 1.0


def test_the_arrays_of_a_solution_are_read_only() -> None:
    """Down to the rows: the record is data, not a scratch space."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "is read-only", at="ps.state.x"):
        ps.state.x = np.zeros(3)


@not_yet("gap", "a solution's arrays are writeable, so the record can be edited in place")
def test_a_solution_cannot_be_edited_in_place() -> None:
    """Protecting the attribute is not enough while the array it returns is writeable.

    `ps.state.v = x` is refused, but `ps.state.v[:] = -99.0` succeeds and changes the record:
    afterwards the solution reports a trajectory nobody solved for, with no sign that anything
    happened. The rows are built as `ReadOnlyRows` -- "rows the user reads but never writes" --
    and the ndarray beneath them has `writeable` set. `_backend/spec.py::frozen_array` is the
    technique the rest of the codebase uses for this.
    """
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(ValueError, "read-only", at="ps.state.v[:]"):
        ps.state.v[:] = -99.0
