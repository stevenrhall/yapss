"""What a solution holds, how it is read, and what it refuses.

A solution is a record of what was solved, read under the names the problem declared. It is
data: every quantity reads back, nothing about it can be assigned, and a name that was never
declared is refused rather than answered with something plausible.
"""

from __future__ import annotations

import numpy as np

from ._api import raises, solvable


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


def test_a_solutions_names_are_fixed() -> None:
    """Each name is one solved quantity, so it cannot be rebound; the refusal says what can be
    done instead, which is to edit the arrays in place."""
    _, result = solution()
    with raises(AttributeError, "names are fixed", "edited in place", at="result.objective"):
        result.objective = 1.0


def test_a_phase_solutions_names_are_fixed() -> None:
    """The same for a phase's record."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "names are fixed", "edited in place", at="ps.state"):
        ps.state = 1.0


def test_a_solutions_fields_are_fixed() -> None:
    """Down to the fields: a field names its array, and the array is edited, not replaced."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    with raises(AttributeError, "'x' cannot be assigned", "names are fixed", at="ps.state.x"):
        ps.state.x = np.zeros(3)


def _arrays(obj: object, path: str, found: dict[str, np.ndarray], depth: int = 0) -> None:
    """Collect every array reachable from `obj` by public names, keyed by its path."""
    if isinstance(obj, np.ndarray):
        found[path] = obj
        return
    if depth > 5:  # noqa: PLR2004 -- deeper than any solution tree
        return
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001, S112 -- a name that cannot be read is not walked
            continue
        if isinstance(value, np.ndarray) or type(value).__module__.startswith("yapss"):
            _arrays(value, f"{path}.{name}", found, depth + 1)


def test_a_solutions_arrays_are_the_users_and_share_nothing_else() -> None:
    """Spec 8: a solution's arrays are ordinary writable arrays, and two of them share storage
    only when they are the same quantity.

    Writable, because once the solve has returned nothing depends on these numbers; sharing
    nothing else, because a write for a plot must not change what a neighbouring quantity
    reports. The costate and the multiplier of the dynamics are one quantity under two names.
    """
    problem, result = solution()
    found: dict[str, np.ndarray] = {}
    _arrays(result, "solution", found)
    _arrays(result[problem.phases.slide], "ps", found)
    assert len(found) > 20
    assert all(array.flags.writeable for array in found.values())
    same_quantity = {("costate", "multiplier.dynamics")}
    items = list(found.items())
    shared = [
        (a, b)
        for i, (a, x) in enumerate(items)
        for b, y in items[i + 1 :]
        if x is not y
        and np.shares_memory(x, y)
        and not any(p in a and q in b or q in a and p in b for p, q in same_quantity)
    ]
    assert not shared, shared


def test_an_edit_to_a_solution_array_is_kept() -> None:
    """Spec 8: an array handed out on every read is the one the solution holds, not a copy, so
    an edit made through one read is what the next read returns -- it does not land in a
    temporary and do nothing."""
    problem, result = solution()
    ps = result[problem.phases.slide]
    ps.state.v[0] = -99.0
    assert ps.state.v[0] == -99.0


def test_the_0_3_0_name_nlp_info_points_to_nlp() -> None:
    """The name a script being ported reaches for is answered with the new one."""
    p = solvable()
    p.ipopt_options.print_level = 0
    solution = p.solve()
    with raises(AttributeError, "'nlp'", at="nlp_info"):
        _ = solution.nlp_info  # type: ignore[attr-defined]
