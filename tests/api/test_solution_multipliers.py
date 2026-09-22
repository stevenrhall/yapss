"""A solution's dual layer: the multipliers, in the shapes of what they belong to.

Each is checked against the back end's own record of the same solve, which `solve_problem`
returns beside the solution, so these are identities rather than tolerances: the solution must
report what was computed, under the right name, and nothing else.
"""

import pickle

import numpy as np
import pytest

from yapss._api.compile import solve_problem
from yapss._api.spec import snapshot
from yapss.examples import goddard_problem_3_phase, hs071

from .test_solution_pickling import declared_in_a_function


@pytest.fixture(scope="module")
def goddard():
    problem = goddard_problem_3_phase.setup()
    problem.ipopt_options.print_level = 0
    solution, record = solve_problem(snapshot(problem))
    return problem, solution, record


@pytest.fixture(scope="module")
def local():
    """A problem with every kind of vector: control, path, integral, parameter, discrete."""
    problem = declared_in_a_function()
    solution, record = solve_problem(snapshot(problem))
    return problem, solution, record


def test_the_costate_is_the_multiplier_of_the_dynamics(goddard):
    """One array under two names, so an edit to one is an edit to the other."""
    _, solution, _ = goddard
    ps = solution["boost"]
    assert ps.costate is ps.multiplier.dynamics


def test_the_phase_multipliers_are_the_back_end_s(local):
    _, solution, record = local
    ps, data = solution["run"], record.phase[0]
    np.testing.assert_array_equal(ps.multiplier.dynamics[:], data.costate)
    np.testing.assert_array_equal(ps.multiplier.control.u, data.control_multiplier[0])
    np.testing.assert_array_equal(ps.multiplier.path.size, data.path_multiplier[0])
    assert ps.multiplier.integral.effort == data.integral_multiplier[0]
    assert ps.multiplier.duration == data.duration_multiplier


def test_the_time_multipliers_are_under_the_phase_s_own_name(local):
    """The phase runs over `s`, and its endpoint multipliers say `s`, as `ps.initial` does."""
    _, solution, record = local
    ps, data = solution["run"], record.phase[0]
    assert ps.multiplier.initial.s == data.initial_time_multiplier
    assert ps.multiplier.final.s == data.final_time_multiplier


def test_the_problem_multipliers_are_the_back_end_s(local):
    _, solution, record = local
    assert solution.multiplier.parameter.k == record.parameter_multiplier[0]
    assert solution.multiplier.discrete.end == record.discrete_multiplier[0]


def test_the_integrand_is_restored(local):
    _, solution, record = local
    np.testing.assert_array_equal(solution["run"].integrand.effort, record.phase[0].integrand[0])


def test_the_solution_names_its_problem_and_method(goddard):
    problem, solution, _ = goddard
    assert solution.name == problem.name
    assert solution.method == problem.method


def test_the_state_bound_multipliers_are_owed_and_say_so(goddard):
    """Reserved names, not unknown ones: the message says what is outstanding."""
    _, solution, _ = goddard
    ps = solution["boost"]
    with pytest.raises(AttributeError, match="state's bounds are not reported yet"):
        ps.multiplier.state  # noqa: B018
    with pytest.raises(AttributeError, match="multiplier of 'h' at this endpoint is not reported"):
        ps.multiplier.initial.h  # noqa: B018


def test_a_misspelled_multiplier_is_refused_with_a_suggestion(goddard):
    _, solution, _ = goddard
    ps = solution["boost"]
    with pytest.raises(AttributeError, match=r"no 'dynamic'\. Did you mean 'dynamics'\?"):
        ps.multiplier.dynamic  # noqa: B018
    with pytest.raises(AttributeError, match=r"no 'tim'\. Did you mean 'time'\?"):
        ps.multiplier.initial.tim  # noqa: B018


def test_the_old_suffixed_name_is_gone(goddard):
    """`solution.discrete_multiplier` became `solution.multiplier.discrete`."""
    _, solution, _ = goddard
    with pytest.raises(AttributeError, match="no 'discrete_multiplier'"):
        solution.discrete_multiplier  # noqa: B018
    assert np.isfinite(solution.multiplier.discrete.singular_coast_time)


def test_a_problem_without_phases_has_problem_multipliers_only():
    problem = hs071.setup()
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    assert solution.multiplier.discrete.product != 0.0


def test_the_multipliers_pickle_with_the_solution(local):
    _, solution, _ = local
    copy = pickle.loads(pickle.dumps(solution))
    ps, ps_copy = solution["run"], copy["run"]
    np.testing.assert_array_equal(ps_copy.multiplier.control.u, ps.multiplier.control.u)
    assert ps_copy.multiplier.final.s == ps.multiplier.final.s
    assert copy.multiplier.discrete.end == solution.multiplier.discrete.end
    assert ps_copy.costate is ps_copy.multiplier.dynamics
