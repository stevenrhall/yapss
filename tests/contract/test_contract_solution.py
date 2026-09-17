"""Contract: `Solution`.

What a user may rely on
    - Under every spectral method, `time` holds the state points and `time_c` the
      collocation points; `state` is (nx, len(time)); `control`, `dynamics`, `path`,
      `integrand`, `costate`, and the control and path multipliers are (n, len(time_c));
      `hamiltonian` is (len(time_c),); `integral`, `parameter`, `discrete`, and their
      multipliers are 1-D.
    - `initial_time`, `final_time`, and `duration` agree with `time`; `initial_state` and
      `final_state` are the first and last columns of `state`.
    - `objective` is a float; arrays are float64.
    - `nlp_info` holds the raw NLP result under Ipopt's names.
    - A `Solution`'s attributes cannot be rebound.
    - `problem.guess(solution)` warm-starts a problem with a different mesh or method.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ._contract import callback_problem, not_yet, raises

SPECTRAL_METHODS = ["lgr", "lg", "lgl"]


@pytest.fixture(scope="module", params=SPECTRAL_METHODS)
def solution(request):
    ocp = callback_problem()
    ocp.spectral_method = request.param
    return ocp.solve()


# ---------------------------------------------------------------- what a user may do


def test_arrays_have_the_documented_shapes(solution):
    phase = solution.phase[0]
    n_state_points, n_collocation = len(phase.time), len(phase.time_c)
    assert phase.state.shape == (3, n_state_points)
    for name, rows in [
        ("control", 1),
        ("dynamics", 3),
        ("path", 1),
        ("integrand", 1),
        ("costate", 3),
        ("control_multiplier", 1),
        ("path_multiplier", 1),
    ]:
        assert getattr(phase, name).shape == (rows, n_collocation), name
    assert phase.hamiltonian.shape == (n_collocation,)
    assert phase.integral.shape == phase.integral_multiplier.shape == (1,)
    assert solution.discrete.shape == solution.discrete_multiplier.shape == (1,)
    assert solution.parameter.shape == solution.parameter_multiplier.shape == (0,)


def test_state_points_contain_the_collocation_points(solution):
    phase = solution.phase[0]
    assert np.all(np.isin(phase.time_c, phase.time))
    assert np.all(np.diff(phase.time) >= 0)


def test_times_and_endpoint_states_agree_with_the_arrays(solution):
    phase = solution.phase[0]
    assert phase.initial_time == phase.time[0]
    assert phase.final_time == phase.time[-1]
    assert phase.duration == pytest.approx(phase.final_time - phase.initial_time)
    np.testing.assert_array_equal(phase.initial_state, phase.state[:, 0])
    np.testing.assert_array_equal(phase.final_state, phase.state[:, -1])


def test_types(solution):
    phase = solution.phase[0]
    assert type(solution.objective) is float
    assert type(phase.initial_time) is float
    for name in ("time", "time_c", "state", "control", "costate", "hamiltonian"):
        assert getattr(phase, name).dtype == np.float64, name


def test_nlp_info_holds_the_raw_result_under_ipopt_names(solution):
    fields = {field.name for field in dataclasses.fields(solution.nlp_info)}
    assert fields == {
        "ipopt_status",
        "ipopt_status_message",
        "obj_val",
        "x",
        "g",
        "mult_x_L",
        "mult_x_U",
        "mult_g",
    }


@pytest.mark.parametrize(
    ("owner", "name"),
    [(lambda s: s, "objective"), (lambda s: s.phase[0], "state")],
    ids=["Solution", "SolutionPhase"],
)
def test_attributes_cannot_be_rebound(solution, owner, name):
    with raises(AttributeError, name, at="setattr"):
        setattr(owner(solution), name, 1.0)


@pytest.mark.parametrize("target_method", SPECTRAL_METHODS)
def test_warm_start_across_mesh_and_method(solution, target_method):
    ocp = callback_problem()
    ocp.spectral_method = target_method
    ocp.mesh.phase[0].collocation_points = (6, 6)
    ocp.mesh.phase[0].fraction = (0.5, 0.5)
    ocp.guess(solution)
    assert ocp.solve().nlp_info.ipopt_status == 0


# ----------------------------------------------------------------- not yet met


@not_yet("redesign (E2 part 1)", "Solution arrays are read-only")
def test_solution_arrays_are_read_only():
    solution = callback_problem().solve()
    with raises(ValueError, at="state[0, 0] ="):
        solution.phase[0].state[0, 0] = 99.0


@not_yet("redesign (E7d)", "reprs use the public path yapss.Solution")
def test_repr_uses_the_public_path():
    solution = callback_problem().solve()
    assert repr(solution).startswith("<yapss.Solution")


@not_yet(
    "redesign (E3)", "every Solution quantity is given on `time` (control shape (nu, len(time)))"
)
@pytest.mark.parametrize("method", ["lgr", "lg"])  # under LGL the two grids already coincide
def test_control_is_on_the_state_points(method):
    ocp = callback_problem()
    ocp.spectral_method = method
    phase = ocp.solve().phase[0]
    assert phase.control.shape == (1, len(phase.time))


@not_yet(
    "redesign (E3)", "solution.phase[p].collocated indexes the collocation points within `time`"
)
def test_collocated_indices():
    phase = callback_problem().solve().phase[0]
    np.testing.assert_array_equal(phase.time[phase.collocated], phase.time_c)
