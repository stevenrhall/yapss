"""End-to-end solves of the Delta III ascent problem through the redesigned API.

This is the problem that decided two questions left open by the lean prototype: whether
discrete constraints need nesting (they do not, provided a group is split where its scale
changes), and whether a block field needs per-row bounds (it does -- the launch state fixes
each component of the position separately).
"""

import numpy as np
import pytest

from yapss.examples.delta_iii_ascent import (
    Constraints,
    Omega_f,
    Vehicle,
    a_f,
    e_f,
    i_f,
    length_scale,
    omega_f,
    setup,
    velocity_scale,
)

FINAL_MASS = 7529.712268
"""What the same problem gives through the released API."""


@pytest.fixture(scope="module")
def solution():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem, problem.solve()


def test_it_solves_to_the_released_answer(solution):
    _, result = solution
    assert result.converged
    assert result.objective == pytest.approx(FINAL_MASS, rel=1e-8)


def test_it_agrees_with_the_same_problem_in_the_released_api(solution):
    from yapss._legacy.examples.delta_iii_ascent import setup as legacy_setup

    legacy = legacy_setup()
    legacy.ipopt_options.print_level = 0
    _, result = solution
    assert result.objective == pytest.approx(legacy.solve().objective, rel=1e-8)


def test_the_target_orbit_is_reached_and_reached_by_name(solution):
    """Splitting the group at its scale boundaries gave every row a name."""
    _, result = solution
    assert result.discrete.semi_major_axis == pytest.approx(a_f, rel=1e-6)
    assert result.discrete.eccentricity == pytest.approx(e_f, rel=1e-6)
    assert result.discrete.inclination == pytest.approx(i_f, rel=1e-6)
    assert result.discrete.raan == pytest.approx(Omega_f, rel=1e-6)
    assert result.discrete.argument_of_perigee == pytest.approx(omega_f, rel=1e-6)


def test_the_stages_are_joined(solution):
    _, result = solution
    for index in range(3):
        for what in ("position", "velocity"):
            group = getattr(result.discrete, f"stage_{index}_{index + 1}_{what}")
            assert group.shape == (3,)
            assert np.abs(group).max() == pytest.approx(0.0, abs=1e-5)


def test_block_fields_are_read_as_arrays(solution):
    problem, result = solution
    phase = result[problem.phases.stage_3]
    assert phase.state.r.shape == (3, len(phase.time))
    assert phase.control.u.shape == (3, len(phase.time))
    assert len(phase.state) == 7
    assert Vehicle._nrows == 7
    assert Constraints._nrows == 23


def test_the_launch_state_is_fixed_component_by_component():
    """A list gives a block field one bound per row, which the launch position needs."""
    problem = setup()
    bounds = problem.phases.stage_0.state.initial
    lower = [pair[0] for pair in bounds._elements("r")]
    upper = [pair[1] for pair in bounds._elements("r")]
    assert lower == upper
    assert len({round(value, 6) for value in lower}) == 3


def test_the_scales_reach_the_transcription():
    """Scale is a per-field aspect, including the separate scale of the defect rows."""
    from yapss._api.compile import to_transcription_spec
    from yapss._api.spec import snapshot

    spec = to_transcription_spec(snapshot(setup()))
    phase = spec.phases[0]
    assert list(phase.state_scale[:3]) == [length_scale] * 3
    assert list(phase.state_scale[3:6]) == [velocity_scale] * 3
    assert list(phase.dynamics_scale[:3]) == [length_scale] * 3
    assert spec.discrete_scale[18] == length_scale
    assert spec.discrete_scale[19] == 1.0


def test_the_sampled_guess_reaches_the_transcription():
    """Each field carries its own sample times; they are merged onto one grid per phase."""
    from yapss._api.compile import to_transcription_spec
    from yapss._api.spec import snapshot

    phase = to_transcription_spec(snapshot(setup())).phases[0]
    assert len(phase.guess_time) == 9
    assert phase.guess_state.shape == (7, 9)
    assert phase.guess_control.shape == (3, 9)
