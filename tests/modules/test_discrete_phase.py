"""

Test the phase object passed to the objective and discrete callbacks.

`DiscretePhase` listed empty `_allowed_attrs` without inheriting `Protected`, so the lists
did nothing: `arg.phase[0].objective = arg.phase[0].final_time` in an objective callback
was stored and ignored, and the solve reported status 0 with an objective of 0.

"""

from __future__ import annotations

import traceback

import pytest

from yapss._legacy.examples import brachistochrone_minimal, goddard_problem_3_phase


def test_misspelled_output_on_objective_phase_raises_at_the_user_line():
    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0

    def objective(arg) -> None:
        arg.phase[0].objective = arg.phase[0].final_time  # should be arg.objective

    problem.functions.objective = objective
    with pytest.raises(
        AttributeError, match="cannot set 'DiscretePhase' attribute 'objective'"
    ) as info:
        problem.solve()
    frames = traceback.extract_tb(info.value.__traceback__)
    user_frames = [frame for frame in frames if frame.name == "objective"]
    assert user_frames, "the traceback does not reach the user's callback"
    assert "arg.phase[0].objective" in (user_frames[-1].line or "")


def test_misspelled_attribute_on_discrete_phase_raises():
    problem = goddard_problem_3_phase.setup()
    problem.ipopt_options.print_level = 0
    original = problem.functions.discrete

    def wrap(function):
        def discrete(arg) -> None:
            function(arg)
            arg.phase[0].final_tme = 0.0

        return discrete

    problem.functions.discrete = wrap(original)
    with pytest.raises(AttributeError, match="cannot set 'DiscretePhase' attribute 'final_tme'"):
        problem.solve()


def test_discrete_phase_inputs_are_still_readable_and_not_assignable():
    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0
    seen = {}

    def objective(arg) -> None:
        phase = arg.phase[0]
        seen["read"] = (
            phase.initial_time,
            phase.final_time,
            phase.initial_state,
            phase.final_state,
        )
        try:
            phase.final_time = 1.0
        except AttributeError as exc:
            seen["error"] = str(exc)
        arg.objective = phase.final_time

    problem.functions.objective = objective
    solution = problem.solve()
    assert solution.nlp_info.ipopt_status == 0
    assert len(seen["read"]) == 4
    assert "cannot set 'DiscretePhase' attribute 'final_time'" in seen["error"]
