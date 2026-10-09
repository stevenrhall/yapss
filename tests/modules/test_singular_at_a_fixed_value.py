"""A function singular at a fixed value solves: Ipopt never uses a derivative there.

A vehicle starting from rest with drag through ``sqrt(vx**2 + vy**2)``, ``sqrt(t)`` with the
initial time fixed at zero, and ``sqrt(p)`` with a parameter fixed at zero each have a NaN or
infinite derivative with respect to a variable whose bounds are equal. Ipopt removes such a
variable (make_parameter), so 0.2.3 solved these; 0.3.0 refused them until the derivative checks
left those entries out. A bound multiplier of such a variable is NaN, since the derivative it is
computed from is not finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.support import legacy as yapss
from tests.support.legacy import Problem
from yapss._backend.structure import get_nlp_dv_structure
from yapss.math import sqrt

SPECTRAL = ["lg", "lgr", "lgl"]
METHODS = ["auto", "central-difference", "central-difference-full"]


def from_rest(spectral_method: str, method: str) -> Problem:
    problem = Problem(name="from rest", nx=[2], nu=[1])

    def objective(arg: yapss.ObjectiveArg) -> None:
        arg.objective = arg.phase[0].final_time

    def continuous(arg: yapss.ContinuousArg) -> None:
        vx, vy = arg.phase[0].state
        (u,) = arg.phase[0].control
        v = sqrt(vx**2 + vy**2)
        arg.phase[0].dynamics[:] = u - 0.1 * v * vx, 1.0 - 0.1 * v * vy

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower[:] = bounds.initial_state.upper[:] = 0.0
    bounds.state.lower[:], bounds.state.upper[:] = -10.0, 10.0
    bounds.control.lower[:], bounds.control.upper[:] = -10.0, 10.0
    guess = problem.guess.phase[0]
    guess.time, guess.state, guess.control = [0.0, 1.0], [[0.1, 1.0], [0.1, 1.0]], [[0.5, 0.5]]
    problem.spectral_method = spectral_method
    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    return problem


def fixed_at_zero(case: str, spectral_method: str, method: str) -> Problem:
    problem = Problem(name=case, nx=[1], nu=[1], ns=1 if case == "parameter" else 0)
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower[0] = bounds.initial_state.upper[0] = 0.0
    bounds.state.lower[:], bounds.state.upper[:] = -10.0, 10.0
    bounds.control.lower[:], bounds.control.upper[:] = -1.0, 1.0
    if case == "parameter":
        problem.bounds.parameter.lower[:] = problem.bounds.parameter.upper[:] = 0.0
    guess = problem.guess.phase[0]
    guess.time, guess.state, guess.control = [0.0, 1.0], [[0.0, 0.5]], [[0.0, 0.0]]

    def continuous(arg: yapss.ContinuousArg) -> None:
        (x,) = arg.phase[0].state
        (u,) = arg.phase[0].control
        w = sqrt(arg.phase[0].time) if case == "time" else sqrt(arg.parameter[0]) * x
        arg.phase[0].dynamics[0] = u - w

    def objective(arg: yapss.ObjectiveArg) -> None:
        arg.objective = arg.phase[0].final_state[0]

    problem.functions.continuous = continuous
    problem.functions.objective = objective
    problem.spectral_method = spectral_method
    problem.mesh.phase[0].collocation_points = 3 * (4,)
    problem.mesh.phase[0].fraction = 3 * (1 / 3,)
    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    return problem


@pytest.mark.parametrize("spectral_method", SPECTRAL)
@pytest.mark.parametrize("method", METHODS)
def test_start_from_rest_solves(spectral_method: str, method: str) -> None:
    solution = from_rest(spectral_method, method).solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED


@pytest.mark.parametrize("spectral_method", SPECTRAL)
def test_start_from_rest_agrees_across_methods(spectral_method: str) -> None:
    auto = from_rest(spectral_method, "auto").solve()
    cd = from_rest(spectral_method, "central-difference").solve()
    np.testing.assert_allclose(auto.phase[0].state, cd.phase[0].state, atol=1e-6)


@pytest.mark.parametrize(("case", "objective"), [("time", -1.664773), ("parameter", -1.0)])
@pytest.mark.parametrize("spectral_method", SPECTRAL)
@pytest.mark.parametrize("method", METHODS)
def test_singular_at_a_fixed_time_or_parameter_solves(
    case: str, objective: float, spectral_method: str, method: str
) -> None:
    solution = fixed_at_zero(case, spectral_method, method).solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    if case == "parameter" or spectral_method == "lgl":
        # the values 0.2.3 gives; sqrt(t)'s integral differs between transcriptions by more
        # than solver tolerance, so it is pinned for the transcription it was measured on
        assert solution.objective == pytest.approx(objective, abs=1e-6)


def test_the_multiplier_of_a_variable_with_an_undefined_derivative_is_nan() -> None:
    """Under "auto" the derivative of sqrt(vx**2 + vy**2) at rest is NaN.

    Read from the NLP's bound multipliers, at the initial state's positions.
    """
    problem = from_rest("lgr", "auto")
    solution = problem.solve()
    index = get_nlp_dv_structure(problem._to_spec(), np.intp)
    index.z[:] = np.arange(len(index.z))
    initial = np.asarray(index.phase[0].x0, dtype=np.intp)
    nlp = solution.nlp_info
    assert np.all(np.isnan(nlp.mult_x_L[initial]))
    assert np.all(np.isnan(nlp.mult_x_U[initial]))
    assert np.all(np.isfinite(solution.phase[0].costate))
