"""
Continuous-time multipliers against a problem with a known solution.

Minimize x(tf) subject to xdot = u, x(0) = 0, and -1 <= u <= 1. The optimal control is
u = -1 throughout, the costate is identically 1, and stationarity of the Hamiltonian
H = lambda * u in u gives an active-bound multiplier of magnitude 1 -- for every final
time. The same bound expressed as a path constraint h = u must give the same multiplier.

Durations other than 2 discriminate: at tf - t0 = 2 the half-duration is 1 and any power
of it is invisible, which is how the scaling of both multipliers went wrong through 0.2.2.
"""

import numpy as np
import pytest

import yapss

DURATIONS = [2.0, 4.0, 0.5]
METHODS = ["lg", "lgr", "lgl"]


def _setup(tf: float, *, as_path: bool) -> yapss.Problem:
    problem = yapss.Problem(name="bound multiplier", nx=[1], nu=[1], nh=[1 if as_path else 0])

    def objective(arg):
        arg.objective = arg.phase[0].final_state[0]

    def continuous(arg):
        for p in arg.phase_list:
            u = arg.phase[p].control[0]
            arg.phase[p].dynamics[:] = (u,)
            if as_path:
                arg.phase[p].path[:] = (u,)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = tf
    bounds.initial_state.lower = bounds.initial_state.upper = [0.0]
    if as_path:
        bounds.path.lower, bounds.path.upper = [-1.0], [1.0]
    else:
        bounds.control.lower, bounds.control.upper = [-1.0], [1.0]
    problem.guess.phase[0].time = [0.0, tf]
    problem.guess.phase[0].state = [[0.0, -tf]]
    problem.guess.phase[0].control = [[-1.0, -1.0]]
    problem.mesh.phase[0].collocation_points = (5, 5)
    problem.mesh.phase[0].fraction = (0.5, 0.5)
    problem.ipopt_options.print_level = 0
    return problem


@pytest.mark.parametrize("spectral_method", METHODS)
@pytest.mark.parametrize("tf", DURATIONS)
def test_bound_multipliers_are_densities_in_time(spectral_method: str, tf: float) -> None:
    control_problem = _setup(tf, as_path=False)
    path_problem = _setup(tf, as_path=True)
    for problem in (control_problem, path_problem):
        problem.spectral_method = spectral_method
    control_solution = control_problem.solve()
    path_solution = path_problem.solve()

    for solution in (control_solution, path_solution):
        assert solution.nlp_info.ipopt_status == 0
        np.testing.assert_allclose(solution.phase[0].costate, 1.0, atol=1e-6)

    control_multiplier = control_solution.phase[0].control_multiplier
    path_multiplier = path_solution.phase[0].path_multiplier
    # magnitude 1, independent of tf: dH/du + mu = 0 with lambda = 1
    np.testing.assert_allclose(np.abs(control_multiplier), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.abs(path_multiplier), 1.0, atol=1e-6)
    # the same constraint, however expressed, has the same multiplier
    np.testing.assert_allclose(control_multiplier, path_multiplier, atol=1e-6)


def test_zero_duration_phase_multipliers_are_nan() -> None:
    """A density in time on a phase of zero duration is undefined, and says so quietly."""
    problem = _setup(1.0, as_path=True)
    problem.bounds.phase[0].control.lower = [-1.0]
    problem.bounds.phase[0].control.upper = [1.0]
    problem.bounds.phase[0].final_time.lower = 0.0
    problem.bounds.phase[0].final_time.upper = 0.0
    with np.errstate(all="raise"):
        solution = problem.solve()
    assert np.all(np.isnan(solution.phase[0].control_multiplier))
    assert np.all(np.isnan(solution.phase[0].path_multiplier))
    assert np.all(np.isfinite(solution.phase[0].costate))


def test_zero_duration_phase_multipliers_are_nan_whatever_ipopt_returns(monkeypatch) -> None:
    """NaN must not depend on the NLP multipliers happening to be exactly zero.

    The bound and the path constraint above duplicate each other, so how Ipopt splits the
    multiplier between them is build-dependent: conda-forge Ipopt 3.14.19 on macOS returned
    nonzero path-row values, which divided by the zero duration to -inf. Forcing every NLP
    multiplier nonzero makes the case deterministic on any build.
    """
    from yapss._private import solver

    make_solution_object = solver.make_solution_object

    def nonzero_multipliers(problem, mesh, nlp, nlp_info):
        for key in ("mult_g", "mult_x_L", "mult_x_U"):
            nlp_info[key] = np.linspace(0.5, 1.5, len(nlp_info[key]))
        nlp_info["mult_x_U"] = -nlp_info["mult_x_U"]
        return make_solution_object(problem, mesh, nlp, nlp_info)

    monkeypatch.setattr(solver, "make_solution_object", nonzero_multipliers)
    problem = _setup(1.0, as_path=True)
    problem.bounds.phase[0].control.lower = [-1.0]
    problem.bounds.phase[0].control.upper = [1.0]
    problem.bounds.phase[0].final_time.lower = 0.0
    problem.bounds.phase[0].final_time.upper = 0.0
    with np.errstate(all="raise"):
        solution = problem.solve()
    assert np.all(np.isnan(solution.phase[0].control_multiplier))
    assert np.all(np.isnan(solution.phase[0].path_multiplier))
