"""
Solution arrays in phases with no controls or no states keep their point count.

``np.array([])`` is ``(0,)``; the per-point arrays must be ``(0, n)`` so that anything
indexing axis 1 -- ``Guess.from_solution`` in particular -- works on a coast phase or a
parameter-only phase. Through 0.2.2 a warm start from such a solution raised.
"""

import numpy as np
import pytest

from yapss import Problem

METHODS = ["lg", "lgr", "lgl"]


def _no_control_problem(spectral_method: str) -> Problem:
    """x' = 1, x(0) = 0, tf = 1, minimize x(tf): trivial, but nu == 0."""
    problem = Problem(name="coast", nx=[1], nu=[0])
    problem.spectral_method = spectral_method

    def objective(arg):
        arg.objective = arg.phase[0].final_state[0]

    def continuous(arg):
        for p in arg.phase_list:
            arg.phase[p].dynamics[:] = (1.0,)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower = bounds.initial_state.upper = [0.0]
    problem.guess.phase[0].time = [0.0, 1.0]
    problem.guess.phase[0].state = [[0.0, 1.0]]
    problem.ipopt_options.print_level = 0
    return problem


def _no_state_problem(spectral_method: str) -> Problem:
    """Minimize the integral of (u - 1)^2 with a path bound on u: nx == 0."""
    problem = Problem(name="stateless", nx=[0], nu=[1], nq=[1], nh=[1])
    problem.spectral_method = spectral_method

    def objective(arg):
        arg.objective = arg.phase[0].integral[0]

    def continuous(arg):
        for p in arg.phase_list:
            u = arg.phase[p].control[0]
            arg.phase[p].integrand[:] = ((u - 1.0) ** 2,)
            arg.phase[p].path[:] = (u,)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 2.0
    bounds.path.lower, bounds.path.upper = [-5.0], [5.0]
    problem.guess.phase[0].time = [0.0, 2.0]
    problem.guess.phase[0].control = [[0.0, 0.0]]
    problem.ipopt_options.print_level = 0
    return problem


@pytest.mark.parametrize("spectral_method", METHODS)
def test_no_control_phase_arrays_keep_point_count(spectral_method: str) -> None:
    problem = _no_control_problem(spectral_method)
    solution = problem.solve()
    phase = solution.phase[0]
    assert solution.nlp_info.ipopt_status == 0
    assert phase.control.shape == (0, len(phase.time_c))
    assert phase.control_multiplier.shape[0] == 0 and phase.control_multiplier.ndim == 2
    assert phase.state.shape == (1, len(phase.time))
    assert phase.costate.ndim == 2

    # the documented warm-start path
    problem.guess.from_solution(solution)
    assert problem.guess.phase[0].control.shape == (0, len(phase.time))
    second = problem.solve()
    np.testing.assert_allclose(second.objective, 1.0, atol=1e-8)


@pytest.mark.parametrize("spectral_method", METHODS)
def test_no_state_phase_arrays_keep_point_count(spectral_method: str) -> None:
    problem = _no_state_problem(spectral_method)
    solution = problem.solve()
    phase = solution.phase[0]
    assert solution.nlp_info.ipopt_status == 0
    assert phase.state.shape == (0, len(phase.time))
    assert phase.costate.shape[0] == 0 and phase.costate.ndim == 2
    assert phase.dynamics.shape[0] == 0 and phase.dynamics.ndim == 2
    assert phase.control.shape == (1, len(phase.time_c))
    assert phase.path_multiplier.ndim == 2

    problem.guess.from_solution(solution)
    assert problem.guess.phase[0].state.shape == (0, len(phase.time))
    second = problem.solve()
    np.testing.assert_allclose(second.phase[0].control, 1.0, atol=1e-6)


@pytest.mark.parametrize("spectral_method", ["lgr", "lg", "lgl"])
def test_initial_and_final_state_are_the_endpoint_variables(spectral_method: str) -> None:
    """`initial_state`/`final_state` equal the NLP's own endpoint state variables.

    The mesh is multi-segment and non-uniform so that LG, which stores its endpoint and
    segment-boundary states after the collocation states, is exercised.
    """
    from yapss._private.structure import get_nlp_dv_structure
    from yapss.examples import goddard_problem_3_phase

    problem = goddard_problem_3_phase.setup()
    problem.spectral_method = spectral_method
    problem.ipopt_options.print_level = 0
    for mesh_phase in problem.mesh.phase:
        mesh_phase.collocation_points = (5, 6, 4)
        mesh_phase.fraction = (0.3, 0.3, 0.4)
    solution = problem.solve()

    dv = get_nlp_dv_structure(solution.problem, float)
    dv.z[:] = solution.nlp_info.x
    for p, phase in enumerate(solution.phase):
        nx = problem.nx[p]
        assert phase.initial_state.shape == (nx,)
        assert phase.final_state.shape == (nx,)
        np.testing.assert_array_equal(phase.initial_state, dv.phase[p].x0)
        np.testing.assert_array_equal(phase.final_state, dv.phase[p].xf)
        np.testing.assert_array_equal(phase.initial_state, phase.state[:, 0])
        np.testing.assert_array_equal(phase.final_state, phase.state[:, -1])


def test_initial_and_final_state_are_copies() -> None:
    from yapss.examples import brachistochrone_minimal

    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0
    phase = problem.solve().phase[0]
    before = phase.state.copy()
    phase.initial_state[:] = -1.0
    phase.final_state[:] = -1.0
    np.testing.assert_array_equal(phase.state, before)
