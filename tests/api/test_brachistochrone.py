"""End-to-end solves of the brachistochrone problem through the redesigned API."""

import math

import pytest

from yapss.examples.brachistochrone import Phases, setup

ANALYTIC = math.sqrt(math.pi / 32.174)
"""The minimum time to reach the line x = 1, sliding from rest under gravity."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_solves_to_the_analytic_answer(problem):
    solution = problem.solve()
    assert solution.converged
    assert solution.objective == pytest.approx(ANALYTIC, rel=1e-8)


def test_it_agrees_with_the_same_problem_in_the_released_api(problem):
    from yapss._legacy.examples.brachistochrone_minimal import setup as legacy_setup

    legacy = legacy_setup()
    legacy.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(legacy.solve().objective, rel=1e-8)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(ANALYTIC, rel=1e-6)


@pytest.mark.parametrize("method", ["lgl", "lgr", "lg"])
def test_every_spectral_method_agrees(problem, method):
    problem.spectral_method = method
    assert problem.solve().objective == pytest.approx(ANALYTIC, rel=1e-6)


def test_the_solution_is_reached_by_name(problem):
    solution = problem.solve()
    phase = solution[problem.phases.slide]
    assert phase.state.x.shape == phase.time.shape
    assert phase.control.u.shape == phase.time.shape
    assert phase.final.x == pytest.approx(1.0)
    assert phase.initial.v == pytest.approx(0.0, abs=1e-6)
    assert len(phase.state) == 3
    assert phase.state[:].shape == (3, len(phase.time))


def test_a_solutions_names_are_fixed(problem):
    solution = problem.solve()
    with pytest.raises(AttributeError, match="names are fixed"):
        solution[problem.phases.slide].state.x = 1.0
    with pytest.raises(AttributeError, match="names are fixed"):
        solution.objective = 1.0


def test_a_phase_is_reached_by_handle_not_by_index(problem):
    solution = problem.solve()
    with pytest.raises(KeyError, match="takes a phase handle"):
        solution[0]


def test_a_later_edit_does_not_alter_an_earlier_solution(problem):
    """Each solve snapshots the problem, so a continuation loop cannot rewrite its own past."""
    first = problem.solve()
    problem.phases.slide.state.x.bounds = (0, 100)
    assert first.objective == pytest.approx(ANALYTIC, rel=1e-8)
    assert problem.solve().objective == pytest.approx(ANALYTIC, rel=1e-8)


def test_the_mesh_is_one_value(problem):
    from yapss._api import Mesh

    problem.phases.slide.mesh = Mesh.uniform(segments=4, points=6)
    solution = problem.solve()
    assert solution.objective == pytest.approx(ANALYTIC, rel=1e-6)
    assert solution[problem.phases.slide].mesh.collocation_points == (6, 6, 6, 6)


def test_the_phases_declaration_is_reached_by_name_and_index():
    phases = Phases()
    assert phases.slide.name == "slide"
    assert phases.slide.index == 0
    assert len(phases) == 1
    assert phases[0] is phases.slide
    with pytest.raises(AttributeError, match=r"has no phase 'slid'\. Did you mean 'slide'\?"):
        phases.slid
