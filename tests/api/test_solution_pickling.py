"""A solution is data, so it pickles, however its problem was declared.

It keeps no callbacks, no reference to the problem, and not even the classes the problem was
declared with -- only their shapes, as plain data. The case that matters is a problem declared
inside a function, whose classes cannot be pickled by reference, and which is how a continuation
sweep or a multiprocessing worker is naturally written.
"""

import importlib
import pickle

import numpy as np
import pytest

import yapss


def declared_in_a_function():
    """Return a problem whose every declaration is local to this function."""

    class State(yapss.State):
        r = yapss.vector(2)
        y = yapss.scalar()

    class Control(yapss.Control):
        u = yapss.scalar()

    class Path(yapss.Path):
        size = yapss.scalar()

    class Integral(yapss.Integral):
        effort = yapss.scalar()

    class Parameter(yapss.Parameter):
        k = yapss.scalar()

    class Discrete(yapss.Discrete):
        end = yapss.scalar()

    class Run(yapss.Phase):
        state: State
        control: Control
        path: Path
        integral: Integral
        s: yapss.Independent

    class Phases(yapss.Phases):
        run: Run

    problem = yapss.Problem("local", phases=Phases, discrete=Discrete, parameter=Parameter)
    ph = problem.phases.run

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.r = [arg.control.u, -arg.control.u]
        out.dynamics.y = arg.parameter.k * arg.control.u
        out.path.size = arg.control.u
        out.integrand.effort = arg.control.u**2
        return out

    @problem.register.objective
    def objective(arg):
        return arg[ph].integral.effort

    @problem.register.discrete
    def discrete(arg, out):
        out.discrete.end = arg[ph].final.y
        return out

    ph.s.initial = (0.0, 0.0)
    ph.s.final = (1.0, 1.0)
    ph.s.guess = (0.0, 1.0)
    ph.state.r.initial[:] = (0.0, 0.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.control.u.bounds = (-10.0, 10.0)
    ph.path.size.bounds = (-10.0, 10.0)
    problem.parameter.k.bounds = (2.0, 2.0)
    problem.parameter.k.guess = 2.0
    problem.discrete.end.bounds = (1.0, 1.0)
    problem.ipopt_options.print_level = 0
    return problem


@pytest.fixture(scope="module")
def solved():
    problem = declared_in_a_function()
    return problem, problem.solve()


def test_a_solution_to_a_problem_declared_in_a_function_pickles(solved):
    problem, solution = solved
    ph = problem.phases.run
    copy = pickle.loads(pickle.dumps(solution))
    assert copy.objective == solution.objective
    assert copy.parameter.k == solution.parameter.k
    assert copy.discrete.end == solution.discrete.end
    ps, ps_copy = solution[ph], copy[ph]
    np.testing.assert_array_equal(ps_copy.s, ps.s)
    np.testing.assert_array_equal(ps_copy.state.r, ps.state.r)
    np.testing.assert_array_equal(ps_copy.state.y, ps.state.y)
    np.testing.assert_array_equal(ps_copy.control.u, ps.control.u)
    np.testing.assert_array_equal(ps_copy.costate.y, ps.costate.y)
    assert ps_copy.integral.effort == ps.integral.effort
    assert ps_copy.final.y == ps.final.y
    assert ps_copy.final.s == ps.final.s
    assert ps_copy.mesh == ps.mesh


def test_a_pickled_solution_keeps_its_names_and_messages(solved):
    problem, solution = solved
    copy = pickle.loads(pickle.dumps(solution))
    with pytest.raises(AttributeError, match=r"has no field 'yy'\. Did you mean 'y'\?"):
        copy[problem.phases.run].state.yy  # noqa: B018
    with pytest.raises(AttributeError, match="read-only"):
        copy.objective = 0.0


def test_a_solution_holds_neither_the_problem_nor_its_classes(solved):
    """What pickles is data: nothing in the pickle names the function the problem was made in."""
    _, solution = solved
    assert b"declared_in_a_function" not in pickle.dumps(solution)


def test_one_piece_of_a_solution_pickles_on_its_own(solved):
    problem, solution = solved
    ps = solution[problem.phases.run]
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(ps.state)).r, ps.state.r)
    assert pickle.loads(pickle.dumps(ps.final)).y == ps.final.y


def test_a_solution_is_read_with_the_handles_of_a_rebuilt_problem(solved):
    """Handles are matched by position and name, so a solution outlives the problem object."""
    _, solution = solved
    rebuilt = declared_in_a_function()
    assert solution[rebuilt.phases.run].final.y == pytest.approx(1.0)


def test_a_pickled_solution_is_read_by_phase_name_with_no_problem_at_all(solved):
    """What makes pickling useful in a worker: nothing from the problem is needed to read it."""
    problem, solution = solved
    copy = pickle.loads(pickle.dumps(solution))
    assert copy["run"].final.y == solution[problem.phases.run].final.y


def test_a_misspelled_phase_name_is_refused_with_a_suggestion(solved):
    _, solution = solved
    with pytest.raises(KeyError, match=r"has no phase 'rn'\. Did you mean 'run'\?"):
        solution["rn"]


def test_a_vector_of_a_solution_is_not_an_instance_of_the_users_class(solved):
    """The one visible cost of holding no classes, stated so it is not discovered."""
    problem, solution = solved
    declaration = problem.phases.run._declaration.state
    assert not isinstance(solution[problem.phases.run].state, declaration)


EXAMPLES = [
    "brachistochrone",
    "brachistochrone_minimal",
    "brachistochrone_user_derivatives",
    "delta_iii_ascent",
    "dynamic_soaring",
    "goddard_problem_1_phase",
    "goddard_problem_3_phase",
    "hs071",
    "isoperimetric",
    "minimum_time_to_climb",
    "newton",
    "orbit_raising",
    "rosenbrock",
]


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_example_s_solution_pickles(name):
    module = importlib.import_module(f"yapss.examples.{name}")
    problem = module.setup()
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    copy = pickle.loads(pickle.dumps(solution))
    assert copy.objective == solution.objective
    for ph in problem.phases:
        np.testing.assert_array_equal(copy[ph].state[:], solution[ph].state[:])
