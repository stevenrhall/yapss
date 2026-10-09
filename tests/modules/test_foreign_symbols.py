"""Under "auto", a callback that uses a value kept from another trace is refused clearly."""

# standard imports
import traceback

# third party imports
import pytest

# package imports
from tests.support.legacy import Problem

MESSAGE = "symbolic value that did not come from"


def make_problem(continuous, objective, discrete=None):
    problem = Problem(name="foreign", nx=[2], nu=[1], nq=[1], nd=0 if discrete is None else 1)
    problem.functions.continuous = continuous
    problem.functions.objective = objective
    if discrete is not None:
        problem.functions.discrete = discrete
        problem.bounds.discrete.lower[:] = -5
        problem.bounds.discrete.upper[:] = 5
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0
    bounds.final_time.lower = bounds.final_time.upper = 1
    bounds.initial_state.lower[:] = bounds.initial_state.upper[:] = [0, 0]
    bounds.final_state.lower[:] = [-3, 0]
    bounds.final_state.upper[:] = [3, 0]
    guess = problem.guess.phase[0]
    guess.time = [0, 1]
    guess.state = [[0, 1], [0, 0]]
    guess.control = [[0, 0]]
    guess.integral = [0.0]
    problem.mesh.phase[0].collocation_points = [4]
    problem.mesh.phase[0].fraction = [1.0]
    problem.derivatives.method = "auto"
    problem.derivatives.order = "first"
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.sb = "yes"
    return problem


def continuous_keeping(cache):
    def continuous(arg):
        x, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = v, u
        arg.phase[0].integrand[:] = (u**2,)
        cache["x"] = x[-1]

    return continuous


def plain_objective(arg):
    arg.objective = arg.phase[0].integral[0]


def raised_on_line(error, name):
    """Return the line of function `name` the traceback passes through."""
    frames = [
        f
        for f in traceback.extract_tb(error.__traceback__)
        if f.name == name and f.filename == __file__
    ]
    return frames[-1].line if frames else None


def test_objective_using_a_continuous_value():
    cache = {}

    def objective(arg):
        arg.objective = arg.phase[0].integral[0] + (cache.get("x", 0.0) - 1) ** 2

    problem = make_problem(continuous_keeping(cache), objective)
    with pytest.raises(TypeError, match=f"the objective was assigned a {MESSAGE}") as info:
        problem.solve()
    assert "arg.objective =" in raised_on_line(info.value, "objective")


def test_objective_keeping_its_own_symbol():
    kept = {}

    def objective(arg):
        kept.setdefault("xf", arg.phase[0].final_state[0])
        arg.objective = arg.phase[0].integral[0] + (kept["xf"] - 1) ** 2

    problem = make_problem(continuous_keeping({}), objective)
    with pytest.raises(TypeError, match=f"the objective was assigned a {MESSAGE}"):
        problem.solve()


def test_discrete_using_a_continuous_value():
    cache = {}

    def discrete(arg):
        arg.discrete[0] = cache.get("x", 0.0)

    problem = make_problem(continuous_keeping(cache), plain_objective, discrete)
    with pytest.raises(TypeError, match=rf"arg.discrete\[0\] was assigned a {MESSAGE}") as info:
        problem.solve()
    assert "arg.discrete[0] =" in raised_on_line(info.value, "discrete")


def test_discrete_in_place_using_a_continuous_value():
    cache = {}

    def discrete(arg):
        arg.discrete[:] = (0.0,)
        arg.discrete += (cache.get("x", 0.0),)

    problem = make_problem(continuous_keeping(cache), plain_objective, discrete)
    with pytest.raises(TypeError, match=f"arg.discrete was assigned a {MESSAGE}"):
        problem.solve()


def test_continuous_using_an_objective_value():
    cache = {}

    def objective(arg):
        cache["xf"] = arg.phase[0].final_state[0]
        arg.objective = arg.phase[0].integral[0]

    def continuous(arg):
        x, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = v, u
        arg.phase[0].integrand[:] = (u**2 + 1e-3 * cache.get("xf", 0.0) ** 2,)

    problem = make_problem(continuous, objective)
    with pytest.raises(TypeError, match=f"The continuous callback, .*continuous .*line") as info:
        problem.solve()
    assert MESSAGE in str(info.value)
    assert "are free" in str(info.value.__cause__)


def test_symbol_kept_from_an_earlier_solve():
    traced = []

    def continuous(arg):
        continuous_keeping({})(arg)
        traced.append(arg.phase[0].state[0][-1])

    problem = make_problem(continuous, plain_objective)
    problem.solve()
    stale = traced[0]  # the first call is the trace

    def objective(arg):
        arg.objective = arg.phase[0].integral[0] + 0.1 * stale**2

    problem.functions.objective = objective
    with pytest.raises(TypeError, match=f"The objective callback, .*objective .*line"):
        problem.solve()


def test_numeric_errors_are_unchanged():
    def discrete(arg):
        arg.discrete[0] = "one"

    problem = make_problem(continuous_keeping({}), plain_objective, discrete)
    with pytest.raises((TypeError, ValueError)) as info:
        problem.solve()
    assert MESSAGE not in str(info.value)
