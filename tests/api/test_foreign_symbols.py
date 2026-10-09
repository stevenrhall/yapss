"""Under "auto", a callback that uses a value kept from another trace is refused clearly.

Each callback is traced on its own, so a symbol one callback keeps, through a closure or a
global, means nothing in another callback's trace, or in a later call of its own. The refusal
names the user's callback, not the adapter the front end calls it through.
"""

import pytest

from yapss.examples import brachistochrone_minimal
from yapss.math import cos, sin

MESSAGE = "symbolic value that did not come from"


def test_continuous_using_a_value_kept_from_the_objective():
    problem = brachistochrone_minimal.setup()
    ph = problem.phases.phase
    kept = {}

    @problem.register.objective
    def objective(arg):
        kept["tf"] = arg[ph].final_time
        return arg[ph].final_time

    @ph.register.continuous
    def continuous(arg, out):
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u) + 1e-9 * kept.get("tf", 1.0)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = brachistochrone_minimal.G0 * sin(u)

    problem.ipopt_options.print_level = 0
    with pytest.raises(TypeError, match=MESSAGE) as info:
        problem.solve()
    where = f"{continuous.__qualname__} ({__file__}, line {continuous.__code__.co_firstlineno})"
    assert f"The continuous callback, {where}" in str(info.value)


def test_objective_returning_a_value_kept_from_an_earlier_call():
    problem = brachistochrone_minimal.setup()
    ph = problem.phases.phase
    first = []

    @problem.register.objective
    def objective(arg):
        if not first:
            first.append(arg[ph].final_time)
        return first[0]

    problem.ipopt_options.print_level = 0
    with pytest.raises(TypeError, match=f"the objective was assigned a {MESSAGE}"):
        problem.solve()
