"""Every symbolic array a callback is handed is an `SXArray`, and arithmetic on it is quiet.

A block of symbols is an object-dtype array, and numpy has no loop for it but the object loop,
which reports a stale floating-point flag as a spurious ``RuntimeWarning: invalid value
encountered in divide`` whenever a large constant is involved (numpy issue 21416). `SXArray`
implements ``__array_ufunc__`` and routes every ufunc through the `yapss.math` table instead, so
that loop never runs -- which is the fix 0.2.3 made for the released front end, and which this
front end lost for *block* fields: a scalar row kept the view by slicing, while a block was
rebuilt by `_stack` into a plain array. Delta III showed it, dividing the gravitational parameter
by a radius cubed.

Nothing else catches it. The values are right either way, the suite passes, and only a warning on
stderr says anything is wrong.
"""

import warnings
from typing import Any

import numpy as np
import pytest

import yapss
from yapss.math.wrapper import SXW, SXArray

MU = 3.986012e14
"""A constant large enough to set the flag numpy's object loop reports (> 2**31)."""


class State(yapss.State):
    """A scalar beside a block, so both paths are exercised."""

    h = yapss.scalar()
    r = yapss.vector(3)


class Control(yapss.Control):
    """Likewise for the control."""

    u = yapss.scalar()
    n = yapss.vector(2)


class Parameter(yapss.Parameter):
    """And for the parameters, which the endpoint callbacks also see."""

    mass = yapss.scalar()
    k = yapss.vector(2)


class Discrete(yapss.Discrete):
    """One scalar and one block of discrete constraints."""

    gap = yapss.scalar()
    vec_gap = yapss.vector(3)


class Flight(yapss.Phase):
    """One phase carrying every kind of vector."""

    state: State
    control: Control


class Phases(yapss.Phases):
    """One phase."""

    flight: Flight


def record(seen: dict[str, Any], label: str, value: Any) -> None:
    """Record what `value` is, and whether arithmetic on it warns."""
    if label in seen:
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = MU / (value + 1.0) ** 3
        warned = [w.category.__name__ for w in caught]
    seen[label] = (isinstance(value, (SXArray, SXW)), warned)


def build(seen: dict[str, Any]) -> yapss.Problem:
    """Return a problem whose callbacks record every symbolic array they are handed."""
    problem = yapss.Problem("blocks", phases=Phases, parameter=Parameter, discrete=Discrete)
    ph = problem.phases.flight

    @ph.register.continuous
    def continuous(arg, out):
        record(seen, "continuous state scalar", arg.state.h)
        record(seen, "continuous state block", arg.state.r)
        record(seen, "continuous state block row", arg.state.r[0])
        record(seen, "continuous control block", arg.control.n)
        record(seen, "continuous time", arg.time)
        record(seen, "continuous parameter block", arg.parameter.k)
        out.dynamics.h = arg.control.u
        out.dynamics.r = arg.state.r * 0.0

    @problem.register.discrete
    def discrete(arg, out):
        end = arg[ph]
        record(seen, "endpoint scalar", end.final.h)
        record(seen, "endpoint block", end.final.r)
        record(seen, "endpoint parameter block", arg.parameter.k)
        out.discrete.gap = end.final.h - end.initial.h
        out.discrete.vec_gap = end.final.r - end.initial.r

    @problem.register.objective
    def objective(arg):
        record(seen, "objective parameter block", arg.parameter.k)
        return arg[ph].final.h

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 1.0)
    ph.state.h.initial = (1.0, 1.0)
    ph.state.h.bounds = (0.0, 10.0)
    ph.state.r.bounds[:] = (-10.0, 10.0)
    ph.state.r.initial[:] = (1.0, 1.0)
    ph.control.u.bounds = (-1.0, 1.0)
    ph.control.n.bounds[:] = (-1.0, 1.0)
    problem.parameter.mass.bounds = (1.0, 1.0)
    problem.parameter.k.bounds[:] = (1.0, 1.0)
    problem.discrete.gap.bounds = (None, 100.0)
    problem.discrete.vec_gap.bounds[:] = (None, 100.0)

    ph.time.guess = (0.0, 1.0)
    ph.state.h.guess = (1.0, 1.0)
    ph.state.r.guess[:] = (1.0, 1.0)
    ph.control.u.guess = (0.0, 0.0)
    ph.control.n.guess[:] = (0.0, 0.0)
    problem.parameter.mass.guess = 1.0
    problem.parameter.k.guess[:] = 1.0

    problem.ipopt_options.print_level = 0
    problem.ipopt_options.max_iter = 1
    return problem


@pytest.fixture(scope="module")
def recorded() -> dict[str, Any]:
    """Solve once under the trace, recording what each callback was handed."""
    seen: dict[str, Any] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", yapss.IpoptConvergenceWarning)
        build(seen).solve()
    return seen


def test_every_symbolic_array_is_an_sxarray(recorded):
    """A block field is not a plain object array; the view carries `__array_ufunc__`."""
    plain = [label for label, (is_symbolic, _) in recorded.items() if not is_symbolic]
    assert plain == [], f"handed a plain object array: {plain}"


def test_arithmetic_on_a_symbolic_array_is_quiet(recorded):
    """No spurious RuntimeWarning, which is what numpy's object loop would produce."""
    noisy = {label: warned for label, (_, warned) in recorded.items() if warned}
    assert noisy == {}, f"numpy's object loop ran: {noisy}"


def test_a_plain_object_array_is_what_it_would_warn_about():
    """The warning is real, so the test above is testing something.

    Built by hand from the same symbols, without the view, and the loop runs.
    """
    import casadi as ca

    symbols = [SXW(ca.SX.sym(f"x_{i}")) for i in range(3)]
    plain = np.array(symbols, dtype=object)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = MU / (plain + 1.0) ** 3
        categories = [w.category.__name__ for w in caught]
    assert "RuntimeWarning" in categories
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = MU / (plain.view(SXArray) + 1.0) ** 3
        assert [w.category.__name__ for w in caught] == []
