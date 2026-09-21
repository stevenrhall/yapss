"""What the 0.4.0 clauses are written against.

The declarations here are the subject of most of the suite: one problem with at least one of
every kind of vector, a scalar field and a block field side by side in each, and two phases so
that a message naming a phase can be checked for naming the right one.

Nothing here is a fixture. A clause that needs a problem builds one, because half of what the
suite states is what happens *during* construction, and a shared object would have to be built
before the clause could say what building it does.

The vocabulary -- `raises`, `warns`, `not_yet`, `proposed` -- is the same as the 0.3.0 suite's
and comes from the same module, so the two read alike and the catalogue is rendered from both
by one renderer. What differs is only what the clauses are about.
"""

from __future__ import annotations

from typing import Any

import yapss
from yapss.math import cos, sin

from .._harness import not_yet, proposed, raises, warns

__all__ = [
    "Control",
    "Discrete",
    "Integral",
    "Parameter",
    "Path",
    "Phases",
    "State",
    "callback_problem",
    "not_yet",
    "problem",
    "proposed",
    "raises",
    "solvable",
    "warns",
]

G0 = 32.174
"""Gravitational acceleration, for the small solvable problem below."""


class State(yapss.State):
    """A scalar row and a block of two, so both shapes are in every clause's reach."""

    x = yapss.scalar()
    """A scalar state."""
    y = yapss.vector(2)
    """A block state of two rows."""


class Control(yapss.Control):
    """One of each, as the state has."""

    u = yapss.scalar()
    """A scalar control."""
    w = yapss.vector(2)
    """A block control of two rows."""


class Path(yapss.Path):
    """One path constraint of each shape."""

    g = yapss.scalar()
    """A scalar path constraint."""
    h = yapss.vector(2)
    """A block path constraint of two rows."""


class Integral(yapss.Integral):
    """One integral of each shape."""

    q = yapss.scalar()
    """A scalar integral."""
    r = yapss.vector(2)
    """A block integral of two rows."""


class Discrete(yapss.Discrete):
    """One discrete constraint of each shape."""

    d = yapss.scalar()
    """A scalar discrete constraint."""
    e = yapss.vector(2)
    """A block discrete constraint of two rows."""


class Parameter(yapss.Parameter):
    """One parameter of each shape."""

    s = yapss.scalar()
    """A scalar parameter."""
    t = yapss.vector(2)
    """A block parameter of two rows."""


class Phases(yapss.Phases):
    """Two phases, so that a message naming one can be checked for naming the right one."""

    first = yapss.phase(state=State, control=Control, path=Path, integral=Integral)
    second = yapss.phase(state=State, control=Control, path=Path, integral=Integral)


def problem() -> Any:
    """Return a problem with one of everything, declared and otherwise untouched."""
    return yapss.Problem("contract", phases=Phases, discrete=Discrete, parameter=Parameter)


# ------------------------------------------------------------------ a problem that solves

# The brachistochrone, with a path constraint on speed, an integral of the control squared and
# a discrete constraint on the final height, so that every output kind is exercised. Three
# segments of four points keep each solve to milliseconds.


class Slide(yapss.State):
    """The bead's position and speed."""

    x = yapss.scalar()
    """Horizontal position."""
    y = yapss.scalar()
    """Vertical drop."""
    v = yapss.scalar()
    """Speed."""


class Angle(yapss.Control):
    """The slope of the wire."""

    theta = yapss.scalar()
    """Path angle."""


class Speed(yapss.Path):
    """A path constraint that is always satisfied, so the solve is not about it."""

    speed = yapss.scalar()
    """The bead's speed, bounded loosely."""


class Effort(yapss.Integral):
    """What is accumulated along the slide."""

    effort = yapss.scalar()
    """The integral of the control squared."""


class Height(yapss.Discrete):
    """One discrete constraint, on where the bead ends up."""

    drop = yapss.scalar()
    """The final depth."""


class OnePhase(yapss.Phases):
    """One phase: the slide."""

    slide = yapss.phase(state=Slide, control=Angle, path=Speed, integral=Effort)


def solvable(method: str = "auto") -> Any:
    """Return the small solvable problem, with the given derivative method.

    Every callback is registered and every aspect set, so a clause can take this problem,
    break exactly one thing, and be sure that what it sees is what it broke.
    """
    problem = yapss.Problem("callbacks", phases=OnePhase, discrete=Height)
    ph = problem.phases.slide

    @ph.register.continuous
    def continuous(arg, out):
        """Slide down the wire."""
        v, theta = arg.state.v, arg.control.theta
        out.dynamics.x = v * cos(theta)
        out.dynamics.y = v * sin(theta)
        out.dynamics.v = G0 * sin(theta)
        out.path.speed = v
        out.integrand.effort = theta**2
        return out

    @problem.register.objective
    def objective(arg):
        """Reach the far end quickly, with a little regard for effort."""
        return arg[ph].final.time + 1e-3 * arg[ph].integral.effort

    @problem.register.discrete
    def discrete(arg, out):
        """Report the final depth, which is constrained."""
        out.discrete.drop = arg[ph].final.y
        return out

    ph.time.initial = (0.0, 0.0)
    ph.state.x.initial = (0.0, 0.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.x.final = (1.0, 1.0)
    ph.state.x.bounds = (0.0, 10.0)
    ph.state.y.bounds = (0.0, 10.0)
    ph.state.v.bounds = (0.0, 10.0)
    ph.control.theta.bounds = (-1.6, 1.6)
    ph.path.speed.bounds = (0.0, 100.0)
    problem.discrete.drop.bounds = (0.5, 0.5)

    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0.0, 1.0)
    ph.state.y.guess = (0.0, 0.5)
    ph.state.v.guess = (0.0, 5.0)
    ph.control.theta.guess = (0.0, 0.0)

    ph.mesh = yapss.Mesh.uniform(segments=3, points=4)
    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    return problem


def callback_problem(method: str = "auto") -> Any:
    """Return the solvable problem; the name the 0.3.0 suite uses for the same thing."""
    return solvable(method)
