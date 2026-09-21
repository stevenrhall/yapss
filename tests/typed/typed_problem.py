"""A problem written with every annotation a user can write, checked by mypy in strict mode.

Two things are guarded here, and neither by pytest. The first is that correct, fully annotated
code passes the package's strict mypy configuration with no ``# type: ignore``: every tox mypy
environment checks this directory, so a type narrower than the runtime fails CI. The second is
that mistakes are caught. Each line in `mistakes` carries an ignore for the error it must
raise, and the configuration sets ``warn_unused_ignores``, so an ignore that suppresses nothing
-- a mistake that stopped being caught -- is itself an error.

`test_typed_problem.py` solves the problem, so the annotations are also exercised at run time:
without ``from __future__ import annotations`` each one is evaluated where it is written, and
``yapss.ContinuousArg[Slide]`` has to be a subscriptable class for that to work.
"""

from typing import Any

from numpy import pi

import yapss
from yapss.math import cos, sin


class State(yapss.State):
    """Where the bead is and how fast it is going."""

    x = yapss.scalar()
    y = yapss.scalar()
    v = yapss.scalar()


class Control(yapss.Control):
    """The slope of the path."""

    u = yapss.scalar()


class Path(yapss.Path):
    """A limit on the speed, which the solution never reaches."""

    speed = yapss.scalar()


class Integral(yapss.Integral):
    """The distance travelled, which is only recorded."""

    distance = yapss.scalar()


class Parameter(yapss.Parameter):
    """Gravity, held fixed by its bounds, so that a parameter is read in two callbacks."""

    g = yapss.scalar()


class Discrete(yapss.Discrete):
    """Where the bead must land."""

    landing = yapss.scalar()


class Slide(yapss.Phase):
    """The bead's descent."""

    state: State
    control: Control
    path: Path
    integral: Integral


class Phases(yapss.Phases):
    """One phase: the bead slides."""

    slide: Slide


def dynamics(arg: yapss.ContinuousArg[Slide, Parameter], out: State) -> None:
    """Fill the dynamics: a helper takes the output vector itself, typed as the state."""
    v, u = arg.state.v, arg.control.u
    out.x = v * cos(u)
    out.y = v * sin(u)
    out.v = arg.parameter.g * sin(u)


def setup() -> yapss.Problem[Phases, Discrete, Parameter]:
    """Set up the problem, typed with the declarations it was built from."""
    problem = yapss.Problem("typed", phases=Phases, discrete=Discrete, parameter=Parameter)
    ph = problem.phases.slide

    @ph.register.continuous
    def continuous(
        arg: yapss.ContinuousArg[Slide, Parameter], out: yapss.ContinuousOut[Slide]
    ) -> None:
        dynamics(arg, out.dynamics)
        out.path.speed = arg.state.v
        out.integrand.distance = arg.state.v

    @problem.register.objective
    def objective(arg: yapss.EndpointArg[Parameter]) -> Any:
        return arg[ph].final.time

    @problem.register.discrete
    def discrete(arg: yapss.EndpointArg[Parameter], out: yapss.DiscreteOut[Discrete]) -> None:
        out.discrete.landing = arg[ph].final.x

    # the decorator form taking options, which strict mode reports if it returns `Any`
    @problem.register.objective(replace=True)
    def objective_again(arg: yapss.EndpointArg[Parameter]) -> Any:
        return arg[ph].final.time + 0.0 * arg[ph].integral.distance

    ph.time.initial = (0.0, 0.0)
    ph.state.x.initial = (0.0, 0.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.x.bounds = (0, 10)
    ph.state.y.bounds = (0, 10)
    ph.state.v.bounds = (0, 10)
    ph.control.u.bounds = (-pi / 2, pi / 2)
    ph.path.speed.bounds = (None, 100.0)
    ph.integral.distance.bounds = (0.0, None)
    problem.parameter.g.bounds = (32.174, 32.174)
    problem.discrete.landing.bounds = (1.0, 1.0)

    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0, 1)
    ph.state.y.guess = (0, 1)
    ph.state.v.guess = (0, 5)
    problem.parameter.g.guess = 32.174

    problem.ipopt_options.print_level = 0
    return problem


class Radius(yapss.State):
    """A state for a phase that runs over a radius."""

    y = yapss.scalar()


class Nose(yapss.Phase):
    """A phase whose independent variable is not time."""

    state: Radius
    r: yapss.Independent


def over_radius(arg: yapss.ContinuousArg[Nose], out: yapss.ContinuousOut[Nose]) -> None:
    """Read the independent variable by the name the phase gave it, which must check."""
    out.dynamics.y = arg.r


def mistakes(
    problem: yapss.Problem[Phases, Discrete, Parameter],
    arg: yapss.ContinuousArg[Slide, Parameter],
    out: yapss.ContinuousOut[Slide],
    endpoint: yapss.EndpointArg[Parameter],
    discrete: yapss.DiscreteOut[Discrete],
) -> None:
    """Each line is a mistake the checker must report, asserted by its ignore."""
    ph = problem.phases.slide
    problem.phases.slid  # type: ignore[attr-defined]
    ph.stat  # type: ignore[attr-defined]
    ph.state.xx.bounds = (0, 1)  # type: ignore[attr-defined]
    ph.state.x.bond = (0, 1)  # type: ignore[attr-defined]
    ph.state.x.bounds = 5.0  # type: ignore[assignment]
    arg.state.xx  # type: ignore[attr-defined]
    arg.control.v  # type: ignore[attr-defined]
    arg.parameter.gg  # type: ignore[attr-defined]
    out.dynamic.x = 0.0  # type: ignore[attr-defined]
    out.dynamics.xx = 0.0  # type: ignore[attr-defined]
    out.path.sped = 0.0  # type: ignore[attr-defined]
    out.integrand.distanse = 0.0  # type: ignore[attr-defined]
    endpoint.parameter.gg  # type: ignore[attr-defined]
    endpoint[ph].integral.distanse  # type: ignore[attr-defined]
    endpoint["slide"]  # type: ignore[index]
    discrete.discrete.landin = 0.0  # type: ignore[attr-defined]


def swapped() -> None:
    """A role swap is an incompatible override of the base class's annotation.

    Inside a function because the runtime refuses the class where it is defined.
    """

    class Swapped(yapss.Phase):
        state: Control  # type: ignore[assignment]
