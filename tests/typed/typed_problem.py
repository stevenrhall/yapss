"""A problem written with every annotation a user can write, checked by mypy in strict mode.

Two things are guarded here, and neither by pytest. The first is that correct, fully annotated
code passes the package's strict mypy configuration with no ``# type: ignore``: every tox mypy
environment checks this directory, so a type narrower than the runtime fails CI. The second is
that mistakes are caught. Each line in `mistakes` carries an ignore for the error it must
raise, and the configuration sets ``warn_unused_ignores``, so an ignore that suppresses nothing
-- a mistake that stopped being caught -- is itself an error.

`test_typed_problem.py` solves the problem, so the annotations are also exercised at run time:
without ``from __future__ import annotations`` each one is evaluated where it is written, and
``yapss.ContinuousArg[State, Control]`` has to be a subscriptable class for that to work.
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
    time: yapss.Independent


class Phases(yapss.Phases):
    """One phase: the bead slides."""

    slide: Slide


def dynamics(arg: yapss.ContinuousArg[State, Control, Parameter], out: State) -> None:
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
        arg: yapss.ContinuousArg[State, Control, Parameter],
        out: yapss.ContinuousOut[State, Path, Integral],
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
    @problem.register.objective
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
    problem.ipopt_options.max_iter = None  # None deletes an option: Ipopt's default
    return problem


# The alias a PyCharm user writes once: its engine does not follow `solution[ph]` to the phase's
# vectors, which mypy does, so the variable is annotated instead.
SlideSolution = yapss.PhaseSolution[State, Control, Path, Integral]


def report(solution: yapss.Solution[Discrete, Parameter], ps: SlideSolution) -> dict[str, Any]:
    """Read a solution through every tree, each checked down to the field."""
    nlp = solution.nlp
    var, con = ps.nlp.index.variable, ps.nlp.index.constraint
    return {
        "objective": solution.objective,
        "gravity": solution.parameter.g,
        "landing multiplier": solution.multiplier.discrete.landing,
        "landing": ps.state.x[-1],  # the typed read of `ps.final.x`
        "costate": ps.costate.v,
        "same costate": ps.multiplier.dynamics.v,
        "slope": ps.control.u,
        "speed multiplier": ps.multiplier.path.speed,
        "distance": ps.integral.distance,
        "defects": nlp.g[con.dynamics.x],
        "raw speed multiplier": nlp.mult_g[con.path.speed],
        "slope gradient": nlp.grad_f[var.control.u],
        "gravity position": nlp.index.variable.parameter.g,
        "landing position": nlp.index.constraint.discrete.landing,
        "defect points": ps.time[ps.nlp.point.dynamics],
        "iterations": nlp.convergence.iterations,
        "jacobian": (nlp.jac_g.row, nlp.jac_g.col, nlp.jac_g.value),
        "collocated": ps.hamiltonian[ps.collocated],
    }


def solve_and_report() -> dict[str, Any]:
    """Solve and read, typed with no annotation: the solution from the problem, the phase from
    its handle."""
    problem = setup()
    solution = problem.solve()
    ps = solution[problem.phases.slide]
    return report(solution, ps)


class Radius(yapss.State):
    """A state for a phase that runs over a radius."""

    y = yapss.scalar()


class Nose(yapss.Phase):
    """A phase whose independent variable is not time."""

    state: Radius
    r: yapss.Independent


def over_radius(arg: yapss.ContinuousArg[Radius], out: yapss.ContinuousOut[Radius]) -> None:
    """Read the independent variable by the name the phase gave it, which must check."""
    out.dynamics.y = arg.r


class NosePhases(yapss.Phases):
    """One phase, which runs over a radius."""

    nose: Nose


def independent_mistakes(problem: yapss.Problem[NosePhases]) -> None:
    """A phase has the independent variable it named, and no other.

    `time` is not defaulted, so nothing declares it here, and reaching for it is the ordinary
    misspelling every other line in `mistakes` is. While `time` was supplied by the base class
    this line checked, and failed only when it ran.
    """
    problem.phases.nose.time  # type: ignore[attr-defined]
    problem.phases.nose.r.guess = (0.0, 1.0)


def mistakes(
    problem: yapss.Problem[Phases, Discrete, Parameter],
    arg: yapss.ContinuousArg[State, Control, Parameter],
    out: yapss.ContinuousOut[State, Path, Integral],
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
    problem.ipopt_options.max_iters = 5000  # type: ignore[attr-defined]
    problem.ipopt_options.max_iter = "5000"  # type: ignore[assignment]
    problem.ipopt_options.hessian_approximation = "exact"  # type: ignore[attr-defined]


def solution_mistakes(problem: yapss.Problem[Phases, Discrete, Parameter]) -> None:
    """Each line is a mistake in reading a solution that the checker must report."""
    solution = problem.solve()
    ps = solution[problem.phases.slide]
    solution.objectiv  # type: ignore[attr-defined]
    solution.parameter.gg  # type: ignore[attr-defined]
    solution.multiplier.discrete.landin  # type: ignore[attr-defined]
    solution.multiplier.dynamics  # type: ignore[attr-defined]
    ps.state.xx  # type: ignore[attr-defined]
    ps.costate.xx  # type: ignore[attr-defined]
    ps.control.v  # type: ignore[attr-defined]
    ps.multiplier.path.sped  # type: ignore[attr-defined]
    ps.multiplier.integral.distanse  # type: ignore[attr-defined]
    ps.multiplier.dynamic  # type: ignore[attr-defined]
    ps.nlp.index.variable.state.xx  # type: ignore[attr-defined]
    ps.nlp.index.constraint.dynamics.xx  # type: ignore[attr-defined]
    ps.nlp.index.constraint.path.sped  # type: ignore[attr-defined]
    ps.nlp.points  # type: ignore[attr-defined]
    solution.nlp.index.variable.parameter.gg  # type: ignore[attr-defined]
    solution.nlp.convergence.inf_prr  # type: ignore[attr-defined]
    solution.nlp.jac_g.rows  # type: ignore[attr-defined]
    solution.nlp.mult_gg  # type: ignore[attr-defined]


def no_parameters_declared() -> None:
    """A problem that declares no parameters reports reading one, from its solution too."""
    problem = yapss.Problem("none", phases=Phases)
    problem.solve().parameter.g  # type: ignore[attr-defined]


def any_problem(problem: yapss.Problem) -> Any:
    """The bare annotation cannot say what was declared, so its solution answers any name."""
    return problem.solve().parameter.anything


def wrong_role(arg: yapss.ContinuousArg[Control]) -> None:  # type: ignore[type-var]
    """A control named where the state belongs is reported: each parameter is bounded by its role."""


def swapped() -> None:
    """A role swap is an incompatible override of the base class's annotation.

    Inside a function because the runtime refuses the class where it is defined.
    """

    class Swapped(yapss.Phase):
        state: Control  # type: ignore[assignment]
        time: yapss.Independent
