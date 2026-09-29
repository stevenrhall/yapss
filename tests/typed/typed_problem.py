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

from typing import Any, Final

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


class Phase(yapss.Phase):
    """The bead's descent."""

    state: State
    control: Control
    path: Path
    integral: Integral


class Phases(yapss.Phases):
    """One phase: the bead slides."""

    phase: Phase


class Brachistochrone(yapss.Problem):
    """The problem, declared as a class: its annotations are what a checker reads."""

    phases: Phases
    discrete: Discrete
    parameter: Parameter


def dynamics(arg: yapss.ContinuousArg[State, Control, Parameter], out: State) -> None:
    """Fill the dynamics: a helper takes the output vector itself, typed as the state."""
    v, u = arg.state.v, arg.control.u
    out.x = v * cos(u)
    out.y = v * sin(u)
    out.v = arg.parameter.g * sin(u)


def setup() -> Brachistochrone:
    """Set up the problem, typed by its class."""
    problem = Brachistochrone("Brachistochrone")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(
        arg: yapss.ContinuousArg[State, Control, Parameter],
        out: yapss.ContinuousOut[State, Path, Integral],
    ) -> None:
        dynamics(arg, out.dynamics)
        out.path.speed = arg.state.v
        out.integrand.distance = arg.state.v

    @problem.register.objective
    def objective(arg: yapss.DiscreteArg[Parameter]) -> Any:
        return arg[ph].final_time

    @problem.register.discrete
    def discrete(arg: yapss.DiscreteArg[Parameter], out: yapss.DiscreteOut[Discrete]) -> None:
        out.discrete.landing = arg[ph].final_state.x

    # the decorator form taking options, which strict mode reports if it returns `Any`
    @problem.register.objective
    def objective_again(arg: yapss.DiscreteArg[Parameter]) -> Any:
        return arg[ph].final_time + 0.0 * arg[ph].integral.distance

    ph.time.initial = (0.0, 0.0)
    ph.duration.bounds = (0.0, None)
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
    ph.state.v.scale = 10.0
    ph.dynamics.v.scale = None  # YAPSS chooses: the state's scale
    ph.dynamics.x.scale = 2.0

    ph.time.guess = (0.0, 1.0)
    ph.state.x.guess = (0, 1)
    ph.state.y.guess = (0, 1)
    ph.state.v.guess = (0, 5)
    problem.parameter.g.guess = 32.174

    problem.ipopt_options.print_level = 0
    problem.ipopt_options.max_iter = None  # None deletes an option: Ipopt's default
    problem.comment = "the typed brachistochrone"
    return problem


# The alias a PyCharm user writes once. PyCharm completes from annotations, and it does not follow
# `solution.phases[ph]` to the phase's vectors, as mypy and pyright do.
Solved = yapss.PhaseSolution[State, Control, Path, Integral]


def report(solution: yapss.Solution[Discrete, Parameter], ps: Solved) -> dict[str, Any]:
    """Read a solution through every tree, each checked down to the field."""
    nlp = solution.nlp
    var, con = ps.nlp.index.variable, ps.nlp.index.constraint
    return {
        "objective": solution.objective,
        "comment": solution.settings.comment,
        "method": solution.settings.spectral_method,
        "solve time": solution.run.seconds.total,
        "ipopt version": solution.run.ipopt_version,
        "warnings": solution.run.warnings,
        "gravity": solution.parameter.g,
        "landing multiplier": solution.multiplier.discrete.landing,
        "landing": ps.final_state.x,
        "start": ps.initial_state.x,
        "final time": ps.final_time,
        "final time multiplier": ps.multiplier.final_time,
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
    ps = solution.phases[problem.phases.phase]
    return report(solution, ps)


class Radius(yapss.State):
    """A state for a phase that runs over a radius."""

    y = yapss.scalar()


class Nose(yapss.Phase):
    """A phase that runs over a radius, which is its `time`."""

    state: Radius


def over_radius(arg: yapss.ContinuousArg[Radius], out: yapss.ContinuousOut[Radius]) -> None:
    """Read the independent variable, `time` whatever it measures; any other name is reported."""
    out.dynamics.y = arg.time
    out.dynamics.y = arg.r  # type: ignore[attr-defined]


class NosePhases(yapss.Phases):
    """One phase, which runs over a radius."""

    nose: Nose


class NoseProblem(yapss.Problem):
    """A problem of that one phase."""

    phases: NosePhases


def independent_mistakes(problem: NoseProblem) -> None:
    """Every phase's independent variable is `time`; another name is an ordinary misspelling."""
    problem.phases.nose.time.guess = (0.0, 1.0)
    problem.phases.nose.r  # type: ignore[attr-defined]
    problem.solve().phases[problem.phases.nose].r  # type: ignore[attr-defined]
    problem.solve().phases[problem.phases.nose].hamiltonain  # type: ignore[attr-defined]


def setup_with(
    method: yapss.SpectralMethod = "lgl", sense: yapss.ObjectiveSense = "minimize"
) -> Brachistochrone:
    """Pass an option's value through a variable: annotated with the alias, it checks."""
    problem = setup()
    problem.spectral_method = method
    problem.objective.sense = sense
    order: yapss.DerivativeOrder = "first"
    how: yapss.DerivativeMethod = "central-difference"
    problem.derivatives.order = order
    problem.derivatives.method = how
    return problem


METHOD: Final = "lgr"
"""A module constant keeps its literal type when declared `Final` (PEP 586)."""


def options_through_variables() -> None:
    """A constant declared Final passes; one typed only as a string is reported."""
    setup_with(METHOD)
    loose: str = "lgr"
    setup_with(loose)  # type: ignore[arg-type]
    setup().spectral_method = loose  # type: ignore[assignment]
    setup().spectral_method = "lq"  # type: ignore[assignment]


def mistakes(
    problem: Brachistochrone,
    arg: yapss.ContinuousArg[State, Control, Parameter],
    out: yapss.ContinuousOut[State, Path, Integral],
    endpoint: yapss.DiscreteArg[Parameter],
    discrete: yapss.DiscreteOut[Discrete],
) -> None:
    """Each line is a mistake the checker must report, asserted by its ignore."""
    ph = problem.phases.phase
    problem.phases.phas  # type: ignore[attr-defined]
    ph.stat  # type: ignore[attr-defined]
    ph.state.xx.bounds = (0, 1)  # type: ignore[attr-defined]
    ph.state.x.bond = (0, 1)  # type: ignore[attr-defined]
    ph.state.x.bounds = 5.0  # type: ignore[assignment]
    ph.duration.bound = (0, 1)  # type: ignore[attr-defined]
    ph.duration.bounds = (None, 1.0)  # type: ignore[assignment]
    ph.duration = (0, 1)  # type: ignore[assignment]
    arg.state.xx  # type: ignore[attr-defined]
    arg.control.v  # type: ignore[attr-defined]
    arg.parameter.gg  # type: ignore[attr-defined]
    out.dynamic.x = 0.0  # type: ignore[attr-defined]
    out.dynamics.xx = 0.0  # type: ignore[attr-defined]
    out.path.sped = 0.0  # type: ignore[attr-defined]
    out.integrand.distanse = 0.0  # type: ignore[attr-defined]
    endpoint.parameter.gg  # type: ignore[attr-defined]
    endpoint[ph].integral.distanse  # type: ignore[attr-defined]
    endpoint[ph].final_state.xx  # type: ignore[attr-defined]
    endpoint[ph].final_state.time  # type: ignore[attr-defined]
    endpoint[ph].final_tme  # type: ignore[attr-defined]
    endpoint[ph].final  # type: ignore[attr-defined]
    endpoint["phase"]  # type: ignore[index]
    discrete.discrete.landin = 0.0  # type: ignore[attr-defined]
    arg.stat = arg.state  # type: ignore[attr-defined]
    out.dynamic = out.dynamics  # type: ignore[attr-defined]
    problem.ipopt_options.max_iters = 5000  # type: ignore[attr-defined]
    problem.comment = 3  # type: ignore[assignment]
    problem.ipopt_options.max_iter = "5000"  # type: ignore[assignment]
    problem.ipopt_options.hessian_approximation = "exact"  # type: ignore[attr-defined]


def registration_mistakes(problem: Brachistochrone) -> None:
    """A continuous or discrete callback fills `out` and returns nothing; one annotated to
    return something is reported, by the decorator and by the call alike."""
    ph = problem.phases.phase

    def returns_out(
        arg: yapss.ContinuousArg[State, Control], out: yapss.ContinuousOut[State, Path, Integral]
    ) -> yapss.ContinuousOut[State, Path, Integral]:
        return out

    def returns_rows(arg: yapss.DiscreteArg, out: yapss.DiscreteOut[Discrete]) -> float:
        return 0.0

    ph.register.continuous(returns_out)  # type: ignore[type-var]
    problem.register.discrete(returns_rows)  # type: ignore[type-var]

    @ph.register.continuous  # type: ignore[type-var]
    def decorated(
        arg: yapss.ContinuousArg[State, Control], out: yapss.ContinuousOut[State, Path, Integral]
    ) -> yapss.ContinuousOut[State, Path, Integral]:
        return out


def solution_mistakes(problem: Brachistochrone) -> None:
    """Each line is a mistake in reading a solution that the checker must report."""
    solution = problem.solve()
    ps = solution.phases[problem.phases.phase]
    solution.objectiv  # type: ignore[attr-defined]
    solution.objectiv = 0.0  # type: ignore[attr-defined]
    ps.hamiltonain = ps.hamiltonian  # type: ignore[attr-defined]
    solution.settings.coment = ""  # type: ignore[attr-defined]
    solution.settings.coment  # type: ignore[attr-defined]
    solution.run.secnds  # type: ignore[attr-defined]
    solution.run.seconds.totl  # type: ignore[attr-defined]
    solution.nlp.version  # type: ignore[attr-defined]
    solution.parameter.gg  # type: ignore[attr-defined]
    solution.multiplier.discrete.landin  # type: ignore[attr-defined]
    solution.multiplier.dynamics  # type: ignore[attr-defined]
    ps.state.xx  # type: ignore[attr-defined]
    ps.final_state.xx  # type: ignore[attr-defined]
    ps.initial_state.time  # type: ignore[attr-defined]
    ps.final  # type: ignore[attr-defined]
    ps.multiplier.final  # type: ignore[attr-defined]
    ps.costate.xx  # type: ignore[attr-defined]
    ps.control.v  # type: ignore[attr-defined]
    ps.multiplier.path.sped  # type: ignore[attr-defined]
    ps.multiplier.integral_defect.distanse  # type: ignore[attr-defined]
    ps.multiplier.integral  # type: ignore[attr-defined]
    ps.multiplier.state.xx  # type: ignore[attr-defined]
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
    """A problem that declares it has no parameters reports reading one, from its solution too.

    Declared, not omitted: an omitted member is `Any`, and a checker reports nothing on it.
    """

    class NoParameters(yapss.Problem):
        phases: Phases
        parameter: yapss.Parameter

    problem = NoParameters("none")
    problem.solve().parameter.g  # type: ignore[attr-defined]


def any_problem(problem: yapss.Problem) -> Any:
    """The bare annotation cannot say what was declared, so its solution answers any name."""
    return problem.solve().parameter.anything


def wrong_role(arg: yapss.ContinuousArg[Control]) -> None:  # type: ignore[type-var]
    """A control named where the state belongs is reported: each parameter is bounded by its role."""
