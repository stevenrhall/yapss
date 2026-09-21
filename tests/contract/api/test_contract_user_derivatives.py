"""What a derivative supplied by hand is named, and what a misnamed one says.

Under ``derivatives.method = "user"`` the entries are *named* rather than indexed:
``jacobian.dynamics.v.h`` is the derivative of the rate of change of speed with respect to
altitude. The set of names written is the sparsity structure, which is why writing a different
set on a later call is refused, and why a derivative that happens to vanish at one point is
written as ``0.0`` rather than left out.
"""

from __future__ import annotations

import yapss
from yapss.math import cos, sin

from ._api import not_yet, proposed, raises

G = 32.174


class State(yapss.State):
    """Where the bead is and how fast it is going."""

    y = yapss.scalar()
    v = yapss.scalar()


class Control(yapss.Control):
    """The slope of the wire."""

    theta = yapss.scalar()


class Phases(yapss.Phases):
    """One phase: the fall."""

    fall = yapss.phase(state=State, control=Control)


def build() -> tuple[yapss.Problem, object]:
    """Return a problem whose derivatives are all supplied by hand, and its phase."""
    problem = yapss.Problem("user derivatives", phases=Phases)
    ph = problem.phases.fall

    @ph.register.continuous
    def continuous(arg, out):
        v, theta = arg.state.v, arg.control.theta
        out.dynamics.y = v * sin(theta)
        out.dynamics.v = G * cos(theta)
        return out

    @problem.register.objective
    def objective(arg):
        return arg[ph].final.time

    @ph.register.continuous_jacobian
    def jacobian(arg, jacobian):
        v, theta = arg.state.v, arg.control.theta
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.theta = v * cos(theta)
        jacobian.dynamics.v.theta = -G * sin(theta)
        return jacobian

    @ph.register.continuous_hessian
    def hessian(arg, hessian):
        v, theta = arg.state.v, arg.control.theta
        hessian.dynamics.y.v.theta = cos(theta)
        hessian.dynamics.y.theta.theta = -v * sin(theta)
        hessian.dynamics.v.theta.theta = -G * cos(theta)
        return hessian

    @problem.register.objective_gradient
    def gradient(arg, gradient):
        gradient[gradient.phases[ph].final.time] = 1.0
        return gradient

    @problem.register.objective_hessian
    def objective_hessian(arg, hessian):
        return hessian

    problem.derivatives.method = "user"
    ph.time.initial = (0.0, 0.0)
    ph.time.final = (0.1, 10.0)
    ph.state.y.initial = (0.0, 0.0)
    ph.state.v.initial = (0.0, 0.0)
    ph.state.y.final = (1.0, 1.0)
    ph.state.y.bounds = (0.0, 10.0)
    ph.state.v.bounds = (0.0, 100.0)
    ph.control.theta.bounds = (-1.5, 1.5)
    ph.time.guess = (0.0, 1.0)
    ph.state.y.guess = (0.0, 1.0)
    ph.state.v.guess = (0.0, 5.0)
    ph.control.theta.guess = (0.0, 0.0)
    ph.mesh = yapss.Mesh.uniform(segments=2, points=4)
    problem.ipopt_options.print_level = 0
    return problem, ph


# ------------------------------------------------------------- the derivatives that work


def test_derivatives_written_by_name_solve() -> None:
    """The whole feature, stated once: named entries, and a solve that agrees with tracing."""
    problem, _ = build()
    user = problem.solve()
    assert user.converged

    traced, _ = build()
    traced.derivatives.method = "auto"
    assert abs(user.objective - traced.solve().objective) < 1e-6


def test_a_derivative_that_vanishes_is_written_as_zero() -> None:
    """Writing 0.0 puts the entry in the structure; leaving it out says it is never nonzero."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        v, theta = arg.state.v, arg.control.theta
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.theta = v * cos(theta)
        jacobian.dynamics.v.v = 0.0
        jacobian.dynamics.v.theta = -G * sin(theta)
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    assert problem.solve().converged


# ------------------------------------------------------------------- naming an entry wrong


def test_a_derivative_names_a_variable_that_exists() -> None:
    """Checked against the phase's own variables, with a suggestion."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.y.vv = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(AttributeError, "has no variable 'vv'", at="jacobian.dynamics.y.vv"):
        problem.solve()


def test_a_first_derivative_names_one_variable() -> None:
    """Chaining a second name in the Jacobian is a second derivative in the wrong callback."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.y.v.theta = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        (AttributeError, TypeError),
        "first derivative",
        "hessian",
        at="jacobian.dynamics.y.v.theta",
    ):
        problem.solve()


def test_a_second_derivative_names_two_variables() -> None:
    """One name in the Hessian is half an entry, and the message shows the shape of a whole one."""
    problem, ph = build()

    def hessian(arg, hessian):
        hessian.dynamics.y.v = 1.0
        return hessian

    ph.register.continuous_hessian(hessian, replace=True)
    with raises(
        (AttributeError, TypeError),
        "names one variable",
        "second derivative names two",
        at="hessian.dynamics.y.v",
    ):
        problem.solve()


def test_a_row_alone_is_not_a_derivative() -> None:
    """A derivative names both a row and a variable."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.y = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        (AttributeError, TypeError),
        "names a row but no variable",
        at="jacobian.dynamics.y",
    ):
        problem.solve()


# ----------------------------------------------------------------- writing one entry twice


@proposed("writing one derivative entry twice in a call is accepted today; the last wins")
def test_the_same_derivative_may_not_be_written_twice() -> None:
    """Two statements for one entry is a mistake about which one wins.

    YAPSS refuses two *spellings* of one entry -- the Hessian pair below is the case it was
    written for -- but the same name assigned twice is simply overwritten. Whether that is
    worth catching is not decided: unlike the pair, it is visible on the page.

    The endpoint derivatives behave the same way: `gradient[ph].final.h` written twice is
    accepted too, so whatever is decided here applies to both.
    """
    problem, ph = build()

    def jacobian(arg, jacobian):
        v, theta = arg.state.v, arg.control.theta
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.theta = v * cos(theta)
        jacobian.dynamics.v.theta = -G * sin(theta)
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(ValueError, "the same derivative", "Write each one once", at="dynamics.y.v"):
        problem.solve()


def test_a_hessian_pair_may_not_be_written_both_ways_round() -> None:
    """``.a.b`` and ``.b.a`` are one entry; writing both is refused rather than summed."""
    problem, ph = build()

    def hessian(arg, hessian):
        v, theta = arg.state.v, arg.control.theta
        hessian.dynamics.y.v.theta = cos(theta)
        hessian.dynamics.y.theta.v = cos(theta)
        hessian.dynamics.y.theta.theta = -v * sin(theta)
        hessian.dynamics.v.theta.theta = -G * cos(theta)
        return hessian

    ph.register.continuous_hessian(hessian, replace=True)
    with raises(ValueError, "both ways round", "unordered pair once", at="theta.v"):
        problem.solve()


# ------------------------------------------------- the structure must not vary from call to call


def test_the_same_names_must_be_written_on_every_call() -> None:
    """What is written is the sparsity structure, so it cannot depend on the values."""
    problem, ph = build()
    calls = []

    def jacobian(arg, jacobian):
        v, theta = arg.state.v, arg.control.theta
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.theta = v * cos(theta)
        jacobian.dynamics.v.theta = -G * sin(theta)
        calls.append(1)
        if len(calls) > 1:
            jacobian.dynamics.v.v = 0.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        ValueError,
        "the same names must be written on every call",
        "jacobian.dynamics.v.v",
        at="problem.solve()",
    ):
        problem.solve()


# ------------------------------------------------------------- the endpoint derivatives


def test_an_endpoint_variable_is_a_value() -> None:
    """An endpoint variable is four coordinates, so it is a value and not a path of names.

    Binding the end is what keeps the cost of naming one down to a single dot, and it is what
    lets the same variable be written twice, kept in a list, or built in a loop.
    """
    problem, ph = build()

    def gradient(arg, gradient):
        final = gradient.phases[ph].final
        gradient[final.time] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    assert problem.solve().converged


def test_an_endpoint_derivative_names_an_end() -> None:
    """The gradient of the objective is by a variable at an end of a phase."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases[problem.phases.fall].middle.time] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        AttributeError,
        "an endpoint variable is at 'initial', 'final' or 'integral'",
        at="gradient.phases[problem.phases.fall].middle",
    ):
        problem.solve()


def test_an_endpoint_gradient_takes_a_phase_handle() -> None:
    """The same rule as `arg[ph]`: a phase is named by its handle."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases["fall"].final.time] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(KeyError, "takes a phase handle", at='gradient.phases["fall"]'):
        problem.solve()


def test_a_variable_is_not_a_derivative() -> None:
    """Assigning to a variable says nothing about what the derivative is of."""
    problem, ph = build()

    def gradient(arg, gradient):
        gradient.phases[ph].final.time = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        AttributeError,
        "is a variable, not a derivative",
        at="gradient.phases[ph].final.time = 1.0",
    ):
        problem.solve()


def test_a_variable_of_another_problem_is_refused() -> None:
    """A variable carries its own namespace, so it cannot land in the wrong problem.

    Without the check the key would still be structurally valid, so the entry would go to the
    wrong column and cost iterations rather than raise -- the failure this surface exists to
    prevent.
    """
    problem, ph = build()
    elsewhere, elsewhere_ph = build()
    stolen = []

    def lend(arg, gradient):
        stolen.append(gradient.phases[elsewhere_ph].final.time)
        gradient[gradient.phases[elsewhere_ph].final.time] = 1.0
        return gradient

    elsewhere.register.objective_gradient(lend, replace=True)
    elsewhere.solve()

    def gradient(arg, gradient):
        gradient[stolen[0]] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(ValueError, "belongs to another problem", at="gradient[stolen[0]] = 1.0"):
        problem.solve()


def test_a_missing_derivative_callback_is_refused_before_the_solve() -> None:
    """Under "user", every derivative is registered, including the ones that are zero."""
    bare = yapss.Problem("bare", phases=Phases)
    bare_ph = bare.phases.fall

    @bare_ph.register.continuous
    def continuous(arg, out):
        out.dynamics.y = arg.state.v
        out.dynamics.v = 0.0
        return out

    @bare.register.objective
    def objective(arg):
        return arg[bare_ph].final.time

    bare.derivatives.method = "user"
    bare_ph.time.guess = (0.0, 1.0)
    with raises(
        ValueError,
        "has no continuous_jacobian callback",
        "@ph.register.continuous_jacobian",
        at="validate",
    ):
        bare.validate()


@not_yet("message", "the varying-structure message says 'callback' twice")
def test_the_varying_structure_message_reads_cleanly() -> None:
    """It currently reads "the jacobian callback for phase 'fall' callback wrote ...".

    The phrase is assembled from a label that already ends in "callback" and a fragment that
    begins with one. Nothing is wrong with what it says; it is the first message a user of
    hand-written derivatives is likely to meet, and it should not stutter.
    """
    problem, ph = build()
    calls = []

    def jacobian(arg, jacobian):
        v, theta = arg.state.v, arg.control.theta
        jacobian.dynamics.y.v = sin(theta)
        jacobian.dynamics.y.theta = v * cos(theta)
        jacobian.dynamics.v.theta = -G * sin(theta)
        calls.append(1)
        if len(calls) > 1:
            jacobian.dynamics.v.v = 0.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(ValueError, "phase 'fall' wrote", at="problem.solve()"):
        problem.solve()
