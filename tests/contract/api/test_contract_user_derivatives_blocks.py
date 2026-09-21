"""Hand-written derivatives of a problem with block fields, parameters and endpoints.

The companion of `test_contract_user_derivatives`, which states the rules on scalar fields.
Everything here is about the two things that make a spelling longer: a *row* of a block field,
and a variable that belongs to an *endpoint* of a phase rather than to a point along it.
"""

from __future__ import annotations

import yapss

from ._api import not_yet, raises

AREA = "user_derivatives"
"""These clauses belong with the rest of the user-derivative ones, not in a section of their own."""

A = 0.7


class State(yapss.State):
    """A scalar row and a block of two."""

    h = yapss.scalar()
    r = yapss.vector(2)


class Control(yapss.Control):
    """A block control of two rows."""

    u = yapss.vector(2)


class Integral(yapss.Integral):
    """What is accumulated along the phase."""

    cost = yapss.scalar()


class Parameter(yapss.Parameter):
    """A scalar parameter and a block of two."""

    k = yapss.scalar()
    m = yapss.vector(2)


class Discrete(yapss.Discrete):
    """One constraint, closing a row of the state."""

    close = yapss.scalar()


class Phases(yapss.Phases):
    """One phase."""

    run = yapss.phase(state=State, control=Control, integral=Integral)


def build():
    """Return a problem whose every derivative is written by hand, and its phase."""
    problem = yapss.Problem("blocks", phases=Phases, parameter=Parameter, discrete=Discrete)
    ph = problem.phases.run

    @ph.register.continuous
    def continuous(arg, out):
        u = arg.control.u
        k = arg.parameter.k
        out.dynamics.h = k * u[0]
        out.dynamics.r = [u[1], u[1]]
        out.integrand.cost = A * (u[0] ** 2 + u[1] ** 2)
        return out

    @ph.register.continuous_jacobian
    def jacobian(arg, jacobian):
        u = arg.control.u
        jacobian.dynamics.h.u[0] = arg.parameter.k
        jacobian.dynamics.h.k = u[0]
        jacobian.dynamics.r[0].u[1] = 1.0
        jacobian.dynamics.r[1].u[1] = 1.0
        jacobian.integrand.cost.u[0] = 2 * A * u[0]
        jacobian.integrand.cost.u[1] = 2 * A * u[1]
        return jacobian

    @ph.register.continuous_hessian
    def hessian(arg, hessian):
        hessian.dynamics.h.u[0].k = 1.0
        hessian.integrand.cost.u[0].u[0] = 2 * A
        hessian.integrand.cost.u[1].u[1] = 2 * A
        return hessian

    @problem.register.objective
    def objective(arg):
        return arg[ph].integral.cost + arg[ph].final.h

    @problem.register.objective_gradient
    def gradient(arg, gradient):
        run = gradient.phases[ph]
        gradient[run.integral.cost] = 1.0
        gradient[run.final.h] = 1.0
        return gradient

    @problem.register.objective_hessian
    def objective_hessian(arg, hessian):
        return hessian

    @problem.register.discrete
    def discrete(arg, out):
        out.discrete.close = arg[ph].final.r[0] - arg[ph].initial.r[0]
        return out

    @problem.register.discrete_jacobian
    def discrete_jacobian(arg, jacobian):
        run = jacobian.phases[ph]
        jacobian.discrete.close[run.final.r[0]] = 1.0
        jacobian.discrete.close[run.initial.r[0]] = -1.0
        return jacobian

    @problem.register.discrete_hessian
    def discrete_hessian(arg, hessian):
        return hessian

    problem.derivatives.method = "user"
    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 1.0)
    ph.time.guess = (0.0, 1.0)
    ph.state.initial.h = (0.0, 0.0)
    ph.state.initial.r[:] = (0.0, 0.0)
    ph.state.final.h = (1.0, 1.0)
    ph.state.bounds.h = (-10.0, 10.0)
    ph.state.bounds.r[:] = (-10.0, 10.0)
    ph.control.bounds.u[:] = (-5.0, 5.0)
    ph.state.guess.h = (0.0, 1.0)
    ph.state.guess.r[:] = (0.0, 1.0)
    ph.control.guess.u[:] = (0.0, 1.0)
    problem.parameter.bounds.k = (0.5, 2.0)
    problem.parameter.guess.k = 1.0
    problem.parameter.bounds.m[:] = (0.0, 1.0)
    problem.parameter.guess.m[:] = 0.5
    problem.discrete.bounds.close = (0.0, 0.0)
    ph.mesh = yapss.Mesh.uniform(segments=2, points=4)
    problem.ipopt_options.print_level = 0
    return problem, ph


def test_block_derivatives_written_by_row_solve() -> None:
    """The whole feature on block fields, checked against tracing."""
    problem, _ = build()
    user = problem.solve()
    assert user.converged

    traced, _ = build()
    traced.derivatives.method = "auto"
    assert abs(user.objective - traced.solve().objective) < 1e-6


# ------------------------------------------------------------------- a row of a block


def test_a_block_row_must_be_named() -> None:
    """A block field is several rows, so a derivative of it names which one."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.r.u[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        AttributeError,
        "is a block field of 2 rows",
        "Give the row",
        at="jacobian.dynamics.r.u",
    ):
        problem.solve()


def test_a_block_row_index_is_in_range() -> None:
    """Out of range is a mistake about the declaration, and the message says how many rows."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.r[5].u[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        IndexError,
        "out of range for a block field of 2 rows",
        at="jacobian.dynamics.r[5]",
    ):
        problem.solve()


def test_a_row_index_is_an_integer() -> None:
    """A row is a position, so a name is refused where it is written."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.r["h"].u[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(TypeError, "takes a row index", "an integer from 0 to 1", at="jacobian.dynamics"):
        problem.solve()


@not_yet("message", "indexing a field of one row gives Python's own subscripting error")
def test_a_field_of_one_row_takes_no_index() -> None:
    """Indexing a scalar row is refused rather than accepted as row zero.

    YAPSS has the message -- "has one row, so it takes no row index" -- and this spelling does
    not reach it: what a user gets is ``'_JacobianRow' object is not subscriptable``.
    """
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.h[0].u[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        (TypeError, AttributeError, IndexError),
        "one row",
        at="jacobian.dynamics.h[0]",
    ):
        problem.solve()


def test_a_block_variable_needs_its_row_too() -> None:
    """The variable side of a derivative names a row of a block, as the row side does."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.h.u = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(AttributeError, "block field of 2 rows", at="jacobian.dynamics.h.u"):
        problem.solve()


# ----------------------------------------------------------- half a spelling, and too much


def test_part_of_a_spelling_is_not_a_derivative() -> None:
    """Assigning partway along the chain says that more of it was expected."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        (AttributeError, TypeError),
        "cannot be replaced",
        "one derivative at a time",
        at="jacobian.dynamics =",
    ):
        problem.solve()


def test_an_unknown_output_group_is_refused() -> None:
    """The groups are the callback's own outputs, with a suggestion."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamic.h.k = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(AttributeError, "has no 'dynamic'", at="jacobian.dynamic"):
        problem.solve()


def test_an_unknown_row_is_refused() -> None:
    """Within a group, the rows are the declared fields."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.hh.k = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(AttributeError, "has no 'hh'", at="jacobian.dynamics.hh"):
        problem.solve()


# ------------------------------------------------------- the endpoint gradient and Hessian


def test_an_endpoint_gradient_names_the_end_and_the_variable() -> None:
    """`gradient.phases[ph].final.h`: which phase, which end, which variable."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases[problem.phases.run].final] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        TypeError,
        "takes an endpoint variable",
        at="gradient[gradient.phases[problem.phases.run].final] = 1.0",
    ):
        problem.solve()


def test_an_endpoint_gradient_names_a_variable_that_exists() -> None:
    """Checked against the phase's state and its independent variable."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases[problem.phases.run].final.nope] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(AttributeError, "has no 'nope'", at="final.nope"):
        problem.solve()


def test_a_parameter_derivative_is_named_directly() -> None:
    """A parameter belongs to the problem, so it is not reached through a phase."""
    problem, _ = build()

    def gradient(arg, gradient):
        run = gradient.phases[problem.phases.run]
        gradient[run.integral.cost] = 1.0
        gradient[run.final.h] = 1.0
        gradient[gradient.parameter.k] = 0.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    assert problem.solve().converged


def test_an_unknown_parameter_says_where_endpoint_variables_live() -> None:
    """The likeliest mistake is naming a state here, so the message says how to reach one."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.parameter.h] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        AttributeError,
        "the problem has no 'h'",
        "reached through its phase",
        at="gradient.parameter.h",
    ):
        problem.solve()


def test_an_endpoint_hessian_names_both_of_its_variables() -> None:
    """A second derivative relates two variables, so one alone is half an entry."""
    problem, _ = build()

    def objective_hessian(arg, hessian):
        hessian[hessian.phases[problem.phases.run].final.h] = 1.0
        return hessian

    problem.register.objective_hessian(objective_hessian, replace=True)
    with raises(
        TypeError,
        "A second derivative names two",
        at="hessian[hessian.phases[problem.phases.run].final.h] = 1.0",
    ):
        problem.solve()


def test_a_discrete_jacobian_names_the_constraint_first() -> None:
    """A discrete derivative is of one constraint by one variable, in that order."""
    problem, _ = build()

    def discrete_jacobian(arg, jacobian):
        jacobian.discrete.nope[jacobian.phases[problem.phases.run].final.h] = 1.0
        return jacobian

    problem.register.discrete_jacobian(discrete_jacobian, replace=True)
    with raises(AttributeError, "has no group 'nope'", at="jacobian.discrete.nope"):
        problem.solve()


# ------------------------------------------------ a spelling that stops too early or too late


def test_indexing_a_finished_derivative_is_refused() -> None:
    """Once the spelling names a derivative, there is nothing left to index."""
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.h.k[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        AttributeError,
        "is a first derivative and is written by assigning it",
        "belongs in the hessian callback",
        at="jacobian.dynamics.h.k[0]",
    ):
        problem.solve()


@not_yet("message", "indexing a scalar row of a derivative gives Python's own error")
def test_indexing_partway_along_a_spelling_is_refused() -> None:
    """A row index belongs to a block field, not to a half-written derivative.

    The same gap as indexing a field of one row, reached by the other spelling: what a user
    gets is ``'_JacobianRow' object does not support item assignment``.
    """
    problem, ph = build()

    def jacobian(arg, jacobian):
        jacobian.dynamics.h[0] = 1.0
        return jacobian

    ph.register.continuous_jacobian(jacobian, replace=True)
    with raises(
        (TypeError, AttributeError),
        "has one row",
        at="jacobian.dynamics.h[0] = 1.0",
    ):
        problem.solve()


def test_a_second_derivative_of_a_block_variable_names_its_row() -> None:
    """Both variables of a Hessian entry name rows when they are blocks."""
    problem, ph = build()

    def hessian(arg, hessian):
        hessian.integrand.cost.u[0].u = 1.0
        return hessian

    ph.register.continuous_hessian(hessian, replace=True)
    with raises(AttributeError, "block field of 2 rows", at="hessian.integrand.cost.u[0].u"):
        problem.solve()


def test_a_finished_second_derivative_takes_no_more_names() -> None:
    """Two variables is the whole of it, and a third says so."""
    problem, ph = build()

    def hessian(arg, hessian):
        hessian.integrand.cost.u[0].u[1].k = 1.0
        return hessian

    ph.register.continuous_hessian(hessian, replace=True)
    with raises(
        TypeError,
        "is the whole of the derivative",
        at="hessian.integrand.cost.u[0].u[1].k",
    ):
        problem.solve()


# ----------------------------------------------------------- the discrete derivatives


def test_a_discrete_derivative_names_a_constraint_and_a_variable() -> None:
    """A constraint alone is half of it, and the message shows the other half."""
    problem, _ = build()

    def discrete_jacobian(arg, jacobian):
        jacobian.discrete.close = 1.0
        return jacobian

    problem.register.discrete_jacobian(discrete_jacobian, replace=True)
    with raises(
        AttributeError,
        "names a constraint but no variable",
        at="jacobian.discrete.close = 1.0",
    ):
        problem.solve()


def test_a_discrete_derivative_names_the_discrete_group() -> None:
    """The constraints hang under `discrete`, and a near miss is suggested."""
    problem, _ = build()

    def discrete_jacobian(arg, jacobian):
        jacobian.discreet.close[jacobian.phases[problem.phases.run].final.h] = 1.0
        return jacobian

    problem.register.discrete_jacobian(discrete_jacobian, replace=True)
    with raises(
        AttributeError,
        "a derivative is written against",
        at="jacobian.discreet",
    ):
        problem.solve()


def test_the_discrete_group_cannot_be_replaced() -> None:
    """Assigning the whole group at once is refused, naming the shape of one entry."""
    problem, _ = build()

    def discrete_jacobian(arg, jacobian):
        jacobian.discrete = 1.0
        return jacobian

    problem.register.discrete_jacobian(discrete_jacobian, replace=True)
    with raises(
        AttributeError,
        "cannot be assigned",
        "one derivative at a time",
        at="jacobian.discrete = 1.0",
    ):
        problem.solve()


def test_an_endpoint_derivative_is_written_not_read() -> None:
    """Reading an entry back is a mistake about what the target is for."""
    problem, _ = build()

    def gradient(arg, gradient):
        _ = gradient[gradient.parameter.k]
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        TypeError,
        "is the whole of the derivative",
        at="_ = gradient[gradient.parameter.k]",
    ):
        problem.solve()


def test_an_endpoint_hessian_names_two_variables() -> None:
    """One parameter alone is half an entry, and the message names the form that works."""
    problem, _ = build()

    def objective_hessian(arg, hessian):
        hessian[hessian.parameter.k] = 1.0
        return hessian

    problem.register.objective_hessian(objective_hessian, replace=True)
    with raises(
        TypeError,
        "A second derivative names two",
        at="hessian[hessian.parameter.k] = 1.0",
    ):
        problem.solve()


def test_a_variable_is_not_a_place_to_write() -> None:
    """The namespace names variables; the target is what a derivative is written on.

    This is the mistake the 0.4.0 path spelling invited, so it is the one the message has to
    answer: what was written names a variable and says nothing about the derivative.
    """
    problem, _ = build()

    def objective_hessian(arg, hessian):
        hessian.phases[problem.phases.run].final.h = 1.0
        return hessian

    problem.register.objective_hessian(objective_hessian, replace=True)
    with raises(
        AttributeError,
        "is a variable, not a derivative",
        at="hessian.phases[problem.phases.run].final.h = 1.0",
    ):
        problem.solve()


# ------------------------------------------- block rows on the endpoint side of a spelling


def test_an_endpoint_block_state_names_its_row() -> None:
    """`final.r` is two rows, so a gradient entry says which."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases[problem.phases.run].final.r] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        TypeError,
        "block field of 2 rows",
        at="gradient[gradient.phases[problem.phases.run].final.r] = 1.0",
    ):
        problem.solve()


def test_a_block_parameter_names_its_row() -> None:
    """A parameter declared with rows is addressed like any other block."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.parameter.m] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(
        TypeError,
        "block field of 2 rows",
        at="gradient[gradient.parameter.m] = 1.0",
    ):
        problem.solve()


def test_a_finished_endpoint_derivative_takes_no_index() -> None:
    """Once the entry is named, indexing it is a mistake about what is left."""
    problem, _ = build()

    def gradient(arg, gradient):
        gradient[gradient.phases[problem.phases.run].final.h[0]] = 1.0
        return gradient

    problem.register.objective_gradient(gradient, replace=True)
    with raises(TypeError, "has one row, so it takes no row index", at="final.h[0]"):
        problem.solve()


def test_the_second_variable_of_an_endpoint_hessian_may_be_a_block_parameter() -> None:
    """Reached by row, as every block is."""
    problem, _ = build()

    def objective_hessian(arg, hessian):
        hessian[hessian.phases[problem.phases.run].final.h, hessian.parameter.m] = 1.0
        return hessian

    problem.register.objective_hessian(objective_hessian, replace=True)
    with raises(
        TypeError,
        "block field of 2 rows",
        at="hessian[hessian.phases[problem.phases.run].final.h, hessian.parameter.m] = 1.0",
    ):
        problem.solve()
