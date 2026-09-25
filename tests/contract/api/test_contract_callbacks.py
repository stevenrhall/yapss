"""What a callback is given, what it must fill in, and what it is told when it does not.

Three callbacks: `continuous` per phase, and `objective` and `discrete` on the problem. Each
is handed its inputs by name and fills in its outputs by name, and returns the output object
so that forgetting to is caught rather than silently producing nothing.

Several messages here come from the transcription rather than the front end, because they are
about what a callback *did* and the front end has not run it yet. A user does not care which
package refused them, so they are stated here, with the page's subject.
"""

from __future__ import annotations

import numpy as np

import yapss
from yapss.math import cos, sin

from ._api import G0, not_yet, raises, solvable


class _Two(yapss.State):
    a = yapss.scalar()
    b = yapss.scalar()


class _One(yapss.Control):
    c = yapss.scalar()


def dynamics(arg, out):
    """Fill in every row of every output, which is what a complete callback does."""
    v, theta = arg.state.v, arg.control.theta
    out.dynamics.x = v * cos(theta)
    out.dynamics.y = v * sin(theta)
    out.dynamics.v = G0 * sin(theta)
    out.path.speed = v
    out.integrand.effort = theta**2
    return out


# ------------------------------------------------------------------- reading the inputs


def test_inputs_are_read_by_name() -> None:
    """The state, the control and the phase's independent variable, under their own names."""
    seen = []
    p = solvable()

    def watch(arg, out):
        seen.append((arg.time, arg.state.v, arg.control.theta, arg.phase))
        return dynamics(arg, out)

    p.phases.slide.register.continuous(watch)
    p.solve()
    assert seen
    assert seen[0][3] is p.phases.slide


def test_a_numeric_function_from_outside_yapss_is_explained() -> None:
    """Under the tracing method the inputs are symbols, and the note says what to use instead."""
    p = solvable()

    def floated(arg, out):
        dynamics(arg, out)
        out.dynamics.x = float(arg.state.v)
        return out

    p.phases.slide.register.continuous(floated)
    with raises(TypeError, "symbolic inputs", "yapss.math", at="float(arg.state.v)"):
        p.solve()


def test_a_misspelled_input_names_the_vector_it_is_not_in() -> None:
    """The message says which vector was asked, so the user knows where to look."""
    p = solvable()

    def misread(arg, out):
        return arg.state.zz

    p.phases.slide.register.continuous(misread)
    with raises(AttributeError, "phase 'slide' state has no field 'zz'", at="arg.state.zz"):
        p.solve()


def test_a_misspelled_output_names_the_output_it_is_not_in() -> None:
    """The same, for the side being written."""
    p = solvable()

    def miswrite(arg, out):
        out.dynamics.zz = 0.0
        return out

    p.phases.slide.register.continuous(miswrite)
    with raises(AttributeError, "phase 'slide' dynamics has no field 'zz'", at="out.dynamics.zz"):
        p.solve()


# --------------------------------------------------------------- filling in the outputs


def test_every_row_must_be_assigned() -> None:
    """A row never written is a mistake, and the message lists every one of them by name."""
    p = solvable()

    def partial(arg, out):
        out.dynamics.x = arg.state.v
        return out

    p.phases.slide.register.continuous(partial)
    with raises(
        ValueError,
        "returned without assigning",
        "dynamics.y",
        "dynamics.v",
        "integrand.effort",
        at="solve",
    ):
        p.solve()


def test_a_callback_may_return_nothing() -> None:
    """The output object is filled in place, so returning it is a cue and not a requirement."""
    p = solvable()

    def forgetful(arg, out):
        dynamics(arg, out)

    p.phases.slide.register.continuous(forgetful)
    assert p.solve().converged


def test_a_callback_may_not_return_something_else() -> None:
    """Returning a tuple of rows is the 0.3.0 habit, and is caught rather than ignored."""
    p = solvable()

    def tupled(arg, out):
        dynamics(arg, out)
        return (0.0, 0.0, 0.0)

    p.phases.slide.register.continuous(tupled)
    with raises((TypeError, ValueError), "return", at="p.solve()"):
        p.solve()


def test_an_explicit_zero_is_an_assignment() -> None:
    """Writing zero is how a user says a row really is zero, and it is taken at its word."""
    p = solvable()

    def flat(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = 0.0
        out.dynamics.v = 0.0
        out.path.speed = arg.state.v
        out.integrand.effort = 0.0
        return out

    p.phases.slide.register.continuous(flat)
    p.phases.slide.state.x.final = (0.0, 1.0)
    p.discrete.drop.bounds = (0.0, 0.0)
    assert p.solve() is not None


def test_the_objective_returns_its_value() -> None:
    """One expression, so there is no output object to fill."""
    p = solvable()

    def empty(arg):
        return None

    p.register.objective(empty)
    with raises(ValueError, "returned nothing", "must return the objective", at="solve"):
        p.solve()


# ---------------------------------------------------------------- reaching a phase's ends


def test_an_endpoint_is_reached_by_phase_handle() -> None:
    """`arg[ph]` takes the handle the problem carries, not the phase's name."""
    p = solvable()

    def by_name(arg):
        return arg["slide"].final.time

    p.register.objective(by_name)
    with raises(KeyError, "takes a phase handle", "problem.phases.", at='arg["slide"]'):
        p.solve()


def test_an_endpoint_holds_the_state_and_the_independent_variable() -> None:
    """One namespace: the phase's variables at a point, which is what a Jacobian indexes."""
    p = solvable()

    def both(arg):
        end = arg[p.phases.slide]
        return end.final.time + 0.0 * end.final.x

    p.register.objective(both)
    assert p.solve().converged


def test_a_misspelled_endpoint_name_is_refused() -> None:
    """Checked against the state's own fields, plus the independent variable."""
    p = solvable()

    def wrong(arg):
        return arg[p.phases.slide].final.nope

    p.register.objective(wrong)
    with raises(AttributeError, "final state has no 'nope'", at="final.nope"):
        p.solve()


# ------------------------------------------------------------- what the callback must be


def test_a_callback_must_be_pointwise() -> None:
    """Mixing values across time points is refused: the callback is called at many at once."""
    p = solvable()

    def summed(arg, out):
        dynamics(arg, out)
        out.dynamics.x = np.sum(np.asarray(arg.state.v))
        return out

    p.phases.slide.register.continuous(summed)
    with raises(ValueError, "continuous callback", at="solve"):
        p.solve()


def test_an_exception_inside_a_callback_says_which_callback() -> None:
    """A user's own error is not swallowed; a note names where it happened."""
    p = solvable()

    def boom(arg, out):
        msg = "boom"
        raise RuntimeError(msg)

    p.phases.slide.register.continuous(boom)
    with raises(RuntimeError, "boom", "Raised in", at="raise RuntimeError"):
        p.solve()


def test_the_note_names_the_callback_the_user_wrote() -> None:
    """A user reads the note to find their own `def`, so that is what it must name, with the
    phase it was registered on -- never the adapter the front end calls it through."""
    p = solvable()

    def my_dynamics(arg, out):
        msg = "boom"
        raise RuntimeError(msg)

    p.phases.slide.register.continuous(my_dynamics)
    with raises(RuntimeError, "my_dynamics", "phase 'slide'", at="raise RuntimeError"):
        p.solve()


@not_yet("message", "non-finite values are reported by position, not by name")
def test_a_non_finite_value_is_reported_by_name() -> None:
    """The whole of this API is that a quantity has a name; a refusal has to use it.

    Today the message says ``phase 0 integrand[0] is not finite``, which is the transcription's
    numbering showing through. What the user wrote was ``out.integrand.effort``.
    """
    p = solvable("central-difference")

    def infinite(arg, out):
        dynamics(arg, out)
        out.integrand.effort = arg.control.theta / 0.0
        return out

    p.phases.slide.register.continuous(infinite)
    with raises(ValueError, "not finite", "phase 'slide'", "effort", at="solve"):
        p.solve()


# ------------------------------------------------------------------ the discrete callback


def test_the_discrete_callback_fills_its_output() -> None:
    """Like the continuous one, it returns what it filled."""
    p = solvable()

    def nothing(arg, out):
        return out

    p.register.discrete(nothing)
    with raises(ValueError, "returned without assigning", "drop", at="solve"):
        p.solve()


def test_a_problem_may_have_no_discrete_constraints() -> None:
    """The zero case: nothing declared, nothing to fill, no callback needed."""

    class Only(yapss.Phase):
        state: _Two
        control: _One
        time: yapss.Independent

    class Bare(yapss.Phases):
        only: Only

    p = yapss.Problem("bare", phases=Bare)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.a = arg.control.c
        out.dynamics.b = 0.0
        return out

    @p.register.objective
    def objective(arg):
        return arg[ph].final.time

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 1.0)
    ph.time.guess = (0.0, 1.0)
    ph.state.a.initial = (0.0, 0.0)
    ph.state.b.initial = (0.0, 0.0)
    ph.state.a.guess = (0.0, 1.0)
    ph.state.b.guess = (0.0, 0.0)
    ph.control.c.guess = (0.0, 0.0)
    ph.control.c.bounds = (-1.0, 1.0)
    p.ipopt_options.print_level = 0
    assert p.solve() is not None


# ------------------------------------------------- what the argument and the output are not


def test_an_output_row_cannot_be_read_before_it_is_written() -> None:
    """The output starts empty, which is what makes completeness decidable at the end."""
    p = solvable()

    def peek(arg, out):
        _ = out.dynamics.x
        return dynamics(arg, out)

    p.phases.slide.register.continuous(peek)
    with raises(AttributeError, "'x' has not been assigned", at="out.dynamics.x"):
        p.solve()


def test_an_input_cannot_be_assigned() -> None:
    """The state a callback is given is the transcription's, and is read-only."""
    p = solvable()

    def overwrite(arg, out):
        arg.state.v = 1.0
        return dynamics(arg, out)

    p.phases.slide.register.continuous(overwrite)
    with raises(AttributeError, "is read-only", "'v' cannot be assigned", at="arg.state.v"):
        p.solve()


def test_an_output_group_cannot_be_replaced() -> None:
    """Assigning the whole group would bypass the per-field checks, so it names the form."""
    p = solvable()

    def replace(arg, out):
        out.dynamics = 1.0
        return out

    p.phases.slide.register.continuous(replace)
    with raises(
        AttributeError,
        "cannot be replaced",
        "fill it field by field",
        at="out.dynamics = 1.0",
    ):
        p.solve()


def test_a_misspelled_output_group_is_suggested() -> None:
    """`out.dynamic` is one letter from `out.dynamics`, and is told so."""
    p = solvable()

    def misspelled(arg, out):
        out.dynamic.x = 1.0
        return out

    p.phases.slide.register.continuous(misspelled)
    with raises(AttributeError, "has no 'dynamic'", "Did you mean 'dynamics'", at="out.dynamic"):
        p.solve()


# --------------------------------------------------------- what a row of an output may be


def test_a_row_is_a_scalar_or_one_value_per_time_point() -> None:
    """Two dimensions is neither, and the message says what it got."""
    p = solvable("central-difference")

    def square(arg, out):
        dynamics(arg, out)
        out.dynamics.x = np.zeros((2, 2))
        return out

    p.phases.slide.register.continuous(square)
    with raises(
        ValueError,
        "a row is a scalar or one value per time point",
        "array of shape (2, 2)",
        at="out.dynamics.x",
    ):
        p.solve()


def test_a_row_is_not_a_string() -> None:
    """The same rule, for a value that is not numeric at all."""
    p = solvable("central-difference")

    def lettered(arg, out):
        dynamics(arg, out)
        out.dynamics.x = "a"
        return out

    p.phases.slide.register.continuous(lettered)
    with raises(TypeError, "a row is a scalar or one value per time point", "got str"):
        p.solve()


def test_a_row_is_not_a_boolean() -> None:
    """A comparison written by accident, caught here as everywhere else."""
    p = solvable("central-difference")

    def flagged(arg, out):
        dynamics(arg, out)
        out.dynamics.x = True
        return out

    p.phases.slide.register.continuous(flagged)
    with raises(TypeError, "a boolean is not a number", at="out.dynamics.x"):
        p.solve()


@not_yet("message", "a row of the wrong length falls through to numpy's broadcast error")
def test_a_row_of_the_wrong_length_is_told_how_many_are_needed() -> None:
    """YAPSS has the message and this path does not reach it.

    What a user gets is ``could not broadcast input array from shape (3,) into shape (10,)``,
    which names neither the field nor the callback nor what 10 is. The message written for it
    -- "a row needs one value per time point, 10; got 3" -- is reached only from elsewhere.
    """
    p = solvable("central-difference")

    def short(arg, out):
        dynamics(arg, out)
        out.dynamics.x = np.zeros(3)
        return out

    p.phases.slide.register.continuous(short)
    with raises(ValueError, "one value per time point", "dynamics 'x'", at="out.dynamics.x"):
        p.solve()
