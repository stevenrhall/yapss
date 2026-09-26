"""What a callback is given, what it must fill in, and what it is told when it does not.

Three callbacks: `continuous` per phase, and `objective` and `discrete` on the problem. Each
is handed its inputs by name. The objective returns its value; the other two fill in their
outputs by name and return nothing, and an output left unfilled is caught rather than silently
producing nothing.

Several messages here come from the transcription rather than the front end, because they are
about what a callback *did* and the front end has not run it yet. A user does not care which
package refused them, so they are stated here, with the page's subject.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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

    p.phases.slide.register.continuous(miswrite)
    with raises(AttributeError, "phase 'slide' dynamics has no field 'zz'", at="out.dynamics.zz"):
        p.solve()


# --------------------------------------------------------------- filling in the outputs


def test_every_row_must_be_assigned() -> None:
    """A row never written is a mistake, and the message lists every one of them by name."""
    p = solvable()

    def partial(arg, out):
        out.dynamics.x = arg.state.v

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

    p.phases.slide.register.continuous(summed)
    with raises(
        ValueError,
        "continuous callback",
        "summed (",
        "not pointwise",
        "phase 'slide' dynamics.x",
        at="solve",
    ):
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


def test_a_non_finite_value_is_reported_by_name() -> None:
    """The whole of this API is that a quantity has a name; a refusal has to use it.

    The transcription numbers the rows (``phase 0 integrand[0]``); what the user wrote was
    ``out.integrand.effort``, in phase 'slide', and that is what the message names.
    """
    p = solvable("central-difference")

    def infinite(arg, out):
        dynamics(arg, out)
        out.integrand.effort = arg.control.theta / 0.0

    p.phases.slide.register.continuous(infinite)
    with raises(ValueError, "not finite", "phase 'slide'", "effort", at="solve"):
        p.solve()


# ------------------------------------------------------------------ the discrete callback


def test_the_discrete_callback_fills_its_output() -> None:
    """Like the continuous one, it must fill every output it declares."""
    p = solvable()

    def nothing(arg, out):
        pass

    p.register.discrete(nothing)
    with raises(ValueError, "returned without assigning", "drop", at="solve"):
        p.solve()


# ------------------------------------------------------------ what a filling callback returns


def test_a_continuous_callback_returning_out_is_told_to_delete_the_return() -> None:
    """One form, fill and end: returning `out` is refused, with the one-line fix."""
    p = solvable()

    def returns_out(arg, out):
        dynamics(arg, out)
        return out

    p.phases.slide.register.continuous(returns_out)
    with raises(TypeError, "returns_out'", "returned 'out'", "delete the return", at="solve"):
        p.solve()


def test_a_discrete_callback_returning_out_is_told_to_delete_the_return() -> None:
    """The discrete callback fills its output the same way, and is held to the same form."""
    p = solvable()

    def returns_out(arg, out):
        out.discrete.drop = arg[p.phases.slide].final.y
        return out

    p.register.discrete(returns_out)
    with raises(TypeError, "discrete callback", "returned 'out'", "delete the return", at="solve"):
        p.solve()


def test_a_callback_returning_its_rows_is_refused() -> None:
    """Returning the values instead of filling `out` -- the 0.3.0 habit -- names what came back."""
    p = solvable()

    def returns_rows(arg, out):
        dynamics(arg, out)
        return (arg.state.v, arg.state.v, arg.state.v)

    p.phases.slide.register.continuous(returns_rows)
    with raises(
        TypeError, "returns_rows'", "returned (", "Fill 'out' and return nothing", at="solve"
    ):
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

    p.phases.slide.register.continuous(lettered)
    with raises(TypeError, "a row is a scalar or one value per time point", "got str"):
        p.solve()


def test_a_row_is_not_a_boolean() -> None:
    """A comparison written by accident, caught here as everywhere else."""
    p = solvable("central-difference")

    def flagged(arg, out):
        dynamics(arg, out)
        out.dynamics.x = True

    p.phases.slide.register.continuous(flagged)
    with raises(TypeError, "a boolean is not a number", at="out.dynamics.x"):
        p.solve()


def test_a_row_of_the_wrong_length_is_told_how_many_are_needed() -> None:
    """A row has one value per time point, and one of another length is refused at its line.

    A single number is still a constant row; a one-element array is not, since it is neither a
    number nor one value per point.
    """
    p = solvable("central-difference")

    def short(arg, out):
        dynamics(arg, out)
        out.dynamics.x = np.zeros(3)

    p.phases.slide.register.continuous(short)
    with raises(ValueError, "one value per time point", "dynamics 'x'", at="out.dynamics.x"):
        p.solve()


# ------------------------------------------------------------- the rows of a block field


def _block_problem(continuous: Any) -> Any:
    """Return a problem whose one state is a two-row block field, with `continuous` registered."""

    class Pair(yapss.State):
        r = yapss.vector(2)

    class Rate(yapss.Control):
        u = yapss.scalar()

    class Only(yapss.Phase):
        state: Pair
        control: Rate
        time: yapss.Independent

    class BlockPhases(yapss.Phases):
        only: Only

    p = yapss.Problem("block rows", phases=BlockPhases)
    ph = p.phases.only
    ph.register.continuous(continuous)
    p.register.objective(lambda arg: arg[ph].final.time)
    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 1.0)
    ph.state.r.initial[:] = (0.0, 0.0)
    ph.control.u.bounds = (-1.0, 1.0)
    ph.time.guess = (0.0, 1.0)
    p.ipopt_options.print_level = 0
    return p


def test_a_block_input_row_cannot_be_written() -> None:
    """An input is read-only, a block's rows included: the write fails at its line."""

    def writes_an_input(arg, out):
        arg.state.r[0] = 0.0
        out.dynamics.r = [arg.control.u, arg.control.u]

    with raises(ValueError, "read-only", "writes_an_input", at="arg.state.r[0]"):
        _block_problem(writes_an_input).solve()


def test_a_write_into_a_read_block_output_fails_rather_than_being_lost() -> None:
    """Reading a block output gives its rows as a new array; writing into that is refused.

    Were it writable, the write would land in the copy and the field would keep its value, so
    the solve would answer a different problem than the callback describes, without a word.
    """

    def writes_a_read_copy(arg, out):
        out.dynamics.r = [arg.control.u, arg.control.u]
        out.dynamics.r[0] = 2.0 * arg.control.u

    with raises(ValueError, "read-only", "writes_a_read_copy", at="out.dynamics.r[0]"):
        _block_problem(writes_a_read_copy).solve()


@not_yet(
    "item 24",
    "one row of a block output cannot be written by index, as a row of an input is read",
)
def test_one_row_of_a_block_output_is_written_by_index() -> None:
    """A block field's rows are reached by index where they are read and where they are
    written, so a callback can fill a block one row at a time."""

    def by_row(arg, out):
        out.dynamics.r[0] = arg.control.u
        out.dynamics.r[1] = arg.control.u

    solution = _block_problem(by_row).solve()
    assert solution.converged


def test_a_one_element_array_is_not_a_constant_row() -> None:
    """Broadcast silently, it made a row that was one value where the rest vary."""
    p = solvable()

    def one_element(arg, out):
        dynamics(arg, out)
        out.integrand.effort = np.atleast_1d(arg.control.theta[0])[:1]

    p.phases.slide.register.continuous(one_element)
    with raises(ValueError, "one value per time point", "got 1", at="out.integrand.effort"):
        p.solve()


def test_the_objective_is_not_a_boolean() -> None:
    """A comparison returned by accident was taken as 0 or 1, and solved."""
    p = solvable()
    ph = p.phases.slide

    def compares(arg):
        return arg[ph].final.time > 1.0

    p.register.objective(compares)
    with raises(TypeError, "compares' returned a boolean", "comparison", at="solve"):
        p.solve()


@pytest.mark.parametrize("returned", ["array", "tuple", "str", "dict", "complex", "0-d complex"])
def test_the_objective_is_one_number(returned: str) -> None:
    """Anything but one real number is refused, naming the objective callback, not 0.3.0's
    arg.objective."""
    p = solvable()
    ph = p.phases.slide
    values = {
        "array": lambda arg: np.array([arg[ph].final.time] * 3),
        "tuple": lambda arg: (arg[ph].final.time, 1.0),
        "str": lambda arg: "time",
        "dict": lambda arg: {"time": arg[ph].final.time},
        "complex": lambda arg: 1j,
        "0-d complex": lambda arg: np.array(1j),
    }
    p.register.objective(values[returned])
    with raises(TypeError, "objective callback", "must return one number", at="solve"):
        p.solve()


def test_the_objective_is_returned_not_assigned() -> None:
    """0.3.0 wrote arg.objective = ...; the message says what 0.4 does instead."""
    p = solvable()
    ph = p.phases.slide

    def assigns(arg):
        arg.objective = arg[ph].final.time

    p.register.objective(assigns)
    with raises(AttributeError, "returned from the objective callback", at="arg.objective"):
        p.solve()


# ------------------------------------------------------------ pinned by the 2026-09-25 audit


@not_yet("freeze item 2", "a callback argument still reads and writes rows by position")
@pytest.mark.parametrize("access", ["out.dynamics[:]", "out.path[0]", "arg.state[0]"])
@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")  # stopped after 1 iteration
def test_a_callback_argument_has_no_positions(access: str) -> None:
    """Rows are reached by name, since a position is only the order a class body happens to
    be written in, and reordering it would silently change what the code means."""
    p = solvable()
    p.ipopt_options.print_level = 0
    p.ipopt_options.max_iter = 1

    def positional(arg, out):
        dynamics(arg, out)
        if access == "out.dynamics[:]":
            out.dynamics[:] = (arg.state.v, arg.state.v, arg.state.v)
        elif access == "out.path[0]":
            out.path[0] = arg.state.v
        else:
            _ = arg.state[0]

    p.phases.slide.register.continuous(positional)
    with pytest.raises(TypeError):
        p.solve()


def test_a_misspelled_discrete_output_is_suggested() -> None:
    """The discrete callback's outputs are checked like the continuous ones."""
    p = solvable()
    ph = p.phases.slide

    def misspelled(arg, out):
        out.discrete.dropp = arg[ph].final.y

    p.register.discrete(misspelled)
    with raises(AttributeError, "'dropp'", "Did you mean 'drop'", at="out.discrete.dropp"):
        p.solve()


@pytest.mark.parametrize("method", ["auto", "central-difference"])
def test_a_python_if_on_an_input_is_refused_under_every_method(method: str) -> None:
    """Under "auto" it names yapss.math.where; numerically, the truth value is ambiguous."""
    p = solvable(method)

    def branches(arg, out):
        dynamics(arg, out)
        if arg.state.v > 1.0:
            out.dynamics.x = 0.0

    p.phases.slide.register.continuous(branches)
    fragment = "yapss.math.where" if method == "auto" else "ambiguous"
    with pytest.raises((TypeError, ValueError), match=fragment):
        p.solve()


def test_an_endpoint_input_cannot_be_written() -> None:
    """The objective reads endpoints; it cannot change them."""
    p = solvable()
    ph = p.phases.slide

    def writes(arg):
        arg[ph].final.x = 0.0
        return arg[ph].final.time

    p.register.objective(writes)
    with raises(AttributeError, "is an input", at="final.x"):
        p.solve()


def test_an_input_container_cannot_be_replaced() -> None:
    """arg.state is the phase's state; the callback cannot swap it for another."""
    p = solvable()

    def replaces(arg, out):
        arg.state = None
        dynamics(arg, out)

    p.phases.slide.register.continuous(replaces)
    with raises(AttributeError, "cannot be assigned", at="arg.state"):
        p.solve()


def test_inputs_follow_the_point_they_are_read_at() -> None:
    """Arguments are reused between calls, so each must read the current point, not a copy."""
    p = solvable("central-difference")
    p.ipopt_options.print_level = 0
    ph = p.phases.slide
    times: set[float] = set()
    finals: set[float] = set()

    def watch(arg, out):
        times.update(float(t) for t in arg.time)
        dynamics(arg, out)

    def objective(arg):
        finals.add(float(arg[ph].final.time))
        return arg[ph].final.time

    ph.register.continuous(watch)
    p.register.objective(objective)
    p.solve()
    assert len(times) > 10
    assert len(finals) > 1


def test_an_explicit_zero_may_then_be_added_to() -> None:
    """Assigning zero is an assignment, so an in-place update after it is too."""
    p = solvable()
    p.ipopt_options.print_level = 0

    def accumulates(arg, out):
        dynamics(arg, out)
        out.integrand.effort = 0.0
        out.integrand.effort += arg.control.theta**2

    p.phases.slide.register.continuous(accumulates)
    assert p.solve().converged
