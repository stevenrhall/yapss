"""What an initial guess is, and what a guess that cannot be used says.

A guess is one of three things -- a number held over the phase, a `(first, last)` pair, or
`yapss.interp(t, values)` -- and the phase's time guess is the only source of the phase's
guessed extent, which is what lets a sampled guess carry no times of its own to disagree with.
"""

from __future__ import annotations

import numpy as np

import yapss

from ._api import Control, State, problem, raises, solvable

# ----------------------------------------------------------------- the three accepted forms


def test_a_pair_is_linear_from_one_end_to_the_other() -> None:
    """The commonest guess: where the row starts and where it ends.

    It reads back tagged with what kind of guess it is, since the three forms have to be told
    apart later, and a pair of numbers cannot say which of them it is.
    """
    ph = problem().phases.first
    ph.state.guess.x = (0.0, 10.0)
    assert ph.state.guess.x == ("linear", 0.0, 10.0)


def test_samples_carry_their_own_times() -> None:
    """A sampled guess is pure data on its own grid, so rows need not share one."""
    ph = problem().phases.first
    ph.state.guess.x = yapss.interp([0.0, 0.5, 1.0], [0.0, 2.0, 1.0])
    assert ph.state.guess.x is not None


def test_a_block_field_takes_one_guess_per_row() -> None:
    """Rows of a block are guessed as the rows of any other aspect are written."""
    ph = problem().phases.first
    ph.state.guess.y[:] = [(0.0, 1.0), (2.0, 3.0)]
    assert list(ph.state.guess.y) == [("linear", 0.0, 1.0), ("linear", 2.0, 3.0)]


# ------------------------------------------------------------------------ what is refused


def test_a_guess_is_a_pair_not_a_number() -> None:
    """One number is ambiguous between a constant and a typo, so the message offers the fix."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "a guess is a (first, last) pair",
        "write (0.5, 0.5)",
        at="guess.x",
    ):
        ph.state.guess.x = 0.5


def test_a_guess_takes_exactly_two_values() -> None:
    """Three values is neither a pair nor samples, and the message counts them."""
    ph = problem().phases.first
    with raises(ValueError, "a guess is (first, last); got 3 values", at="guess.x"):
        ph.state.guess.x = (1, 2, 3)


def test_a_guess_names_both_forms_it_accepts() -> None:
    """When a value is neither, the message says what the two forms are."""
    ph = problem().phases.first
    with raises(TypeError, "(first, last) pair or yapss.interp(...)", at="guess.x"):
        ph.state.guess.x = "a"


def test_samples_are_not_a_bare_pair_of_sequences() -> None:
    """`(t, values)` is too easily confused with the two-point form, so it is refused."""
    ph = problem().phases.first
    with raises(TypeError, "takes two numbers", at="guess.x"):
        ph.state.guess.x = ([0.0, 1.0], [2.0, 3.0])


# ------------------------------------------------------------------------ yapss.interp


def test_interp_needs_one_value_per_time() -> None:
    """The two arrays describe the same samples, so they are the same length."""
    with raises(ValueError, "3 times but 2 values", at="yapss.interp"):
        yapss.interp([0.0, 0.5, 1.0], [1.0, 2.0])


def test_interp_times_must_increase() -> None:
    """Samples out of order are a mistake rather than something to sort silently."""
    with raises(ValueError, "strictly increasing", at="yapss.interp"):
        yapss.interp([1.0, 0.0], [1.0, 2.0])


def test_interp_needs_at_least_two_times() -> None:
    """One sample is a constant, which the pair form already says."""
    with raises(ValueError, "at least two increasing times", at="yapss.interp"):
        yapss.interp([], [])


# --------------------------------------------------------------- the phase's time guess


def test_the_time_guess_is_a_pair() -> None:
    """The phase's extent is two numbers, and the message names them as t0 and tf."""
    ph = problem().phases.first
    with raises(TypeError, "time guess is a (t0, tf) pair", at="time.guess"):
        ph.time.guess = 0.5


def test_the_time_guess_takes_exactly_two() -> None:
    """A mesh is not given here, so three times is a mistake about what this is."""
    ph = problem().phases.first
    with raises(TypeError, "(t0, tf) pair", at="time.guess"):
        ph.time.guess = (0.0, 1.0, 2.0)


def test_a_phase_without_a_time_guess_is_incomplete() -> None:
    """Required, because everything else sampled is placed against it -- and it says how."""

    class OnePhase(yapss.Phases):
        only = yapss.phase(state=State, control=Control)

    p = yapss.Problem("p", phases=OnePhase)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]
        return out

    @p.register.objective
    def objective(arg):
        return arg[ph].final.x

    with raises(ValueError, "has no time guess", "ph.time.guess = (start, end)", at="validate"):
        p.validate()


def test_the_independent_variable_is_named_in_the_complaint() -> None:
    """A phase that renamed its independent variable is told about *that* name."""

    class Radial(yapss.Phases):
        nose = yapss.phase(state=State, control=Control, r=yapss.field())

    p = yapss.Problem("p", phases=Radial)
    ph = p.phases.nose

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]
        return out

    @p.register.objective
    def objective(arg):
        return arg[ph].final.x

    with raises(ValueError, "has no r guess", "ph.r.guess = (start, end)", at="validate"):
        p.validate()


# ------------------------------------------------------- samples must cover the phase


def test_samples_that_fall_well_short_are_refused() -> None:
    """Samples covering a fraction of the phase are a mistake, and the message does the sum."""
    p = solvable()
    ph = p.phases.slide
    ph.state.guess.x = yapss.interp([0.0, 0.1], [0.0, 0.2])
    with raises(ValueError, "samples end at", "must reach", at="validate"):
        p.validate()


def test_samples_that_fall_a_little_short_hold_their_end_values() -> None:
    """Within the tolerance, the ends are held, as ``numpy.interp`` does: never a trend."""
    p = solvable()
    ph = p.phases.slide
    ph.state.guess.x = yapss.interp([0.02, 0.98], [0.0, 1.0])
    p.validate()
    assert p.solve().converged


def test_a_guess_outside_the_bounds_is_clipped_rather_than_refused() -> None:
    """Ipopt moves the starting point inside the bounds; YAPSS does it first, and says nothing."""
    p = solvable()
    ph = p.phases.slide
    ph.state.guess.v = (-5.0, 50.0)
    assert p.solve().converged
    assert np.isfinite(p.solve().objective)


# --------------------------------------------------------- a row of values over the phase


def test_a_boolean_is_not_a_guess() -> None:
    """The same rule as a bound's: a comparison written by accident is caught."""
    ph = problem().phases.first
    with raises(TypeError, "a boolean is not a number", at="guess.x"):
        ph.state.guess.x = True


def test_the_time_guess_refuses_a_boolean_too() -> None:
    """Every aspect that takes a number refuses one, so the rule has no exceptions."""
    ph = problem().phases.first
    with raises(TypeError, "t0 and tf are numbers", at="time.guess"):
        ph.time.guess = (False, True)


def test_a_sampled_row_is_a_scalar_or_one_value_per_time() -> None:
    """Samples of the wrong length are a mistake about which grid they are on."""
    with raises(ValueError, "one value per time", at="yapss.interp"):
        yapss.interp([0.0, 1.0], [[0.0, 1.0, 2.0]])


def test_a_block_field_takes_a_row_of_samples_each() -> None:
    """`(size, n)` samples give each row of a block its own history."""
    ph = problem().phases.first
    t = np.linspace(0.0, 1.0, 5)
    ph.state.guess.y[:] = yapss.interp(t, np.vstack([t, 2 * t]))
    assert ph.state.guess.y is not None


def test_the_ends_of_the_time_guess_are_numbers() -> None:
    """t0 and tf are two numbers, and the message names them as such."""
    ph = problem().phases.first
    with raises(TypeError, "t0 and tf are numbers", at="time.guess"):
        ph.time.guess = (0.0, "one")


def test_a_sampled_row_of_the_wrong_length_is_refused() -> None:
    """Samples are one value per time, so an array of another shape cannot be placed."""
    with raises(ValueError, "one value per time", at="yapss.interp"):
        yapss.interp([0.0, 0.5, 1.0], np.zeros((2, 4)))


def test_a_parameter_is_guessed_as_one_number() -> None:
    """A parameter does not vary over a phase, so a trajectory is a mistake about what it is."""
    p = problem()
    with raises(
        (TypeError, ValueError),
        "one number",
        at="parameter.guess.s",
    ):
        p.parameter.guess.s = (0.0, 1.0)
