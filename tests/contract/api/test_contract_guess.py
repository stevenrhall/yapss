"""What an initial guess is, and what a guess that cannot be used says.

A guess is one of three things -- a number held over the phase, a `(first, last)` pair, or
`yapss.interp(t, values)` -- and the phase's time guess is the only source of the phase's
guessed extent, which is what lets a sampled guess carry no times of its own to disagree with.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import yapss

from ._api import Angle, Control, State, problem, raises, solvable

# ----------------------------------------------------------------- the three accepted forms


def test_a_pair_is_linear_from_one_end_to_the_other() -> None:
    """The commonest guess: where the row starts and where it ends.

    It reads back tagged with what kind of guess it is, since the three forms have to be told
    apart later, and a pair of numbers cannot say which of them it is.
    """
    ph = problem().phases.first
    ph.state.x.guess = (0.0, 10.0)
    assert ph.state.x.guess == ("linear", 0.0, 10.0)


def test_samples_carry_their_own_times() -> None:
    """A sampled guess is pure data on its own grid, so rows need not share one."""
    ph = problem().phases.first
    ph.state.x.guess = yapss.interp([0.0, 0.5, 1.0], [0.0, 2.0, 1.0])
    assert ph.state.x.guess is not None


def test_a_block_field_takes_one_guess_per_row() -> None:
    """Rows of a block are guessed as the rows of any other aspect are written."""
    ph = problem().phases.first
    ph.state.y.guess[:] = [(0.0, 1.0), (2.0, 3.0)]
    assert list(ph.state.y.guess) == [("linear", 0.0, 1.0), ("linear", 2.0, 3.0)]


# ------------------------------------------------------------------------ what is refused


def test_a_guess_is_a_pair_not_a_number() -> None:
    """One number is ambiguous between a constant and a typo, so the message offers the fix."""
    ph = problem().phases.first
    with raises(
        TypeError,
        "a guess is a (first, last) pair",
        "write (0.5, 0.5)",
        at="x.guess",
    ):
        ph.state.x.guess = 0.5


def test_a_guess_takes_exactly_two_values() -> None:
    """Three values is neither a pair nor samples, and the message counts them."""
    ph = problem().phases.first
    with raises(ValueError, "a guess is (first, last); got 3 values", at="x.guess"):
        ph.state.x.guess = (1, 2, 3)


def test_a_guess_names_both_forms_it_accepts() -> None:
    """When a value is neither, the message says what the two forms are."""
    ph = problem().phases.first
    with raises(TypeError, "(first, last) pair or yapss.interp(...)", at="x.guess"):
        ph.state.x.guess = "a"


def test_samples_are_not_a_bare_pair_of_sequences() -> None:
    """`(t, values)` is too easily confused with the two-point form, so it is refused."""
    ph = problem().phases.first
    with raises(TypeError, "takes two numbers", at="x.guess"):
        ph.state.x.guess = ([0.0, 1.0], [2.0, 3.0])


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

    class Only(yapss.Phase):
        state: State
        control: Control
        time: yapss.Independent

    class OnePhase(yapss.Phases):
        only: Only

    p = yapss.Problem("p", phases=OnePhase)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]

    @p.register.objective
    def objective(arg):
        return arg[ph].final.x

    with raises(ValueError, "has no time guess", "ph.time.guess = (start, end)", at="validate"):
        p.validate()


def test_the_independent_variable_is_named_in_the_complaint() -> None:
    """A phase that renamed its independent variable is told about *that* name."""

    class Nose(yapss.Phase):
        state: State
        control: Control
        r: yapss.Independent

    class Radial(yapss.Phases):
        nose: Nose

    p = yapss.Problem("p", phases=Radial)
    ph = p.phases.nose

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]

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
    ph.state.x.guess = yapss.interp([0.0, 0.1], [0.0, 0.2])
    with raises(ValueError, "samples end at", "must reach", at="validate"):
        p.validate()


def test_samples_that_fall_a_little_short_hold_their_end_values() -> None:
    """Within the tolerance, the ends are held, as ``numpy.interp`` does: never a trend."""
    p = solvable()
    ph = p.phases.slide
    ph.state.x.guess = yapss.interp([0.02, 0.98], [0.0, 1.0])
    p.validate()
    assert p.solve().converged


def test_a_guess_outside_the_bounds_is_clipped_rather_than_refused() -> None:
    """Ipopt moves the starting point inside the bounds; YAPSS does it first, and says nothing."""
    p = solvable()
    ph = p.phases.slide
    ph.state.v.guess = (-5.0, 50.0)
    assert p.solve().converged
    assert np.isfinite(p.solve().objective)


# --------------------------------------------------------- a row of values over the phase


def test_a_boolean_is_not_a_guess() -> None:
    """The same rule as a bound's: a comparison written by accident is caught."""
    ph = problem().phases.first
    with raises(TypeError, "a boolean is not a number", at="x.guess"):
        ph.state.x.guess = True


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
    ph.state.y.guess[:] = yapss.interp(t, np.vstack([t, 2 * t]))
    assert ph.state.y.guess is not None


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
        at="parameter.s.guess",
    ):
        p.parameter.s.guess = (0.0, 1.0)


# ------------------------------------------------------------- what interp() accepts


def test_interp_keeps_its_own_copy() -> None:
    """Changing the caller's arrays afterwards cannot change a guess that was already checked."""
    t = np.array([0.0, 1.0])
    v = np.array([0.0, 5.0])
    guess = yapss.interp(t, v)
    t[1] = -1.0
    v[1] = np.nan
    assert list(guess.time) == [0.0, 1.0]
    assert list(guess.values) == [0.0, 5.0]


@pytest.mark.parametrize("values", [["1", "2"], [True, False], [0.0, None], np.array([1 + 1j, 2])])
def test_interp_takes_real_numbers_only(values: object) -> None:
    """Converting would turn "2", True and None into numbers, and drop an imaginary part."""
    with raises(TypeError, "interp(values=) takes real numbers", at="interp"):
        yapss.interp([0.0, 1.0], values)


@pytest.mark.parametrize(
    ("time", "values", "bad"),
    [
        ([0.0, 1.0], [0.0, np.nan], "values[1] is nan"),
        ([0.0, 1.0], [0.0, np.inf], "values[1] is inf"),
        ([0.0, 1.0, np.inf], [0.0, 1.0, 5.0], "time[2] is inf"),
        ([0.0, np.nan, 1.0], [0.0, 1.0, 5.0], "time[1] is nan"),
        ([0.0, 1.0], [[0.0, 1.0], [2.0, np.nan]], "values[1, 1] is nan"),
    ],
)
def test_interp_samples_are_finite(time: list[float], values: object, bad: str) -> None:
    """A NaN or infinite sample is refused where it is given, naming the first bad sample."""
    what = bad.split("[")[0]
    with raises(ValueError, f"interp({what}=) must be finite", bad, at="interp"):
        yapss.interp(time, values)


def test_several_rows_of_samples_fit_only_the_whole_field() -> None:
    """Given to one row, one of the sampled rows was taken without a word."""
    ph = problem().phases.first
    two_rows = yapss.interp([0.0, 1.0], np.zeros((2, 2)))
    with raises(ValueError, "have 2 rows", "covers 1", "give one row of samples", at="y.guess[0]"):
        ph.state.y.guess[0] = two_rows
    with raises(ValueError, "have 2 rows", "the field has one", at="x.guess"):
        ph.state.x.guess = two_rows
    ph.state.y.guess[:] = two_rows


def test_the_row_count_of_samples_is_checked_where_they_are_assigned() -> None:
    """Three rows for a two-row field were refused only when the problem was solved."""
    ph = problem().phases.first
    with raises(ValueError, "have 3 rows", "field's 2", at="y.guess[:]"):
        ph.state.y.guess[:] = yapss.interp([0.0, 1.0], np.zeros((3, 2)))


# ------------------------------------------------------------------ a guess is finite


@pytest.mark.parametrize("pair", [(0.0, np.nan), (np.inf, 0.0), (0.0, -np.inf)])
def test_a_pair_guess_is_finite(pair: tuple[float, float]) -> None:
    """Accepted, it would reach the callbacks or Ipopt, and be reported as the callbacks' fault."""
    ph = solvable().phases.slide
    with raises(ValueError, "control guess 'theta'", "must be finite", at="theta.guess"):
        ph.control.theta.guess = pair


def test_a_one_number_guess_is_finite() -> None:
    """Integrals and parameters are guessed as one number, held to the same rule."""
    ph = solvable().phases.slide
    with raises(ValueError, "integral guess 'effort'", "must be finite", at="effort.guess"):
        ph.integral.effort.guess = np.inf


@pytest.mark.parametrize("guess", [(0.0, np.inf), (0.0, np.nan), (-np.inf, 1.0)])
def test_the_time_guess_is_finite(guess: tuple[float, float]) -> None:
    """Refused for what it is, not for the order its ends happen to compare in."""
    ph = solvable().phases.slide
    with raises(ValueError, "time guess", "must be finite", at="time.guess"):
        ph.time.guess = guess


@pytest.mark.parametrize("guess", [(1.0, 0.0), (1.0, 1.0)])
def test_the_time_guess_increases(guess: tuple[float, float]) -> None:
    """The phase runs forward: t0 must be less than tf."""
    ph = solvable().phases.slide
    with raises(ValueError, "is not less than tf", at="time.guess"):
        ph.time.guess = guess


# --------------------------------------------------------------- a guess from a solution


def _solved() -> Any:
    p = solvable()
    p.ipopt_options.print_level = 0
    return p.solve()


@pytest.mark.parametrize("method", ["lg", "lgr", "lgl"])
def test_a_warm_start_converges_quickly_on_another_mesh_and_method(method: str) -> None:
    """The guess is written from the solution, and it is a good one."""
    solution = _solved()
    cold = solvable()
    warm = solvable()
    for p in (cold, warm):
        p.phases.slide.mesh = yapss.Mesh.uniform(segments=4, points=6)
        p.spectral_method = method
        p.ipopt_options.print_level = 0
    warm.guess_from_solution(solution)
    cold_solution, warm_solution = cold.solve(), warm.solve()
    assert warm_solution.converged
    assert abs(warm_solution.objective - cold_solution.objective) < 1e-6
    assert warm_solution.nlp.convergence.iterations < cold_solution.nlp.convergence.iterations


def test_a_warm_start_writes_ordinary_guesses() -> None:
    """What is written can be read and changed afterwards like any other guess."""
    solution = _solved()
    p = solvable()
    p.guess_from_solution(solution)
    ps = solution[p.phases.slide]
    assert p.phases.slide.time.guess == (float(ps.time[0]), float(ps.time[-1]))
    p.phases.slide.state.v.guess = (0.0, 4.0)


def test_a_mismatched_solution_is_refused_listing_every_mismatch() -> None:
    """Matching loosely would be a silent failure; nothing is written when anything differs."""

    class Other(yapss.State):
        x = yapss.scalar()
        extra = yapss.scalar()

    class OtherPhase(yapss.Phase):
        state: Other
        control: Angle
        time: yapss.Independent

    class OtherPhases(yapss.Phases):
        slide: OtherPhase

    q = yapss.Problem("other", phases=OtherPhases)
    before = q.phases.slide.time.guess
    with raises(
        ValueError,
        "does not match",
        "state 'y' is in the solution but not the problem",
        "state 'extra' is in the problem but not the solution",
        "integral 'effort' is in the solution but not the problem",
        at="guess_from_solution",
    ):
        q.guess_from_solution(_solved())
    assert q.phases.slide.time.guess == before


def test_one_phase_may_be_paired_with_another() -> None:
    """The per-phase form covers subsets and renames, and needs both of its arguments."""
    solution = _solved()
    p = solvable()
    p.guess_from_solution(solution, solution_phase="slide", guess_phase=p.phases.slide)
    with raises(TypeError, "give both", at="guess_from_solution"):
        p.guess_from_solution(solution, solution_phase="slide")
