"""What YAPSS warns about, rather than refuses.

The rule everywhere: raise when YAPSS cannot give a correct answer from the input or it is
almost certainly a mistake; warn only about valid input whose outcome or environment deserves
attention. Every warning is a `YapssWarning`, so one filter turns the lot into errors, and
every warning points at the user's own line, so a filter by module would miss them all.
"""

from __future__ import annotations

import warnings

import pytest

import yapss

from ._api import solvable, warns

# ----------------------------------------------------------------- the categories exist


def test_every_category_is_a_yapss_warning() -> None:
    """One filter is strict mode, which is only true if the hierarchy holds."""
    for category in (
        yapss.IpoptConvergenceWarning,
        yapss.IpoptOptionSettingWarning,
        yapss.LargeSegmentWarning,
        yapss.YapssDeprecationWarning,
    ):
        assert issubclass(category, yapss.YapssWarning)
    assert issubclass(yapss.YapssWarning, UserWarning)


def test_a_deprecation_warning_is_also_a_future_warning() -> None:
    """Python hides DeprecationWarning outside __main__, so a user would never see it."""
    assert issubclass(yapss.YapssDeprecationWarning, FutureWarning)


# --------------------------------------------------------------------- not converging


def test_a_solve_that_does_not_converge_warns_and_returns() -> None:
    """A Solution comes back for any status: the iterate is worth looking at."""
    problem = solvable()
    problem.ipopt_options.max_iter = 1
    with warns(yapss.IpoptConvergenceWarning, "did not converge", at="problem.solve()"):
        result = problem.solve()
    assert not result.converged
    assert result.status is not None


def test_the_warning_names_the_status_ipopt_reported() -> None:
    """Which status it was is the actionable part."""
    problem = solvable()
    problem.ipopt_options.max_iter = 1
    with warns(yapss.IpoptConvergenceWarning, "Maximum Number of Iterations Exceeded"):
        problem.solve()


def test_strict_mode_turns_a_warning_into_an_error() -> None:
    """What `simplefilter("error", yapss.YapssWarning)` is for."""
    problem = solvable()
    problem.ipopt_options.max_iter = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.YapssWarning)
        with pytest.raises(yapss.IpoptConvergenceWarning):
            problem.solve()


def test_a_converged_solve_says_nothing() -> None:
    """The ordinary case, which is what makes the warning above worth noticing."""
    problem = solvable()
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.YapssWarning)
        assert problem.solve().converged


# ------------------------------------------------------------------- a segment too large


def test_a_very_large_segment_warns() -> None:
    """A segment of many points is valid input whose cost deserves a mention."""
    problem = solvable()
    with warns(yapss.LargeSegmentWarning, "usually better split", at="ph.mesh"):
        ph = problem.phases.slide
        ph.mesh = yapss.Mesh([(1.0, 60)])


def test_the_unsupported_function_error_is_a_yapss_error_and_a_type_error() -> None:
    """Caught as YAPSS's own, or as the built-in it specializes."""
    assert issubclass(yapss.UnsupportedMathFunctionError, yapss.YapssError)
    assert issubclass(yapss.UnsupportedMathFunctionError, TypeError)
