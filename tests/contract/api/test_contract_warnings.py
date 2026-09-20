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

from ._api import not_yet, solvable, warns

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


@not_yet(
    "gap",
    "LargeSegmentWarning is raised by the 0.3.0 front end only; the new mesh never warns",
)
def test_a_very_large_segment_warns() -> None:
    """A segment of many points is valid input whose cost deserves a mention.

    The category is exported, documented on the warnings page, and unreachable through this
    front end: the check lives in `_legacy/problem.py`, and `_api/mesh.py` has nothing like
    it. A user who writes `Mesh([(1.0, 60)])` is told nothing.
    """
    problem = solvable()
    with warns(yapss.LargeSegmentWarning, "usually better split", at="ph.mesh"):
        ph = problem.phases.slide
        ph.mesh = yapss.Mesh([(1.0, 60)])
