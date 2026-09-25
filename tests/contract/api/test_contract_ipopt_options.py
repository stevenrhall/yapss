"""What YAPSS passes to Ipopt, what it refuses to pass, and what it only warns about.

Two rules divide this page. What is wrong on every build is refused where it is written: an
option YAPSS sets itself, because letting a user set it would make YAPSS's own choice silently
ineffective. Everything else is Ipopt's to judge, because which options exist and which values
they take depend on the build: an option is passed on, and if Ipopt refuses it, the solve warns
and continues with Ipopt's default -- valid input with an outcome worth mentioning, which is the
line between raising and warning everywhere in YAPSS.
"""

from __future__ import annotations

import warnings

import yapss

from ._api import problem, raises, solvable, warns

# --------------------------------------------------------------------- options YAPSS owns


def test_an_option_yapss_manages_is_refused() -> None:
    """And the message names the YAPSS setting that does the same job."""
    p = problem()
    with raises(
        ValueError,
        "is managed by YAPSS",
        "problem.derivatives",
        at="ipopt_options",
    ):
        p.ipopt_options.hessian_approximation = "limited-memory"


def test_the_scaling_options_are_managed_too() -> None:
    """YAPSS always solves with user-scaling, so these are its own to set."""
    p = problem()
    with raises(ValueError, "is managed by YAPSS", "scales set on", at="ipopt_options"):
        p.ipopt_options.nlp_scaling_method = "gradient-based"


def test_the_objective_scaling_factor_points_at_the_scale() -> None:
    """The message names the objective's sense and scale, which are the supported way."""
    p = problem()
    with raises(
        ValueError,
        "is managed by YAPSS",
        "problem.objective.sense",
        "problem.objective.scale",
        at="ipopt_options",
    ):
        p.ipopt_options.obj_scaling_factor = 2.0


# ------------------------------------------------------------ options Ipopt judges itself


def test_an_unknown_option_is_passed_on_without_comment() -> None:
    """Which options exist is Ipopt's to judge, so nothing is said where one is written."""
    p = problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptOptionSettingWarning)
        p.ipopt_options.no_such_option = 1
        p.ipopt_options.max_iters = 1


def test_an_option_ipopt_refuses_warns_at_the_solve() -> None:
    """Valid input whose outcome deserves attention: warned about, and the solve continues."""
    p = solvable()
    p.ipopt_options.no_such_option = 1
    with warns(
        yapss.IpoptOptionSettingWarning,
        "Ipopt refused option 'no_such_option'",
        "continues with Ipopt's default",
        at="p.solve",
    ):
        solution = p.solve()
    assert solution.converged


def test_a_misspelled_option_is_named_at_the_solve() -> None:
    """The table cannot decide, but it can suggest: a near miss gets its likely intent."""
    p = solvable()
    p.ipopt_options.max_iters = 50
    with warns(yapss.IpoptOptionSettingWarning, "Did you mean 'max_iter'", at="p.solve"):
        p.solve()


def test_a_value_ipopt_refuses_warns_at_the_solve() -> None:
    """A value is Ipopt's to judge too: its refusal warns, and the solve uses the default."""
    p = solvable()
    p.ipopt_options.max_iter = -1
    with warns(yapss.IpoptOptionSettingWarning, "Ipopt refused option 'max_iter'", at="p.solve"):
        solution = p.solve()
    assert solution.converged


def test_a_documented_option_is_set_without_comment() -> None:
    """The ordinary case, which is what makes the warning above worth noticing."""
    p = solvable()
    p.ipopt_options.max_iter = 50
    assert p.solve().converged
