"""What YAPSS passes to Ipopt, what it refuses to pass, and what it only warns about.

Two rules divide this page. An option YAPSS sets itself is refused, because letting a user set
it would make YAPSS's own choice silently ineffective. An option YAPSS does not recognize is
*warned* about and passed on anyway, because the user's Ipopt build may have options the
documented release does not -- valid input with an outcome worth mentioning, which is the line
between raising and warning everywhere in YAPSS.
"""

from __future__ import annotations

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


# ------------------------------------------------------- options YAPSS does not recognize


def test_an_unknown_option_warns_and_is_passed_on() -> None:
    """Valid input whose outcome deserves attention: warned about, not refused."""
    p = problem()
    with warns(
        yapss.IpoptOptionSettingWarning,
        "is not among the options documented",
        at="ipopt_options",
    ):
        p.ipopt_options.no_such_option = 1


def test_a_documented_option_is_set_without_comment() -> None:
    """The ordinary case, which is what makes the warning above worth noticing."""
    p = solvable()
    p.ipopt_options.max_iter = 50
    assert p.solve().converged
