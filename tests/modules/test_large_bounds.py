"""Every finite bound is a bound, however large; only an infinite bound is no bound.

Ipopt treats a bound at or beyond ``nlp_upper_bound_inf`` or ``nlp_lower_bound_inf`` (default
1e19 in magnitude) as no bound. YAPSS sets both to the largest finite double, so a bound of 1e20
holds, while an infinite bound still means no bound.

The problem optimizes one parameter under a bound, so the optimum is the bound itself; if the
bound were dropped, the iterates would diverge.
"""

from __future__ import annotations

import numpy as np
import pytest

import yapss


class Design(yapss.Parameter):
    p = yapss.scalar()


class Phases(yapss.Phases):
    pass


class Bounded(yapss.Problem):
    phases: Phases
    parameter: Design


def bounded_parameter(lower: float, upper: float, sense: str) -> Bounded:
    """Optimize one parameter p in [lower, upper], scaled by the finite bound's magnitude.

    The scale stops at 1e100.
    """
    magnitude = upper if np.isfinite(upper) else -lower
    problem = Bounded("large bound")

    @problem.register.objective
    def objective(arg):
        return arg.parameter.p / magnitude

    problem.objective.sense = sense
    problem.parameter.p.bounds = (lower, upper)
    problem.parameter.p.guess = magnitude / 2 if sense == "maximize" else -magnitude / 2
    problem.parameter.p.scale = min(magnitude, 1e100)
    problem.ipopt_options.print_level = 0
    return problem


@pytest.mark.parametrize("bound", [1e18, 1e20, 1e120])
def test_large_upper_bound_holds(bound: float) -> None:
    solution = bounded_parameter(0.0, bound, "maximize").solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    assert solution.parameter.p == pytest.approx(bound, rel=1e-6)


@pytest.mark.parametrize("bound", [1e18, 1e20, 1e120])
def test_large_lower_bound_holds(bound: float) -> None:
    solution = bounded_parameter(-bound, np.inf, "minimize").solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    assert solution.parameter.p == pytest.approx(-bound, rel=1e-6)


def test_infinite_bound_is_no_bound() -> None:
    """A parameter bounded only by infinities reaches an interior optimum."""
    problem = Bounded("no bound")

    @problem.register.objective
    def objective(arg):
        return (arg.parameter.p - 3.0) ** 2

    problem.parameter.p.bounds = (-np.inf, np.inf)
    problem.parameter.p.guess = 0.0
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    assert solution.parameter.p == pytest.approx(3.0)
