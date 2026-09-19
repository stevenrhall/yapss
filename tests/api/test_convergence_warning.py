"""An unconverged solve through the redesigned API warns, at the caller's own line.

A `Solution` is returned for every Ipopt status: an unconverged solve is valid input whose
outcome deserves attention, not a contract violation, which is the rule CLAUDE.md's
conventions state. So the solve returns and warns rather than raising.

The warning is emitted from `Problem.solve` rather than from inside the solve, for the two
reasons the released API has: its `stacklevel` then points at the caller's own `solve()`, and
a solve run repeatedly from inside YAPSS would otherwise warn once per pass. It reuses
`_private.solution.warn_if_not_converged`, so there is one message and one place that decides
which statuses are quiet.
"""

import warnings

import pytest

import yapss
from yapss.examples.brachistochrone import setup


def unconverged():
    """Return a problem that cannot converge in the iterations it is allowed."""
    problem = setup()
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.max_iter = 2
    return problem


def converged():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_a_converged_solve_is_quiet():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solution = converged().solve()
    assert solution.converged
    assert [w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)] == []


def test_an_unconverged_solve_warns():
    with pytest.warns(yapss.IpoptConvergenceWarning, match="Ipopt did not converge"):
        solution = unconverged().solve()
    assert not solution.converged


def test_the_solution_is_returned_rather_than_raised():
    """The trajectory is still there to look at, which is why this warns instead of raising."""
    problem = unconverged()
    with pytest.warns(yapss.IpoptConvergenceWarning):
        solution = problem.solve()
    ps = solution[problem.phases.slide]
    assert ps.state.x.shape == ps.time.shape
    assert solution.status == yapss.IpoptStatus.MAXIMUM_ITERATIONS_EXCEEDED


def test_the_warning_points_at_the_caller():
    """`stacklevel` must reach user code, not `_api` internals."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unconverged().solve()
    convergence = [w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)]
    assert len(convergence) == 1
    assert convergence[0].filename == __file__


@pytest.mark.parametrize("action", ["default", "once", "always"])
def test_every_unconverged_solve_warns_even_from_one_line(action):
    """Three solves from one line warn three times, under every reporting action.

    Python's "default" and "once" actions report a warning from one location only once, which
    is right for a deprecation notice and wrong here: a solve that quietly returns a
    non-optimal trajectory is what this warning exists to prevent. `warn_if_not_converged`
    clears its own entry in the caller's registry first, and this pins that the clearing still
    finds the right frame when the call arrives through `_api.Problem.solve`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action, yapss.IpoptConvergenceWarning)
        for _ in range(3):
            unconverged().solve()
    convergence = [w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)]
    assert len(convergence) == 3


def test_strict_mode_turns_it_into_an_error():
    """Every YAPSS warning is a `YapssWarning`, so one filter is strict mode."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.YapssWarning)
        with pytest.raises(yapss.IpoptConvergenceWarning):
            unconverged().solve()


def test_validate_still_runs_first():
    """An incomplete problem raises before Ipopt is reached, so there is nothing to warn about."""
    problem = setup()
    problem.phases.slide._continuous = None
    with pytest.raises(ValueError, match="has no continuous callback"):
        problem.solve()
