"""

Test that an unconverged Ipopt solve is not silent.

The failure this guards against is not a crash: `Problem.solve` returns a perfectly
ordinary-looking `Solution` whatever Ipopt reports, so a run that hit `max_iter` yields
a plausible trajectory with nothing to distinguish it from a converged one.

"""

import warnings

import pytest

import yapss
from yapss._private.solution import QUIET_IPOPT_STATUSES, warn_if_not_converged
from yapss.examples.rosenbrock import setup


def test_warns_when_max_iter_exceeded():
    """An unconverged solve warns, and the warning names the status."""
    ocp = setup()
    ocp.ipopt_options.max_iter = 1

    with pytest.warns(yapss.IpoptConvergenceWarning, match=r"Status -1\b"):
        solution = ocp.solve()

    # The warning must describe reality, not just fire.
    assert solution.nlp_info.ipopt_status == -1


def test_silent_when_converged():
    """A normal solve emits no convergence warning.

    Without this, `test_warns_when_max_iter_exceeded` would still pass if the warning
    were emitted unconditionally -- which would be worse than no warning at all, since
    a warning that always fires gets filtered out and stops being read.
    """
    ocp = setup()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solution = ocp.solve()

    assert solution.nlp_info.ipopt_status in QUIET_IPOPT_STATUSES
    convergence_warnings = [
        w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)
    ]
    assert convergence_warnings == []


@pytest.mark.parametrize("status", sorted(QUIET_IPOPT_STATUSES))
def test_quiet_statuses_are_silent(status, monkeypatch):
    """Statuses 0, 1 and 6 do not warn.

    1 ("Solved To Acceptable Level") is included deliberately: it is the normal outcome
    when tolerances are pushed hard, and warning on it would train users to ignore the
    warning entirely.
    """

    class FakeNLPInfo:
        ipopt_status = status

    class FakeSolution:
        nlp_info = FakeNLPInfo()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_not_converged(FakeSolution())

    assert caught == []


@pytest.mark.parametrize("status", [-1, -2, -13, 2, 4, 5, -199, 12345])
def test_other_statuses_warn(status):
    """Every non-quiet status warns, including codes with no message text."""

    class FakeNLPInfo:
        ipopt_status = status

    class FakeSolution:
        nlp_info = FakeNLPInfo()

    with pytest.warns(yapss.IpoptConvergenceWarning, match=rf"Status {status}\b"):
        warn_if_not_converged(FakeSolution())


def test_warning_is_public():
    """The category is importable from the public namespace, so it can be filtered."""
    assert yapss.IpoptConvergenceWarning.__name__ in yapss.__all__
    assert issubclass(yapss.IpoptConvergenceWarning, Warning)
