"""

Test the Ipopt options YAPSS sets on the user's behalf.

YAPSS applies two defaults on the vendored (CasADi-bundled) path: `linear_solver`
is set to MUMPS, and on macOS `mumps_pivot_order` is set to QAMD. Both are applied
only when the user has not chosen a value, and both are wrapped in
`contextlib.suppress(ValueError, TypeError)` -- so if either were rejected, the
solve would continue on the old setting with nothing to indicate it. That is the
failure these tests exist to prevent.

The suite is in two halves, because "YAPSS asked for it" and "Ipopt took it" are
different claims and only the second one matters to a user:

* The `record_options` tests assert the *decision* -- which options YAPSS requests,
  for whom, and in what order -- by recording calls to `MseipoptProblem.add_option`.
* `test_ipopt_accepts_the_mumps_default` asserts that Ipopt echoed the option back,
  which is the only evidence that it took effect. A benchmark in this project once
  compared a solver against itself for want of exactly this check.

**What these tests do NOT cover, and deliberately so.** `sys.platform` is
monkeypatched, so the macOS tests verify the decision logic and nothing about macOS.

There is no crash-regression test here, on purpose. The proposition being defended is
"METIS always crashes on macOS, and YAPSS no longer selects METIS" -- so the thing to
assert is *which ordering is in effect*, which is a static fact about the option. A
solve that survives at one mesh says nothing about the next mesh up; it is an
existence proof against a claim nobody made. The crashing behavior belongs to CasADi's
build, was characterized once during the bug hunt, and is not a per-release regression
risk in this repository. A "did not crash" assertion would also pass vacuously the day
CasADi 3.8.0 fixes it upstream.

That QAMD is genuinely honored rather than silently ignored was established
empirically on macOS, not here -- which is also how PORD was found to be an alias for
METIS.

**Sabotage-verified.** Each condition in `solver.py` was inverted in turn -- both
`ipopt_source` guards dropped, the `"darwin"` comparison flipped, each
`not in get_options()` check removed, and the two blocks swapped -- and the
corresponding test confirmed to fail. `Temporary/sabotage_check.py` automates it;
re-run it after changing either block.

The first run of that script reported all five mutations unnoticed, which was itself
wrong: tox installs YAPSS as a normal package, so the tests were importing
`.tox/.../site-packages/yapss` while the script edited `src/`. The script now sets
`PYTHONPATH` and refuses to run unless `yapss` resolves inside `src/`. A verification
that silently checks the wrong copy reports a clean result, not an error.

`test_explicit_library_path_gets_neither_default` exists because of what that run found
once it was fixed: seven tests, and none of them exercised the `ipopt_source` guard.

Conda is skipped throughout: there the backend is cyipopt and neither default is
applied, by design -- a Conda user's Ipopt is their own and may be built against
HSL.

"""

from __future__ import annotations

import types

import pytest

from yapss._private import solver
from yapss.examples.rosenbrock import setup

pytestmark = pytest.mark.skipif(
    solver._IN_CONDA,
    reason="the vendored-path defaults do not apply in a Conda environment",
)


@pytest.fixture
def problem():
    """A small problem that converges cleanly.

    `tol` is reset from the example's deliberately unattainable 1e-20, which exits
    non-converged by construction and would raise `IpoptConvergenceWarning` in every
    test here -- noise that has nothing to do with what is being measured.
    """
    ocp = setup()
    ocp.ipopt_options.tol = 1e-8
    ocp.ipopt_options.print_level = 0
    return ocp


@pytest.fixture
def record_options(monkeypatch):
    """Record every `add_option` call, then delegate to the real one.

    Delegating matters: a pure mock would let a value through that Ipopt would
    reject, and the test would assert that YAPSS asked for something impossible.
    """
    calls: list[tuple[str, object]] = []
    original = solver.MseipoptProblem.add_option

    def spy(self, keyword, value):
        calls.append((keyword, value))
        return original(self, keyword, value)

    monkeypatch.setattr(solver.MseipoptProblem, "add_option", spy)
    return calls


def set_platform(monkeypatch, name):
    """Make `solver` see a given `sys.platform`, without touching the real `sys`.

    Replacing the module's own `sys` name rather than patching `sys.platform`
    globally: the solve underneath runs NumPy and CasADi, and a lying `sys.platform`
    is not a safe thing to hand them.
    """
    monkeypatch.setattr(solver, "sys", types.SimpleNamespace(platform=name))


def keywords(calls):
    return [keyword for keyword, _ in calls]


class TestLinearSolverDefault:
    """MUMPS is selected unless the user chose otherwise."""

    def test_mumps_is_the_default(self, problem, record_options, monkeypatch):
        set_platform(monkeypatch, "linux")
        problem.solve()
        assert ("linear_solver", "mumps") in record_options

    @pytest.mark.filterwarnings("ignore::yapss._private.solver.IpoptOptionSettingWarning")
    def test_user_choice_is_not_overridden(self, problem, record_options, monkeypatch):
        # "spral" regardless of whether this build has it. Where it is absent -- macOS,
        # and Conda outside Linux -- Ipopt rejects the value and YAPSS warns; the
        # warning is filtered because it is not what this test is about. Either way
        # the user chose, and YAPSS must not substitute its own.
        set_platform(monkeypatch, "linux")
        problem.ipopt_options.linear_solver = "spral"
        problem.solve()
        assert ("linear_solver", "mumps") not in record_options
        assert ("linear_solver", "spral") in record_options


class TestPivotOrderDefault:
    """QAMD is selected on macOS only, and only when the user chose nothing."""

    def test_set_on_darwin(self, problem, record_options, monkeypatch):
        set_platform(monkeypatch, "darwin")
        problem.solve()
        assert ("mumps_pivot_order", 6) in record_options

    def test_not_set_elsewhere(self, problem, record_options, monkeypatch):
        # Patched even on a Linux runner: on a macOS runner the unpatched value
        # would make this test pass for the wrong reason.
        set_platform(monkeypatch, "linux")
        problem.solve()
        assert "mumps_pivot_order" not in keywords(record_options)

    def test_user_choice_is_not_overridden(self, problem, record_options, monkeypatch):
        set_platform(monkeypatch, "darwin")
        problem.ipopt_options.mumps_pivot_order = 0
        problem.solve()
        assert ("mumps_pivot_order", 6) not in record_options
        assert ("mumps_pivot_order", 0) in record_options


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_explicit_library_path_gets_neither_default(problem, record_options, monkeypatch):
    """Neither default is applied when the user supplied their own Ipopt library.

    Both defaults are guarded by ``ipopt_source == "casadi"``, and nothing else in this
    file exercises the other branch -- a sabotage run that removed that guard from the
    `linear_solver` block went unnoticed by all seven other tests.

    The bundled library's own path is used as the "explicit" path. It is already loaded
    in this process, so `ctypes.CDLL` returns the same handle and no second copy of
    Ipopt is mapped -- which is the hazard the whole backend design exists to avoid.
    Nothing here depends on the library being a *different* one; what matters is only
    that `ipopt_source` is no longer the string ``"casadi"``.

    The platform is set to darwin so that the ordering default would fire too if its
    guard were removed, which makes this one test cover both.
    """
    from yapss._private.mseipopt import initialize_ipopt

    path = initialize_ipopt()
    set_platform(monkeypatch, "darwin")
    problem.ipopt_source = path

    problem.solve()

    assert ("linear_solver", "mumps") not in record_options
    assert "mumps_pivot_order" not in keywords(record_options)


def test_linear_solver_is_set_before_pivot_order(problem, record_options, monkeypatch):
    """Ordering is load-bearing, not incidental.

    `mumps_pivot_order` means nothing unless MUMPS is the solver in use, so the two
    blocks in `solver.py` cannot be swapped. Nothing else would catch that.
    """
    set_platform(monkeypatch, "darwin")
    problem.solve()
    names = keywords(record_options)
    assert names.index("linear_solver") < names.index("mumps_pivot_order")


def test_ipopt_accepts_the_mumps_default(problem, tmp_path):
    """Ipopt echoed `linear_solver = mumps`, so the option actually took effect.

    The decision tests above would all pass if `add_option` silently discarded the
    value -- `contextlib.suppress` in `solver.py` would hide the exception, and the
    solve would run on SPRAL. This is the test that fails in that case.
    """
    log = tmp_path / "ipopt.log"
    problem.ipopt_options.print_user_options = "yes"
    problem.ipopt_options.output_file = str(log)
    problem.ipopt_options.file_print_level = 5

    problem.solve()

    text = log.read_text(encoding="utf-8", errors="replace")
    assert "linear_solver = mumps" in text, (
        "Ipopt did not echo the linear_solver option; YAPSS's default was requested "
        "but not applied. Ipopt's echo is the authoritative answer -- the absence of "
        "a warning is not, since the call is wrapped in contextlib.suppress."
    )
