"""Ctrl-C during a solve, with and without YAPSS's SIGINT handler.

With ``catch_keyboard_interrupt = True`` (the default), YAPSS's handler asks Ipopt to stop at
the next iteration, and ``solve()`` returns the current iterate with a convergence warning.
With ``False``, no handler is installed, so the interrupt is raised inside a callback; the
binding retains it while Ipopt owns the stack, stops Ipopt, and ``solve()`` re-raises it, so a
script running many solves can be stopped.

Each case raises SIGINT in its own process, so it runs in a fresh interpreter. It uses
``signal.raise_signal``, not ``os.kill``: on Windows, ``os.kill`` with ``SIGINT`` terminates
the process with exit code 2 instead of delivering the signal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.isolation

SCRIPT = textwrap.dedent("""
    import signal, sys, warnings
    from yapss.examples import brachistochrone_minimal
    from yapss.math import cos, sin

    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0
    problem.derivatives.method = "central-difference"
    problem.catch_keyboard_interrupt = sys.argv[1] == "True"
    handler = signal.getsignal(signal.SIGINT)
    calls = [0]

    @problem.phases.phase.register.continuous
    def continuous(arg, out):
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = brachistochrone_minimal.G0 * sin(u)
        calls[0] += 1
        if calls[0] == 40:  # well inside the solve: the user presses Ctrl-C
            signal.raise_signal(signal.SIGINT)

    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            solution = problem.solve()
        outcome = solution.status.name
    except KeyboardInterrupt:
        outcome = "KeyboardInterrupt"
    assert signal.getsignal(signal.SIGINT) is handler, "SIGINT handler not restored"
    print(outcome)
    """)


@pytest.mark.parametrize(
    ("catch", "expected"),
    [(True, "USER_REQUESTED_STOP"), (False, "KeyboardInterrupt")],
)
def test_ctrl_c_during_a_solve(catch: bool, expected: str) -> None:
    process = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(catch)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, output
    assert process.stdout.strip().splitlines()[-1] == expected, output
