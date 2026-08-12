"""

Load-order tests for the IPOPT resolver, each in a fresh interpreter.

``dlopen`` maps a shared library permanently, so these cases cannot share a
process: once one test has loaded IPOPT, every later one would be observing
its leftovers rather than a clean start. Each check therefore runs
_ipopt_worker.py in a subprocess, which costs a CasADi import (~0.6 s) per
case. Deselect with ``-m "not isolation"`` if that becomes a problem.

"""

import subprocess
import sys
from pathlib import Path

import pytest

from yapss._private.config import get_conda_prefix

WORKER = Path(__file__).parent / "_ipopt_worker.py"

PASS, FAIL, SKIP = 0, 1, 2

pytestmark = [
    pytest.mark.isolation,
    pytest.mark.skipif(
        bool(get_conda_prefix()),
        reason="conda uses the cyipopt backend; the resolver is not used there",
    ),
]


def run_check(name):
    """Run one worker check in its own interpreter and report the outcome."""
    process = subprocess.run(
        [sys.executable, str(WORKER), name],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = process.stdout + process.stderr
    if process.returncode == SKIP:
        pytest.skip(f"{name}: precondition unavailable\n{output}")
    assert process.returncode == PASS, f"{name} failed:\n{output}"
    return output


def test_cold_start():
    """The resolver must work before anything has imported CasADi."""
    run_check("cold_start")


def test_idempotent():
    """Resolving and loading repeatedly must not accumulate copies."""
    run_check("idempotent")


def test_initialize_leaves_one_copy():
    """A full initialize_ipopt() must leave exactly one IPOPT mapped."""
    run_check("initialize")


def test_sabotage_strategy1_survives_a_wrong_glob():
    """THE test: with introspection alive, a wrong glob must not matter.

    Everything else in the suite passes on Linux and macOS even when loader
    introspection is completely dead, because the glob coincidentally picks
    the right file there. This is the only check that fails if strategy 1
    stops working -- the regression guard for the original probe-name bug.
    """
    run_check("sabotage_strategy1_alive")


def test_sabotage_control_shows_the_hazard_is_real():
    """Control: without introspection, the wrong glob maps a second copy.

    If this ever starts failing, guessing filenames has stopped being
    dangerous on this platform and the whole design deserves revisiting.
    """
    run_check("sabotage_strategy1_dead")
