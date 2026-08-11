"""Crash-oriented mseipopt hardening checks run in fresh interpreters."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from yapss._private.config import get_conda_prefix

WORKER = Path(__file__).parent / "_mseipopt_hardening_worker.py"
PASS, SKIP = 0, 2

pytestmark = [
    pytest.mark.isolation,
    pytest.mark.skipif(
        bool(get_conda_prefix()),
        reason="conda uses cyipopt; mseipopt is not the active backend",
    ),
]


def run_check(name: str) -> str:
    """Run one hazardous check without risking the main pytest process."""
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
    assert process.returncode == PASS, f"{name} failed (exit {process.returncode}):\n{output}"
    return output


def test_memory_boundaries_and_callback_lifetime() -> None:
    """Unsafe sparse/buffer inputs fail cleanly and callbacks survive native use."""
    run_check("memory_boundaries")


def test_each_values_callback_failure_has_a_native_non_success_status() -> None:
    """Persistent false/InvalidPoint results cannot become successful solves."""
    run_check("callback_failures")


def test_yapss_sigint_returns_unconverged_solution_and_restores_handler() -> None:
    """YAPSS converts SIGINT to status 5 and restores process signal state."""
    run_check("yapss_sigint")
