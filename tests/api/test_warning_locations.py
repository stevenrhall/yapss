"""Every YAPSS warning points at the user's own line, not inside YAPSS.

The warnings page promises it, and it is what makes the categories worth knowing: a filter
written as ``module="yapss"`` matches none of them. The warnings below take their `stacklevel`
from `exceptions.user_stacklevel`, and these tests pin where each one lands.

The convergence warning is pinned in `test_convergence_warning.py`, and the assignment-time
option warning in the contract suite.
"""

import os
import subprocess
import sys
import warnings

import yapss
from yapss._backend import config
from yapss.examples.brachistochrone import setup


def caught(category, action):
    """Run `action` and return the warnings of `category` it issued."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        action()
    return [w for w in records if issubclass(w.category, category)]


def test_an_option_ipopt_refuses_at_the_solve_points_at_the_caller():
    """A documented option this build lacks warns at the solve, and names the user's line.

    No build YAPSS supports includes WSMP, so Ipopt refuses its options while the value is
    within what Ipopt's documentation allows: a warning, not an error.
    """
    problem = setup()
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.wsmp_num_threads = 2
    found = caught(yapss.IpoptOptionSettingWarning, problem.solve)
    assert len(found) == 1
    assert found[0].filename == __file__


def test_the_ipopt_source_environment_notice_points_at_the_caller(monkeypatch):
    monkeypatch.setenv("YAPSS_IPOPT_SOURCE", "casadi")
    monkeypatch.setattr(config, "_ipopt_source_env_warned", False)
    problem = setup()
    problem.ipopt_options.print_level = 0
    found = caught(yapss.YapssDeprecationWarning, problem.solve)
    assert len(found) == 1
    assert "YAPSS_IPOPT_SOURCE" in str(found[0].message)
    assert found[0].filename == __file__


def test_an_invalid_logging_level_points_at_the_import():
    """Issued while `yapss` is imported, so it belongs to the user's ``import yapss``."""
    env = {**os.environ, "YAPSS_LOGGING": "not-a-level"}
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import yapss"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "<string>:1: YapssWarning: Invalid logging level" in result.stderr
