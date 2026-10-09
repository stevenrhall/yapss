"""YAPSS does not read an Ipopt options file.

Left at its default, Ipopt reads ``ipopt.opt`` from the working directory when a solve starts,
and the file's settings take precedence over options set through the C interface, the ones
YAPSS reserves included. A stray file could therefore change a solve with nothing in the
script to show it, or make the library's smoke test fail with a misleading ABI error. YAPSS
sets ``option_file_name`` to the empty string, which opens no file, and reserves the option.

The file used here would be noticed if it were read: ``max_iter 0`` stops Ipopt before its
first iteration, so the solve could not converge.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

import yapss
from yapss.examples import hs071

OPTION_FILE = "max_iter 0\n"


def test_reserved() -> None:
    problem = hs071.setup()
    with pytest.raises(ValueError, match="managed by YAPSS"):
        problem.ipopt_options.option_file_name = "my_options.opt"


def test_solve_ignores_ipopt_opt(tmp_path, monkeypatch) -> None:
    (tmp_path / "ipopt.opt").write_text(OPTION_FILE)
    monkeypatch.chdir(tmp_path)
    problem = hs071.setup()
    problem.ipopt_options.print_level = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptConvergenceWarning)
        solution = problem.solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED


@pytest.mark.isolation
def test_smoke_test_ignores_ipopt_opt(tmp_path) -> None:
    """The library check runs once per process, at the first solve, so it needs a fresh one."""
    (tmp_path / "ipopt.opt").write_text(OPTION_FILE)
    code = (
        "from yapss.examples import hs071\n"
        "problem = hs071.setup()\n"
        "problem.ipopt_options.print_level = 0\n"
        "print(problem.solve().status.name)\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, output
    assert "SOLVE_SUCCEEDED" in process.stdout, output
