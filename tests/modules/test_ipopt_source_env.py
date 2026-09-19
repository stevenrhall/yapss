"""

Test the notice for the removed `YAPSS_IPOPT_SOURCE` environment variable.

The variable selected the Ipopt backend until 0.3.0. Setting it now has no effect, so
YAPSS says so once per process, at the user's `solve()` line. Remove these tests with the
notice (0.4.0 or after 2027-09, whichever is later).

"""

import sys
import warnings

import pytest

import yapss
from yapss._backend import config
from yapss._legacy.examples.rosenbrock import setup

MESSAGE = "YAPSS_IPOPT_SOURCE environment variable has no effect"


@pytest.fixture(autouse=True)
def _reset_notice(monkeypatch):
    monkeypatch.setattr(config, "_ipopt_source_env_warned", False)


@pytest.fixture
def problem():
    ocp = setup()
    ocp.ipopt_options.tol = 1e-8
    ocp.ipopt_options.print_level = 0
    return ocp


def notices(records):
    return [r for r in records if MESSAGE in str(r.message)]


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_is_silent(problem, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    else:
        monkeypatch.setenv("YAPSS_IPOPT_SOURCE", value)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        problem.solve()
    assert notices(records) == []


@pytest.mark.parametrize("value", ["casadi", "cyipopt", "/some/libipopt.so"])
def test_set_warns_once_at_the_solve_line_and_has_no_effect(problem, monkeypatch, value):
    monkeypatch.setenv("YAPSS_IPOPT_SOURCE", value)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        solve_line = sys._getframe().f_lineno + 1
        solution = problem.solve()
        problem.solve()
    (notice,) = notices(records)
    assert notice.category is yapss.YapssDeprecationWarning
    assert (notice.filename, notice.lineno) == (__file__, solve_line)
    assert solution.nlp_info.ipopt_status == 0
    assert "cyipopt" not in sys.modules
