"""

Test the deprecated `ipopt_source` override and its warnings.

`ipopt_source` selects the Ipopt backend and is removed in 0.3.0; these tests
pin the 0.1.x behavior so the deprecation cannot quietly become a removal. The
warning categories are asserted explicitly rather than incidentally: the custom
path case must be a `FutureWarning`, because `DeprecationWarning` is hidden
outside ``__main__`` and that path can end in a SIGSEGV. Policy:
IPOPT_BACKEND_POLICY.md §3.

Half of these assert that something does *not* warn. That half matters most: a
deprecation that fires for users who never opted in trains them to filter
warnings, and then the ones that matter are invisible too.

"""

import warnings
from contextlib import contextmanager

import pytest

import yapss
from yapss._private import config, solver

DEPRECATIONS = (DeprecationWarning, FutureWarning)


@contextmanager
def no_deprecation_warning():
    """Assert that no deprecation or future warning is emitted in this block.

    `simplefilter("always")` rather than `"error"`, so an unexpected warning is
    reported with its message rather than as a bare exception from wherever it
    happened to be raised.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    offenders = [w for w in caught if issubclass(w.category, DEPRECATIONS)]
    assert offenders == [], f"unexpected warnings: {[str(w.message) for w in offenders]}"


@pytest.fixture(autouse=True)
def _reset_env_warning():
    """Clear the once-per-process env-var warning flag between tests."""
    solver._env_deprecation_warned = False
    yield
    solver._env_deprecation_warned = False


@pytest.fixture
def problem():
    return yapss.Problem(name="test", nx=[1], nu=[1])


def make_problem(monkeypatch, source):
    """Build a problem with `ipopt_source` set, suppressing the set-time warning.

    The warning is the subject of other tests; here it is noise.
    """
    p = yapss.Problem(name="test", nx=[1], nu=[1])
    with pytest.warns((DeprecationWarning, FutureWarning)):
        p.ipopt_source = source
    return p


# --- warning categories and messages --------------------------------------------


@pytest.mark.parametrize("value", ["default", "cyipopt", "casadi"])
def test_setter_warns_deprecation(problem, value):
    with pytest.warns(DeprecationWarning, match="0.3.0"):
        problem.ipopt_source = value


def test_setter_warns_future_for_explicit_path(problem, tmp_path):
    """A custom path gets FutureWarning, which is shown even outside __main__."""
    library = tmp_path / "libipopt.so"
    library.write_bytes(b"")
    with pytest.warns(FutureWarning, match="cannot verify"):
        problem.ipopt_source = str(library)


def test_explicit_path_warning_is_not_a_deprecation_warning(problem, tmp_path):
    """Guards the distinction, not just the behavior.

    `FutureWarning` is not a subclass of `DeprecationWarning`, so asserting the
    category is what keeps someone from "simplifying" these to one call.
    """
    library = tmp_path / "libipopt.so"
    library.write_bytes(b"")
    with pytest.warns(Warning) as record:
        problem.ipopt_source = str(library)
    assert not any(issubclass(w.category, DeprecationWarning) for w in record)


@pytest.mark.parametrize(
    ("value", "in_conda", "expected"),
    [
        ("cyipopt", True, "can simply be deleted"),
        ("cyipopt", False, "OpenMP"),
    ],
)
def test_cyipopt_message_depends_on_environment(monkeypatch, value, in_conda, expected):
    """In conda the advice is "delete the line"; in pip it is "this may crash"."""
    monkeypatch.setattr(config, "get_conda_prefix", lambda: "/fake/prefix" if in_conda else None)
    _, message = config.ipopt_source_deprecation(value)
    assert expected in message


def test_messages_name_the_removal_version():
    for value in ("default", "cyipopt", "casadi", "/tmp/libipopt.so"):
        _, message = config.ipopt_source_deprecation(value)
        assert "0.3.0" in message


# --- silence for users who never opted in ---------------------------------------


def test_constructor_does_not_warn(recwarn):
    """`__init__` assigns `_ipopt_source` directly, bypassing the setter."""
    problem = yapss.Problem(name="test", nx=[1], nu=[1])
    assert problem.ipopt_source == "default"
    assert [w for w in recwarn if issubclass(w.category, (DeprecationWarning, FutureWarning))] == []


def test_reading_the_attribute_does_not_warn(problem, recwarn):
    _ = problem.ipopt_source
    assert [w for w in recwarn if issubclass(w.category, (DeprecationWarning, FutureWarning))] == []


def test_default_resolution_does_not_warn(problem, recwarn, monkeypatch):
    """No env var, untouched attribute: the common case must be silent."""
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    solver.configure_ipopt_source(problem)
    assert [w for w in recwarn if issubclass(w.category, (DeprecationWarning, FutureWarning))] == []


# --- resolution ------------------------------------------------------------------


def test_default_follows_environment(problem, monkeypatch):
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)

    # cyipopt is presumed installed when simulating conda: the import-time check
    # in solver.py already refuses to load there without it, so a conda process
    # that reaches this function always has it.
    monkeypatch.setattr(solver, "_IN_CONDA", True)
    monkeypatch.setattr(solver.importlib.util, "find_spec", lambda name: object())
    assert solver.configure_ipopt_source(problem) == "cyipopt"

    monkeypatch.setattr(solver, "_IN_CONDA", False)
    assert solver.configure_ipopt_source(problem) == "casadi"


def test_explicit_value_overrides_environment(monkeypatch):
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    monkeypatch.setattr(solver, "_IN_CONDA", False)
    problem = make_problem(monkeypatch, "cyipopt")
    monkeypatch.setattr(solver.importlib.util, "find_spec", lambda name: object())
    assert solver.configure_ipopt_source(problem) == "cyipopt"


def test_env_var_is_honored_and_warns_once(problem, monkeypatch):
    monkeypatch.setenv("YAPSS_IPOPT_SOURCE", "casadi")

    with pytest.warns(DeprecationWarning):
        assert solver.configure_ipopt_source(problem) == "casadi"

    # Second call: the value still applies, but the warning does not repeat --
    # it would otherwise fire on every solve.
    with no_deprecation_warning():
        assert solver.configure_ipopt_source(problem) == "casadi"


def test_attribute_takes_precedence_over_env_var(monkeypatch):
    monkeypatch.setenv("YAPSS_IPOPT_SOURCE", "cyipopt")
    monkeypatch.setattr(solver, "_IN_CONDA", False)
    problem = make_problem(monkeypatch, "casadi")
    # The env var is only consulted when the attribute is still "default".
    assert solver.configure_ipopt_source(problem) == "casadi"


# --- errors ----------------------------------------------------------------------


def test_missing_cyipopt_raises_with_actionable_message(monkeypatch):
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    problem = make_problem(monkeypatch, "cyipopt")
    monkeypatch.setattr(solver.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        solver.configure_ipopt_source(problem)

    message = str(excinfo.value)
    assert "conda install" in message
    assert "0.3.0" in message


def test_cyipopt_availability_is_probed_without_importing(monkeypatch):
    """Importing cyipopt would map a second Ipopt; `find_spec` must be used.

    Regression guard for the whole reason cyipopt is imported lazily: probing by
    import would trip the duplicate-copy guard for anyone who merely has cyipopt
    installed, a configuration that is entirely fine.
    """
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    problem = make_problem(monkeypatch, "cyipopt")

    calls = []
    monkeypatch.setattr(
        solver.importlib.util,
        "find_spec",
        lambda name: calls.append(name) or object(),
    )
    solver.configure_ipopt_source(problem)
    assert calls == ["cyipopt"]


def test_nonexistent_explicit_path_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("YAPSS_IPOPT_SOURCE", raising=False)
    missing = str(tmp_path / "no_such_library.so")
    problem = make_problem(monkeypatch, missing)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        solver.configure_ipopt_source(problem)


def test_non_string_value_still_rejected(problem):
    """Type validation predates the deprecation and must survive it."""
    with pytest.raises(TypeError, match="must have type 'str'"):
        problem.ipopt_source = 3
