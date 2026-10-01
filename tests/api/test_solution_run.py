"""``solution.run``: what was true of one solve beyond its problem.

Versions, the Ipopt build, the options Ipopt accepted, the platform, when and how long, and the
warnings shown. Nothing identifies the machine or its user, since a solution is shared. The
warnings are recorded without being swallowed or moved: each still reaches the user, pointing
at their own line.
"""

import getpass
import pickle
import platform
import socket
import sys
import warnings
from datetime import UTC, datetime

import numpy as np
import pytest

import yapss
from yapss.examples.brachistochrone_minimal import setup


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_the_versions_are_recorded(problem):
    run = problem.solve().run
    assert run.yapss_version == yapss.__version__
    assert run.python_version == platform.python_version()
    assert run.numpy_version == np.__version__
    assert run.casadi_version
    assert run.ipopt_version is None or run.ipopt_version.count(".") == 2


def test_the_ipopt_build_names_the_library_file_but_not_its_directory(problem):
    build = problem.solve().run.ipopt_build
    assert build.split(" ")[0] in {"wheel", "conda-forge"}
    assert "ipopt" in build.lower()
    assert "/" not in build and "\\" not in build


def test_the_options_ipopt_received_include_yapss_s_own(problem):
    problem.ipopt_options.max_iter = 500
    options = problem.solve().run.ipopt_options
    assert options["max_iter"] == 500
    assert options["nlp_scaling_method"] == "user-scaling"


def test_an_option_ipopt_refused_is_not_among_those_it_received(problem):
    problem.ipopt_options.not_a_real_option = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = problem.solve().run
    assert "not_a_real_option" not in run.ipopt_options


def test_the_timing_is_measured_and_its_parts_fit_in_the_total(problem):
    before = datetime.now(UTC)
    seconds = problem.solve().run.seconds
    assert 0.0 < seconds.ipopt <= seconds.total
    assert 0.0 <= seconds.setup <= seconds.total
    assert 0.0 <= seconds.solution <= seconds.total
    assert seconds.setup + seconds.ipopt + seconds.solution == pytest.approx(seconds.total)
    assert problem.solve().run.started >= before


def _user_name() -> str:
    """Return the user's name, or nothing where the environment does not say."""
    try:
        return getpass.getuser()
    except OSError:  # Windows under tox: no LOGNAME, USER, LNAME or USERNAME, and no pwd module
        return ""


def test_nothing_identifies_the_machine_or_its_user(problem):
    """A solution is shared, so its pickle carries no hostname, user name, or full path."""
    solution = problem.solve()
    data = pickle.dumps(solution)
    for private in (socket.gethostname(), _user_name()):
        if len(private) > 3:  # a very short name would match by accident
            assert private.encode() not in data, private
    assert sys.prefix.encode() not in data


def test_a_warning_is_recorded_and_still_shown_at_the_user_s_line(problem):
    """The convergence warning of a solve stopped early, and an option Ipopt refused."""
    problem.ipopt_options.max_iter = 2
    problem.ipopt_options.not_a_real_option = 1
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        line = sys._getframe().f_lineno + 1
        solution = problem.solve()
    ours = [w for w in shown if issubclass(w.category, yapss.YapssWarning)]
    assert {w.category for w in ours} == {
        yapss.IpoptConvergenceWarning,
        yapss.IpoptOptionSettingWarning,
    }
    assert all((w.filename, w.lineno) == (__file__, line) for w in ours)
    recorded = solution.run.warnings
    assert [name for name, _ in recorded] == [w.category.__name__ for w in shown]
    assert [message for _, message in recorded] == [str(w.message) for w in shown]


def test_the_display_is_put_back_after_the_solve(problem):
    display = warnings.showwarning
    problem.solve()
    assert warnings.showwarning is display


def test_a_warning_turned_into_an_error_still_raises(problem):
    problem.ipopt_options.max_iter = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptConvergenceWarning)
        with pytest.raises(yapss.IpoptConvergenceWarning):
            problem.solve()


def test_each_solve_in_a_loop_records_its_own_warning(problem):
    """Under the default filter a warning from one line is shown once, but every solve issued
    it, so every solution records it: a record does not depend on earlier solves."""
    problem.ipopt_options.max_iter = 2
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("default")
        solutions = [problem.solve() for _ in range(3)]
    for solution in solutions:
        names = [name for name, _ in solution.run.warnings]
        assert names == ["IpoptConvergenceWarning"]


def test_a_warning_the_user_ignores_is_still_recorded(problem):
    problem.ipopt_options.max_iter = 2
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("ignore", yapss.IpoptConvergenceWarning)
        solution = problem.solve()
    assert not [w for w in shown if issubclass(w.category, yapss.IpoptConvergenceWarning)]
    assert [name for name, _ in solution.run.warnings] == ["IpoptConvergenceWarning"]


def test_a_clean_solve_records_no_warnings(problem):
    assert problem.solve().run.warnings == ()


def test_the_run_pickles_with_the_solution(problem):
    solution = problem.solve()
    copy = pickle.loads(pickle.dumps(solution))
    assert copy.run.started == solution.run.started
    assert copy.run.seconds.total == solution.run.seconds.total


def test_a_warning_at_a_line_outside_any_module_is_shown(problem):
    """A solve called from code with no module, as a doctest's or an exec'd line, still shows
    its warnings: they are issued again at that line, with no module to name."""
    problem.ipopt_options.max_iter = 2
    namespace = {"problem": problem}
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        exec(compile("problem.solve()", "<doctest example>", "exec"), namespace)  # noqa: S102
    assert [w.filename for w in shown if issubclass(w.category, yapss.YapssWarning)] == [
        "<doctest example>"
    ]


def test_a_setup_warning_filtered_into_an_error_stops_the_solve_before_ipopt(problem, monkeypatch):
    """An option Ipopt refuses, under an "error" filter, raises before Ipopt is run."""
    from yapss._backend import solver

    ran = []
    original = solver._solve_ipopt_problem

    def spy(*args, **kwargs):
        ran.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_solve_ipopt_problem", spy)
    problem.ipopt_options.not_a_real_option = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptOptionSettingWarning)
        with pytest.raises(yapss.IpoptOptionSettingWarning):
            problem.solve()
    assert ran == []


def test_a_setup_warning_is_recorded_when_the_solve_goes_ahead(problem):
    problem.ipopt_options.not_a_real_option = 1
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solution = problem.solve()
    assert [name for name, _ in solution.run.warnings] == ["IpoptOptionSettingWarning"]
