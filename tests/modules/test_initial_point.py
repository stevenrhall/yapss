"""

Test the refusal of a starting point at which the NLP is not finite.

A NaN or Inf Jacobian at the starting point used to reach Ipopt's least-squares multiplier
initialization, which factors a system built from the Jacobian before Ipopt checks the
constraint values; for some sparsity patterns MUMPS then crashed the process (SIGBUS, no
traceback). Two fixes guard it: `check_initial_point` raises before Ipopt is created, and
`check_derivatives_for_naninf = "yes"` is a YAPSS default so Ipopt stops with status -13 at
any later iterate. Tests that call `solve()` on a non-finite starting point go through the
first guard, and the second would still stop Ipopt cleanly, so a regression fails a test
instead of killing the test process -- except the one test that disables the first guard,
which runs in a subprocess.

"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from _ipopt_backend import CYIPOPT_ACTIVE

from yapss import Problem
from yapss._private import solver
from yapss._private.initial_point import _constraint_labels, _variable_labels
from yapss._private.ipopt_options import DEFAULT_IPOPT_OPTIONS
from yapss.examples import brachistochrone_minimal
from yapss.math import cos, sin, sqrt

METHODS = ["auto", "central-difference", "central-difference-full"]

# the non-finite values are the point of these tests
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")


def _brachistochrone(method: str, extra: str) -> Problem:
    """Return the minimal brachistochrone with one term added to its dynamics.

    The guess has state ``x`` rising from 0 and speed ``v`` rising from 0, so ``sqrt(v - 1)``
    is NaN at the early collocation points and ``sqrt(x)`` is finite but has an infinite
    derivative at the first one.
    """
    problem = brachistochrone_minimal.setup()
    problem.derivatives.method = method
    problem.derivatives.order = "second"
    problem.ipopt_options.print_level = 0
    g0 = 32.174

    def continuous(arg) -> None:
        x, _, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        vdot = g0 * sin(u)
        xdot = v * cos(u)
        if extra == "nan-value":
            vdot = vdot + sqrt(v - 1.0)
        elif extra == "infinite-derivative":
            xdot = xdot + sqrt(x)
        arg.phase[0].dynamics[:] = xdot, v * sin(u), vdot

    problem.functions.continuous = continuous
    return problem


@pytest.mark.parametrize("method", METHODS)
def test_nan_value_at_initial_guess_raises(method):
    problem = _brachistochrone(method, "nan-value")
    with pytest.raises(ValueError, match="not finite at the initial guess") as info:
        problem.solve()
    message = str(info.value)
    assert re.search(r"phase 0 dynamics\[2\] is NaN in \d+ of \d+ entries", message), message
    # derivatives of a value that is already non-finite are not reported
    assert "derivative of phase 0 dynamics[2]" not in message


@pytest.mark.parametrize("method", METHODS)
def test_infinite_derivative_with_finite_value_is_reported(method):
    problem = _brachistochrone(method, "infinite-derivative")
    with pytest.raises(ValueError, match="not finite at the initial guess") as info:
        problem.solve()
    message = str(info.value)
    assert "phase 0 dynamics[0] is" not in message
    assert "the derivative of phase 0 dynamics[0] with respect to phase 0 state[0]" in message


def test_nan_objective_is_reported():
    problem = brachistochrone_minimal.setup()
    problem.derivatives.method = "central-difference"
    problem.ipopt_options.print_level = 0

    def objective(arg) -> None:
        arg.objective = arg.phase[0].final_time + sqrt(-1.0 - arg.phase[0].final_time)

    problem.functions.objective = objective
    with pytest.raises(ValueError, match="the objective is NaN"):
        problem.solve()


def test_finite_initial_guess_does_not_raise():
    problem = brachistochrone_minimal.setup()
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    assert solution.nlp_info.ipopt_status == 0


@pytest.mark.parametrize("spectral_method", ["lgr", "lg", "lgl"])
def test_every_nlp_entry_has_a_label(spectral_method):
    """The labels cover every decision variable and constraint, for every method."""
    problem = Problem(name="labels", nx=[2, 1], nu=[1, 2], nq=[1, 0], nh=[1, 1], ns=2, nd=3)
    problem.spectral_method = spectral_method
    problem.mesh.phase[0].collocation_points = (3, 4)
    problem.mesh.phase[0].fraction = (0.5, 0.5)
    variables = _variable_labels(problem)
    constraints = _constraint_labels(problem)
    assert all(variables), [i for i, label in enumerate(variables) if not label]
    assert all(constraints), [i for i, label in enumerate(constraints) if not label]
    assert "parameter[1]" in variables
    assert "phase 1 final time" in variables
    assert "discrete[2]" in constraints
    assert "phase 1 duration" in constraints
    if spectral_method == "lg":
        assert "phase 0 dynamics[1] (end-of-segment quadrature)" in constraints


def test_nan_derivative_check_is_a_yapss_default():
    assert DEFAULT_IPOPT_OPTIONS["check_derivatives_for_naninf"] == "yes"
    assert brachistochrone_minimal.setup().ipopt_options.check_derivatives_for_naninf == "yes"


@pytest.mark.skipif(CYIPOPT_ACTIVE, reason="the crash was observed with the vendored Ipopt")
@pytest.mark.isolation
def test_ipopt_stops_cleanly_when_the_initial_point_check_is_bypassed():
    """With YAPSS's own check disabled, Ipopt's derivative check still stops the solve.

    Runs in a subprocess: this is the configuration that crashed with SIGBUS before the
    `check_derivatives_for_naninf` default, and a regression must not kill pytest. The
    assertion is the status Ipopt returns, -13 (Invalid_Number_Detected), not merely that
    the process survived.
    """
    script = textwrap.dedent(f"""
        import sys
        import warnings
        warnings.simplefilter("ignore")
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from yapss._private import solver
        from test_initial_point import _brachistochrone
        solver.check_initial_point = lambda *args: None
        problem = _brachistochrone("central-difference", "nan-value")
        solution = problem.solve()
        print("STATUS", solution.nlp_info.ipopt_status)
        """)
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, f"exit {process.returncode}:\n{output}"
    assert "STATUS -13" in process.stdout, output
