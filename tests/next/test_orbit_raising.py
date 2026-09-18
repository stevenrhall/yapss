"""End-to-end solves of the orbit raising problem through the redesigned API.

This is the example whose continuous callback reads the phase's independent variable: the
vehicle's mass falls as it burns, so its thrust acceleration depends explicitly on the time.
"""

import numpy as np
import pytest

from yapss._next.examples.orbit_raising import setup

RELEASED = 1.525277594542231
"""The final radius the same problem reaches through the released API."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_agrees_with_the_released_api(problem):
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-8)


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-6)


def test_the_final_orbit_is_circular(problem):
    solution = problem.solve()
    ps = solution[problem.phases.raise_]
    assert solution.discrete.circular == pytest.approx(0.0, abs=1e-8)
    assert ps.final.v_theta == pytest.approx(np.sqrt(1.0 / ps.final.r), rel=1e-6)


def test_the_callback_is_given_the_time_at_every_point(problem):
    seen = {}

    def spy(arg, out):
        seen["shape"] = np.shape(arg.time)
        points = np.asarray(arg.time).ravel()
        seen["span"] = points[-1] - points[0]
        r, v_r, v_theta = arg.state.r, arg.state.v_r, arg.state.v_theta
        u_r, u_theta = arg.control.u_r, arg.control.u_theta
        a = 0.1405 / (1.0 - 0.0749 * arg.time)
        out.dynamics.r = v_r
        out.dynamics.theta = v_theta / r
        out.dynamics.v_r = v_theta**2 / r - 1.0 / r**2 + a * u_r
        out.dynamics.v_theta = -(v_r * v_theta) / r + a * u_theta
        out.path.unit_thrust = u_r**2 + u_theta**2
        return out

    problem.phases.raise_.register.continuous(spy, replace=True)
    problem.derivatives.method = "central-difference"
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-6)
    # every point of the phase, not one at a time, and spanning it
    assert seen["shape"][0] > 1
    # a finite-difference stencil perturbs the endpoints, so the span is not exact
    assert seen["span"] == pytest.approx(3.32, rel=1e-3)


def test_the_time_is_read_only_in_the_callback(problem):
    seen = {}

    original = problem.phases.raise_._continuous

    def spy(arg, out):
        seen["writeable"] = np.asarray(arg.time).flags.writeable
        return original(arg, out)

    problem.phases.raise_.register.continuous(spy, replace=True)
    problem.derivatives.method = "central-difference"
    problem.solve()
    assert seen["writeable"] is False
