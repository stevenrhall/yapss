"""The Goddard rocket in one phase, whose derivative callbacks are part of the example.

The entry-by-entry check that those callbacks are right lives in `test_user_derivatives.py`,
which compares the assembled NLP Jacobian and Hessian against automatic differentiation at an
off-guess point. What is here is the end-to-end behaviour: the same answer as the released API,
the same answer from every derivative method, and the bang-singular-bang structure appearing on
its own rather than being imposed as it is in the three-phase version.
"""

import numpy as np
import pytest

from yapss.examples.goddard_problem_1_phase import Tm, m0, mf, setup

RELEASED = 18565.096736988547
"""The altitude the same problem reaches through the released API, to the last bit."""


@pytest.fixture
def problem():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem


def test_it_agrees_with_the_released_api(problem):
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-12)


@pytest.mark.parametrize(
    "method", ["user", "auto", "central-difference", "central-difference-full"]
)
def test_every_derivative_method_agrees(problem, method):
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(RELEASED, rel=1e-6)


def test_the_rocket_burns_to_the_dry_mass(problem):
    ps = problem.solve()[problem.phases.flight]
    assert ps.initial.m == pytest.approx(m0, abs=1e-8)
    assert ps.final.m == pytest.approx(mf, abs=1e-8)
    assert ps.final.h == pytest.approx(RELEASED, rel=1e-6)


def test_the_singular_arc_chatters(problem):
    """One phase cannot represent the singular arc, so the thrust bangs between its limits.

    This is the example's point rather than a defect in it: the released API does the same
    thing on the same mesh, because nothing in the transcription asks the control to be smooth.
    What survives is the average -- the interior thrust sits between the limits in the mean,
    while most individual points are at one limit or the other -- and the altitude, which is
    right to six figures. `goddard_problem_3_phase` states the arcs as phases and gets a clean
    interior thrust instead.
    """
    thrust = np.asarray(problem.solve()[problem.phases.flight].control.thrust)
    interior = thrust[len(thrust) // 4 : len(thrust) // 2]
    at_a_limit = (interior < 0.01 * Tm) | (interior > 0.99 * Tm)

    assert thrust[0] > 0.99 * Tm, "the rocket starts at full thrust"
    assert thrust[-1] < 0.01 * Tm, "and coasts at the end"
    # the mean is interior even though the points are not, which is what chattering means
    assert 0.01 * Tm < interior.mean() < 0.99 * Tm
    assert at_a_limit.mean() > 0.5
