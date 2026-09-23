"""Dynamic soaring, the only problem whose dynamics read a parameter.

`beta` is the wind gradient, and it is both an unknown of the problem and its objective: the
answer is the weakest shear in which a closed circuit exists. So the parameter appears in the
continuous callback as `arg.parameter.beta`, which no other example does, and in the objective
callback as the whole of it.

It is also the largest problem in the corpus -- 50 mesh segments, because the lift coefficient
meets its load-factor limit and the solution's derivatives are discontinuous where it does --
so the derivative-method sweep here is deliberately only `auto` against the released answer.
"""

import numpy as np
import pytest

from yapss.examples.dynamic_soaring import cl_max, load_factor_max, setup

RELEASED = 0.06358655820709537
"""The wind gradient the same problem finds through the released API, to the last bit."""


@pytest.fixture(scope="module")
def solved():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem, problem.solve()


def test_it_agrees_with_the_released_api(solved):
    _, solution = solved
    assert solution.objective == pytest.approx(RELEASED, rel=1e-10)


def test_the_objective_is_the_parameter(solved):
    """Nothing is integrated or evaluated at an endpoint: the objective *is* the unknown."""
    _, solution = solved
    assert solution.parameter.beta == pytest.approx(solution.objective, rel=1e-14)


def test_the_circuit_closes(solved):
    """Speed and flight path angle return to where they began, and the heading after one turn."""
    problem, solution = solved
    ps = solution[problem.phases.loop]
    assert solution.discrete.v_periodic == pytest.approx(0.0, abs=1e-6)
    assert solution.discrete.gamma_periodic == pytest.approx(0.0, abs=1e-8)
    assert solution.discrete.psi_periodic == pytest.approx(np.radians(360), abs=1e-8)
    for name in ("x", "y", "h"):
        assert getattr(ps.initial, name) == pytest.approx(0.0, abs=1e-6)
        assert getattr(ps.final, name) == pytest.approx(0.0, abs=1e-6)


def test_the_vehicle_flies_within_its_limits(solved):
    """Within the solver's tolerance, which is the only sense in which a bound holds.

    The load-factor limit is active over most of the circuit and Ipopt satisfies it to its own
    tolerance, so it is exceeded by a few parts in 1e8. Asserting an exact bound here would be
    asserting something about Ipopt's termination rather than about the trajectory.
    """
    problem, solution = solved
    ps = solution[problem.phases.loop]
    tolerance = 1e-6
    assert np.asarray(ps.control.cl).max() <= cl_max * (1 + tolerance)
    assert np.asarray(ps.path.load_factor).max() <= load_factor_max * (1 + tolerance)
    assert np.asarray(ps.state.h).min() >= -tolerance


def test_the_load_factor_limit_is_reached(solved):
    """The constraint is active, which is why the mesh has to be as dense as it is."""
    problem, solution = solved
    ps = solution[problem.phases.loop]
    assert np.asarray(ps.path.load_factor).max() == pytest.approx(load_factor_max, rel=1e-6)


def test_the_parameter_reaches_the_continuous_callback(solved):
    """The value the dynamics see is the one being solved for, not the guess."""
    problem, _ = solved
    seen = {}
    original = problem.phases.loop._continuous

    def spy(arg, out):
        seen["beta"] = float(np.asarray(arg.parameter.beta).ravel()[0])
        return original(arg, out)

    spied = setup()
    spied.ipopt_options.print_level = 0
    spied.phases.loop.register.continuous(spy)
    spied.derivatives.method = "central-difference"
    solution = spied.solve()
    assert seen["beta"] == pytest.approx(solution.parameter.beta, rel=1e-6)
