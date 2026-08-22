"""

Check that the derivative backends transcribe the *same problem*.

The rest of ``tests/derivatives/`` checks that the backends compute the same
*derivatives* of a given function. This module checks something logically prior: that
they see the same function at all. Nothing else does.

That gap mattered. Because numpy's object-dtype loop coerces elementwise comparison
results to ``bool``, a gated expression such as ``(x <= 0.5) * v`` evaluated correctly
under the finite-difference methods and silently lost its mask under ``"auto"`` -- the
constraint was enforced over the whole domain rather than a finite interval. Every
existing cross-backend test used smooth callbacks, so none of them noticed.

Note what is compared and what is not. These tests compare **function values** at a
fixed point, not solutions. For a non-smooth callback the two backends legitimately
disagree on *derivatives* near a kink -- casadi differentiates a comparison node to zero
almost everywhere, while central differencing straddles the discontinuity and sees a
large finite difference -- so the two can converge to different points on the same
problem. That is a property of the formulation, not a defect, and asserting on solved
objectives would be testing the wrong thing.

"""

import numpy as np
import pytest

from yapss import Problem
from yapss._private.auto import make_auto_functions
from yapss._private.central_difference import make_cd_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss.math import abs as yabs
from yapss.math import cos, maximum, minimum, pi, sign, sin

G0 = 32.174


def gated_path(arg):
    """Enforce the path constraint only where the first state is below 0.5."""
    x, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [(x <= 0.5) * v]


def two_sided_gate(arg):
    """A mask built from two comparisons combined with `&`."""
    x, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [((x >= 0.2) & (x <= 0.8)) * v]


def negated_gate(arg):
    """A mask built by negating a comparison with `~`."""
    x, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [(~(x > 0.5)) * v]


def nonsmooth_mix(arg):
    """Other non-smooth constructs YAPSS supports: abs, sign, min, max."""
    x, y, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [yabs(y) + sign(x) * minimum(v, 3.0) + maximum(x, 0.25)]


def time_gate(arg):
    """A mask on *time*: exercises the symbolic time array's comparison operators."""
    _, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    t = arg.phase[0].time
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [(t <= 0.3) * v]


CALLBACKS = {
    "time_gate": time_gate,
    "gated_path": gated_path,
    "two_sided_gate": two_sided_gate,
    "negated_gate": negated_gate,
    "nonsmooth_mix": nonsmooth_mix,
}


def build_problem(continuous, method):
    """Build a one-phase brachistochrone variant with the given path constraint."""
    problem = Problem(name="agreement", nx=[3], nu=[1], nh=[1])

    def objective(arg):
        arg.objective = arg.phase[0].final_time

    problem.functions.objective = objective
    problem.functions.continuous = continuous

    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.initial_state.lower[:] = bounds.initial_state.upper[:] = 0.0
    bounds.final_state.lower[0] = bounds.final_state.upper[0] = 1.0
    bounds.state.lower[:] = 0.0
    bounds.state.upper[:] = 10.0
    bounds.control.lower[:] = -pi / 2
    bounds.control.upper[:] = pi / 2
    bounds.path.lower[:] = -10.0
    bounds.path.upper[:] = 2.0

    phase = problem.guess.phase[0]
    phase.time = [0.0, 1.0]
    phase.state = [[0.0, 1.0], [0.0, 1.0], [0.0, 5.0]]
    phase.control = [[0.0, 0.0]]

    problem.derivatives.method = method
    return problem


def nlp_values(continuous, method, seed=0):
    """Return (objective, constraints) at a fixed perturbation of the initial guess."""
    problem = build_problem(continuous, method)
    problem.validate()
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)

    if method == "auto":
        functions = make_auto_functions(problem)
    else:
        functions = make_cd_functions(problem, z0, mesh.tau_u)
    nlp = NLP(problem, functions, mesh)

    # perturb off the guess so the gate is active at some collocation points and not
    # others; a point where the mask is 1 everywhere would not discriminate
    rng = np.random.default_rng(seed)
    z = z0 + 0.3 * rng.standard_normal(len(z0))
    return nlp.objective(z), np.asarray(nlp.constraints(z))


@pytest.mark.parametrize("name", sorted(CALLBACKS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auto_and_central_difference_agree_on_values(name, seed):
    """The two backends must transcribe identical objective and constraint values."""
    continuous = CALLBACKS[name]
    objective_auto, constraints_auto = nlp_values(continuous, "auto", seed)
    objective_cd, constraints_cd = nlp_values(continuous, "central-difference", seed)

    assert objective_auto == pytest.approx(objective_cd, rel=1e-12, abs=1e-12)
    assert constraints_auto.shape == constraints_cd.shape
    assert np.allclose(constraints_auto, constraints_cd, rtol=1e-12, atol=1e-12), (
        f"{name}: backends disagree on constraint values; "
        f"max difference {np.max(np.abs(constraints_auto - constraints_cd)):.3e}"
    )


def ungated_path(arg):
    """Same as `gated_path`, with the mask removed -- used only by the guard below."""
    _, _, v = arg.phase[0].state
    (u,) = arg.phase[0].control
    arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), G0 * sin(u)
    arg.phase[0].path[:] = [v]


def test_gate_actually_binds():
    """Guard the test above: a gate that is never active would prove nothing.

    If the mask were 1 at every collocation point, dropping it entirely would still
    produce matching values and the regression test would silently stop testing.
    """
    _, gated = nlp_values(gated_path, "central-difference", seed=0)
    _, ungated = nlp_values(ungated_path, "central-difference", seed=0)
    assert not np.allclose(gated, ungated), "gate is inactive; the test above is vacuous"
