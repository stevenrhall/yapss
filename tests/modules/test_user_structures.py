"""

Tests for validation of user-supplied derivative structures.

The "user" derivative method takes the structures verbatim from the keys of the
dictionaries the user's derivative callbacks fill. A Hessian entry represents one
second partial derivative, so the documented convention is that each unordered
variable pair is supplied exactly once, in either order.

Supplying both orders assembles as two mirrored coordinates that the structure fold
sums. That is correct for a user who split one derivative across the two keys and
wrong (doubled) for a user who supplied both triangles of a symmetric Hessian, and
nothing downstream can tell which was meant. The ambiguity is therefore refused:
``MirroredHessianPairWarning`` (a ``FutureWarning``) in 0.2.x with the summing behavior
unchanged so that currently-correct formulations are not broken by a patch release, and
``ValueError`` from 0.3.0. These tests pin the warning, its diagnosis, and the unchanged
behavior; the 0.3.0 change flips ``test_mirrored_pair_behavior_is_unchanged`` into a
raise test.

"""

import warnings

import numpy as np
import pytest

from yapss import MirroredHessianPairWarning, Problem
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss._private.user import make_user_functions


def build_problem(
    *,
    mirror_continuous=False,
    mirror_objective=False,
    mirror_discrete=False,
    mirror_value=1.0,
):
    """A small problem with user derivatives; flags add the mirror keys."""
    problem = Problem(name="mirror", nx=[1], nu=[1], nd=1)

    def objective(arg):
        arg.objective = (
            arg.phase[0].final_time ** 2 + arg.phase[0].final_time * arg.phase[0].initial_time
        )

    def objective_gradient(arg):
        phase = arg.phase[0]
        arg.gradient[0, "tf", 0] = 2.0 * phase.final_time + phase.initial_time
        arg.gradient[0, "t0", 0] = phase.final_time

    def objective_hessian(arg):
        arg.hessian[(0, "tf", 0), (0, "tf", 0)] = 2.0  # diagonal: always legal
        arg.hessian[(0, "t0", 0), (0, "tf", 0)] = 1.0
        if mirror_objective:
            arg.hessian[(0, "tf", 0), (0, "t0", 0)] = mirror_value

    def continuous(arg):
        (x,) = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = [x * u]

    def continuous_jacobian(arg):
        (x,) = arg.phase[0].state
        (u,) = arg.phase[0].control
        jacobian = arg.phase[0].jacobian
        jacobian[("f", 0), ("x", 0)] = u
        jacobian[("f", 0), ("u", 0)] = x

    def continuous_hessian(arg):
        hessian = arg.phase[0].hessian
        hessian[("f", 0), ("x", 0), ("u", 0)] = 1.0
        if mirror_continuous:
            hessian[("f", 0), ("u", 0), ("x", 0)] = mirror_value

    def discrete(arg):
        arg.discrete[:] = [arg.phase[0].final_state[0] * arg.phase[0].initial_state[0]]

    def discrete_jacobian(arg):
        arg.jacobian[0, (0, "xf", 0)] = arg.phase[0].initial_state[0]
        arg.jacobian[0, (0, "x0", 0)] = arg.phase[0].final_state[0]

    def discrete_hessian(arg):
        arg.hessian[0, (0, "x0", 0), (0, "xf", 0)] = 1.0
        if mirror_discrete:
            arg.hessian[0, (0, "xf", 0), (0, "x0", 0)] = mirror_value

    functions = problem.functions
    functions.objective = objective
    functions.objective_gradient = objective_gradient
    functions.objective_hessian = objective_hessian
    functions.continuous = continuous
    functions.continuous_jacobian = continuous_jacobian
    functions.continuous_hessian = continuous_hessian
    functions.discrete = discrete
    functions.discrete_jacobian = discrete_jacobian
    functions.discrete_hessian = discrete_hessian

    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower, bounds.initial_time.upper = 0.0, 0.5
    bounds.final_time.lower, bounds.final_time.upper = 1.0, 2.0
    bounds.state.lower[:], bounds.state.upper[:] = -5.0, 5.0
    bounds.control.lower[:], bounds.control.upper[:] = -5.0, 5.0
    problem.bounds.discrete.lower[:], problem.bounds.discrete.upper[:] = -50.0, 50.0

    guess = problem.guess.phase[0]
    guess.time = [0.0, 1.0]
    guess.state = [[1.0, 2.0]]
    guess.control = [[1.0, 1.0]]

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    return problem


def build_nlp(problem):
    """Run structure collection the way the solver does, and build the NLP."""
    problem.validate()
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)
    functions = make_user_functions(problem, z0, mesh.tau_u)
    return NLP(problem, functions, mesh), z0


def test_well_formed_structures_do_not_warn():
    """One key per pair -- including a diagonal key -- collects without complaint."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_nlp(build_problem())


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("mirror_continuous", "continuous Hessian of phase 0"),
        ("mirror_objective", "objective Hessian"),
        ("mirror_discrete", "discrete Hessian"),
    ],
)
def test_mirrored_pair_warns(flag, expected):
    """Setting both orders of one pair warns, naming the surface and the 0.3.0 change."""
    with pytest.warns(MirroredHessianPairWarning, match=expected) as record:
        build_nlp(build_problem(**{flag: True}))
    message = str(record[0].message)
    assert "same second derivative" in message
    assert "0.3.0" in message


def test_warning_is_a_future_warning():
    """FutureWarning, so it is shown by default outside __main__ (unlike Deprecation)."""
    assert issubclass(MirroredHessianPairWarning, FutureWarning)


def test_diagnosis_distinguishes_equal_from_different_values():
    """Equal values look like a full symmetric Hessian; different values may be a split."""
    with pytest.warns(MirroredHessianPairWarning, match="doubling the term") as record:
        build_nlp(build_problem(mirror_continuous=True, mirror_value=1.0))
    assert "Remove one" in str(record[0].message)

    with pytest.warns(MirroredHessianPairWarning, match="combine them") as record:
        build_nlp(build_problem(mirror_continuous=True, mirror_value=3.0))
    assert "doubling" not in str(record[0].message)


def test_mirrored_pair_behavior_is_unchanged():
    """In 0.2.x the two orders are still summed; the warning does not alter the Hessian.

    This pins the compatibility promise of a patch release: a user who split one
    derivative across the two orders gets the same answer as before. When 0.3.0 makes
    this a ValueError, replace the body with a ``pytest.raises`` check.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MirroredHessianPairWarning)
        nlp_once, z0 = build_nlp(build_problem())
        nlp_twice, _ = build_nlp(build_problem(mirror_continuous=True, mirror_value=1.0))
    lam = np.ones(len(nlp_once.constraints(z0)))

    def dense(nlp):
        row, col = nlp.hessianstructure()
        out = np.zeros((len(z0), len(z0)))
        for r, c, v in zip(row, col, nlp.hessian(z0, lam, np.float64(1.0)), strict=True):
            out[r, c] += v
        return out

    once, twice = dense(nlp_once), dense(nlp_twice)
    # the mirrored pair contributes at the (x, u) coordinates; everything else is equal
    assert not np.allclose(twice, once)
    mask = ~np.isclose(twice, once)
    np.testing.assert_allclose(twice[mask], 2.0 * once[mask])
