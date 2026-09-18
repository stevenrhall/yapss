"""

Tests for validation of user-supplied derivative structures.

The "user" derivative method takes the structures verbatim from the keys of the
dictionaries the user's derivative callbacks fill. A Hessian entry represents one
second partial derivative, so the documented convention is that each unordered
variable pair is supplied exactly once, in either order.

Supplying both orders would assemble as two mirrored coordinates that the structure fold
sums. That is correct for a user who split one derivative across the two keys and
wrong (doubled) for a user who supplied both triangles of a symmetric Hessian, and
nothing downstream can tell which was meant. The ambiguity is therefore refused: it
warned with ``MirroredHessianPairWarning`` through 0.2.x, and raises ``ValueError`` from
0.3.0. These tests pin the error and its diagnosis.

"""

import warnings

import numpy as np
import pytest

import yapss
from yapss import Problem
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
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem._to_spec(), mesh)
    functions = make_user_functions(problem._to_spec(), z0, mesh.tau_u)
    return NLP(problem._to_spec(), functions, mesh), z0


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
def test_mirrored_pair_raises(flag, expected):
    """Setting both orders of one pair raises, naming the Hessian and both keys."""
    with pytest.raises(ValueError, match=expected) as info:
        build_nlp(build_problem(**{flag: True}))
    assert "same second derivative" in str(info.value)


def test_diagnosis_distinguishes_equal_from_different_values():
    """Equal values look like a full symmetric Hessian; different values may be a split."""
    with pytest.raises(ValueError, match="both triangles of a symmetric Hessian"):
        build_nlp(build_problem(mirror_continuous=True, mirror_value=1.0))
    with pytest.raises(ValueError, match="add them into a single entry") as info:
        build_nlp(build_problem(mirror_continuous=True, mirror_value=3.0))
    assert "triangles" not in str(info.value)


def test_the_removed_warning_says_what_replaced_it():
    with pytest.raises(AttributeError, match="removed in 0.3.0.*now raises ValueError"):
        _ = yapss.MirroredHessianPairWarning


@pytest.mark.parametrize("order", ["first", "second"])
def test_user_method_needs_no_continuous_derivatives_without_phases(order):
    """A parameter-only problem under "user" needs only the objective derivatives.

    ``Problem.validate()`` requires ``continuous_jacobian``/``continuous_hessian`` only
    when there are phases; through 0.2.2 ``make_user_functions`` demanded them anyway,
    so validate() passed and solve() raised for the same problem.
    """
    problem = Problem(name="parameter-only", nx=[], ns=2)

    def objective(arg):
        s = arg.parameter
        arg.objective = (s[0] - 1.0) ** 2 + (s[1] - 2.0) ** 2

    def objective_gradient(arg):
        s = arg.parameter
        arg.gradient[0, "s", 0] = 2 * (s[0] - 1.0)
        arg.gradient[0, "s", 1] = 2 * (s[1] - 2.0)

    def objective_hessian(arg):
        arg.hessian[(0, "s", 0), (0, "s", 0)] = 2.0
        arg.hessian[(0, "s", 1), (0, "s", 1)] = 2.0

    problem.functions.objective = objective
    problem.functions.objective_gradient = objective_gradient
    problem.functions.objective_hessian = objective_hessian
    problem.derivatives.method = "user"
    problem.derivatives.order = order
    problem.guess.parameter = [0.0, 0.0]
    problem.ipopt_options.print_level = 0

    problem.validate()
    solution = problem.solve()
    assert solution.nlp_info.ipopt_status == 0
    np.testing.assert_allclose(solution.parameter, [1.0, 2.0], atol=1e-6)


def test_user_method_still_requires_continuous_jacobian_with_phases():
    problem = Problem(name="has-phase", nx=[1], nu=[1])

    def objective(arg):
        arg.objective = arg.phase[0].final_state[0]

    def objective_gradient(arg):
        arg.gradient[0, "xf", 0] = 1.0

    def continuous(arg):
        for p in arg.phase_list:
            arg.phase[p].dynamics[:] = (arg.phase[p].control[0],)

    problem.functions.objective = objective
    problem.functions.objective_gradient = objective_gradient
    problem.functions.continuous = continuous
    problem.derivatives.method = "user"
    problem.derivatives.order = "first"
    problem.guess.phase[0].time = [0.0, 1.0]
    with pytest.raises(ValueError, match="continuous_jacobian"):
        problem.solve()
