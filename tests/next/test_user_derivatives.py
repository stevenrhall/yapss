"""Derivatives supplied by hand, under ``derivatives.method = "user"``.

Two checks, and they answer different questions.

`test_assembled_derivatives_match_auto` builds the NLP twice from one problem, once from the
hand-written callbacks and once from automatic differentiation, and compares the dense
Jacobian and Hessian at an off-guess point. Both paths are exact, so they must agree to
roundoff. This is the stronger of the two: it checks the names, the keys they translate to,
and the deduced sparsity structure at every entry, at a point nothing was tuned for, and it
does not depend on a solve converging at all.

`test_the_solved_objective_matches_auto` then checks that the whole thing works end to end.

Newton's problem is the second formulation, `setup2`, deliberately. The first one puts the
nosecone's flat tip at zero radius, where the profile has a corner, and Ipopt stops on its KKT
test with the objective still moving: at the default `tol` of 1e-8 the *built-in* methods
disagree with each other at 1.6e-7 there, and the spread tracks the tolerance down (4.5e-10 at
1e-10, 3.2e-12 at 1e-12) rather than sitting on a floor. `setup2` optimizes the radius of the
flat portion instead, which removes the corner; its two paths agree to 2.2e-16 at the default
tolerance. It is also the better exercise, being the only problem here whose objective
gradient reads a value from `arg` and whose objective Hessian has an entry in it.
"""

import numpy as np
import pytest

from yapss import _next as yapss
from yapss._next.compile import to_transcription_spec
from yapss._next.examples import brachistochrone_user_derivatives as user_brachistochrone
from yapss._next.examples.brachistochrone import setup as auto_brachistochrone
from yapss._next.examples.newton import setup2 as auto_newton
from yapss._next.spec import snapshot, validate_problem
from yapss._private.auto import make_auto_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss._private.user import make_user_functions

SPECTRAL_METHODS = ("lgl", "lgr", "lg")


# --------------------------------------------------------------------------------- problems


def user_newton():
    """Return Newton's minimal resistance problem with its derivatives written by hand.

    The phase runs over a radius rather than a time, so the independent variable is named
    ``r`` here as it is everywhere else: ``jacobian.integrand.drag.r``. It is the case 3.1
    exists for, and the only problem in stage 1 whose integrand has derivatives at all.

    The objective carries the flat tip's own drag, ``4 r0**2``, so the gradient has an entry
    that reads a value from `arg` and the Hessian has one at all -- neither of which the
    brachistochrone reaches.
    """
    problem = auto_newton()
    ph = problem.phases.nose

    @ph.register.continuous_jacobian
    def nose_jacobian(arg, jacobian):
        yp, r = arg.state.yp, arg.r
        jacobian.dynamics.y.yp = 1.0
        jacobian.dynamics.yp.u = 1.0
        jacobian.integrand.drag.r = 8 / (1 + yp**2)
        jacobian.integrand.drag.yp = -16 * yp * r / (1 + yp**2) ** 2
        return jacobian

    @ph.register.continuous_hessian
    def nose_hessian(arg, hessian):
        yp, r = arg.state.yp, arg.r
        hessian.integrand.drag.yp.yp = 16 * r * (3 * yp**2 - 1) / (1 + yp**2) ** 3
        hessian.integrand.drag.yp.r = -16 * yp / (1 + yp**2) ** 2
        return hessian

    @problem.register.objective_gradient
    def drag_gradient(arg, gradient):
        gradient[ph].integral.drag = 1.0
        gradient[ph].initial.r = 8 * arg[ph].initial.r
        return gradient

    @problem.register.objective_hessian
    def drag_hessian(_arg, hessian):
        hessian[ph].initial.r[ph].initial.r = 8.0
        return hessian

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    return problem


PROBLEMS = {
    "brachistochrone": (auto_brachistochrone, user_brachistochrone.setup),
    "newton": (auto_newton, user_newton),
}


# ------------------------------------------------------------- the assembled-derivative check


def _build(problem, spectral_method):
    """Return the NLP of `problem` and the guess it was built at."""
    problem.method = spectral_method
    validate_problem(problem)
    spec = to_transcription_spec(snapshot(problem))
    mesh = Mesh(spec.phases)
    mesh.set_matrices(spectral_method)
    z0 = make_initial_guess_nlp(spec, mesh)
    if spec.derivative_method == "user":
        functions = make_user_functions(spec, z0, mesh.tau_u)
    else:
        functions = make_auto_functions(spec)
    return NLP(spec, functions, mesh), z0


def _dense_jacobian(nlp, values, nz):
    """Reconstruct the dense Jacobian from the sparse triple, summing duplicates.

    Written out here rather than imported from `tests/modules`, which is not a package.
    """
    row, col = nlp.jacobianstructure()
    dense = np.zeros((max(row) + 1 if len(row) else 0, nz))
    for r, c, v in zip(row, col, values, strict=True):
        dense[r, c] += v
    return dense


def _dense_hessian(nlp, values, nz):
    """Reconstruct the dense symmetric Hessian, folding duplicates and mirroring."""
    row, col = nlp.hessianstructure()
    lower = np.zeros((nz, nz))
    for r, c, v in zip(row, col, values, strict=True):
        i, j = (r, c) if r >= c else (c, r)
        lower[i, j] += v
    return lower + lower.T - np.diag(np.diag(lower))


@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("spectral_method", SPECTRAL_METHODS)
def test_assembled_derivatives_match_auto(name, spectral_method):
    """Hand-written derivatives must assemble to what automatic differentiation assembles."""
    make_auto, make_user = PROBLEMS[name]
    nlp_auto, z0 = _build(make_auto(), spectral_method)
    nlp_user, _ = _build(make_user(), spectral_method)

    # off the guess, so that no term sits at a symmetric or zero point that could hide an
    # error in a sign or in an index
    nz = len(z0)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(nz))
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(len(nlp_auto.constraints(z))))
    sigma = np.float64(1.3)

    jac_auto = _dense_jacobian(nlp_auto, np.asarray(nlp_auto.jacobian(z)), nz)
    jac_user = _dense_jacobian(nlp_user, np.asarray(nlp_user.jacobian(z)), nz)
    scale = max(1.0, np.abs(jac_auto).max())
    np.testing.assert_allclose(jac_user / scale, jac_auto / scale, rtol=0, atol=1e-13)

    hess_auto = _dense_hessian(nlp_auto, np.asarray(nlp_auto.hessian(z, lam, sigma)), nz)
    hess_user = _dense_hessian(nlp_user, np.asarray(nlp_user.hessian(z, lam, sigma)), nz)
    scale = max(1.0, np.abs(hess_auto).max())
    np.testing.assert_allclose(hess_user / scale, hess_auto / scale, rtol=0, atol=1e-13)


@pytest.mark.parametrize("name", PROBLEMS)
@pytest.mark.parametrize("spectral_method", SPECTRAL_METHODS)
def test_the_solved_objective_matches_auto(name, spectral_method):
    """Solved end to end, the two paths must reach the same answer."""
    problems = []
    for setup in PROBLEMS[name]:
        problem = setup()
        problem.method = spectral_method
        problem.ipopt_options.print_level = 0
        problems.append(problem)
    auto, user = (problem.solve() for problem in problems)
    assert user.converged
    assert user.objective == pytest.approx(auto.objective, rel=1e-12)


# ---------------------------------------------------------------- what the callbacks refuse


class Slide(yapss.Vector):
    x = yapss.field()
    v = yapss.field()


class Angle(yapss.Vector):
    u = yapss.field()


class Block(yapss.Vector):
    """A state with one block field and one scalar, so each side can be tested alone."""

    r = yapss.field(size=3)
    m = yapss.field()


class Phases(yapss.Phases):
    slide = yapss.phase(state=Slide, control=Angle)


def bare_problem():
    """Return a minimal problem with its value callbacks registered and nothing else."""
    problem = yapss.Problem("refusals", phases=Phases)
    ph = problem.phases.slide

    @ph.register.continuous
    def slide(arg, out):
        out.dynamics.x = arg.state.v
        out.dynamics.v = arg.control.u
        return out

    @problem.register.objective
    def final_time(arg):
        return arg[ph].final.time

    ph.time.initial = 0.0
    ph.time.guess = (0.0, 1.0)
    problem.derivatives.method = "user"
    return problem


def _targets(problem):
    """Return the four derivative targets, without running a solve.

    The targets are what the callbacks are handed, so reaching them directly is how the
    grammar is tested without a converging problem behind it.
    """
    from yapss._next.derivatives import (
        ContinuousHessian,
        ContinuousJacobian,
        ObjectiveGradient,
        ObjectiveHessian,
        Structure,
        endpoint_columns,
        phase_columns,
    )

    spec = snapshot(problem)
    columns = phase_columns(spec.phases[0], spec.parameter)
    ends = endpoint_columns(spec.phases, spec.parameter)
    return {
        "jacobian": ContinuousJacobian(Structure("jacobian"), columns),
        "hessian": ContinuousHessian(Structure("hessian"), columns),
        "gradient": ObjectiveGradient(Structure("gradient"), ends),
        "objective_hessian": ObjectiveHessian(Structure("hessian"), ends),
    }


@pytest.fixture
def problem():
    return bare_problem()


@pytest.fixture
def targets(problem):
    return _targets(problem)


@pytest.fixture
def ph(problem):
    return problem.phases.slide


def test_an_unknown_variable_is_refused_with_a_suggestion(targets):
    with pytest.raises(AttributeError, match=r"no variable 'vv'.*Did you mean 'v'"):
        targets["jacobian"].dynamics.x.vv = 1.0


def test_an_unknown_output_row_is_refused_with_a_suggestion(targets):
    with pytest.raises(AttributeError, match=r"dynamics has no 'xx'.*Did you mean 'x'"):
        targets["jacobian"].dynamics.xx.v = 1.0


def test_a_jacobian_row_needs_a_variable(targets):
    with pytest.raises(AttributeError, match=r"names a row but no variable"):
        targets["jacobian"].dynamics.x = 1.0


def test_a_jacobian_entry_cannot_chain(targets):
    with pytest.raises(AttributeError, match=r"belongs in the hessian callback"):
        _ = targets["jacobian"].dynamics.x.v.u


def test_a_hessian_entry_needs_two_variables(targets):
    with pytest.raises(AttributeError, match=r"A second derivative names two"):
        targets["hessian"].dynamics.x.v = 1.0


def test_a_hessian_entry_takes_no_third_variable(targets):
    with pytest.raises(AttributeError, match=r"names two variables"):
        _ = targets["hessian"].dynamics.x.v.u.u


def test_a_mirrored_hessian_pair_is_refused(targets):
    """The two orders name one derivative; summing them would double a user's own split."""
    hessian = targets["hessian"]
    hessian.dynamics.x.v.u = 1.0
    with pytest.raises(ValueError, match=r"same second derivative written both ways round"):
        hessian.dynamics.x.u.v = 1.0


def test_one_hessian_pair_may_be_rewritten(targets):
    """The same spelling twice is a plain overwrite, not a mirrored pair."""
    hessian = targets["hessian"]
    hessian.dynamics.x.v.u = 1.0
    hessian.dynamics.x.v.u = 2.0


def test_a_diagonal_hessian_entry_is_not_a_mirrored_pair(targets):
    targets["hessian"].dynamics.x.v.v = 1.0


def test_a_gradient_takes_a_phase_handle(targets):
    with pytest.raises(KeyError, match=r"takes a phase handle"):
        targets["gradient"]["slide"].final.time = 1.0


def test_an_endpoint_derivative_names_an_end(targets, ph):
    with pytest.raises(AttributeError, match=r"'initial', 'final' or 'integral'"):
        targets["gradient"][ph].finel.time = 1.0


def test_an_unknown_parameter_is_refused(targets):
    with pytest.raises(AttributeError, match=r"no parameter 'beta'"):
        targets["gradient"].beta = 1.0


def test_an_objective_hessian_chains_through_its_phase(targets, ph):
    """``hessian[ph].initial.x[ph].final.x`` names two endpoint variables, each by its phase."""
    targets["objective_hessian"][ph].initial.x[ph].final.x = 3.0


def test_an_objective_hessian_needs_its_second_phase(targets, ph):
    with pytest.raises(AttributeError, match=r"A second derivative names two"):
        targets["objective_hessian"][ph].initial.x = 3.0


# ------------------------------------------------------------------ block fields, stage 1


class BlockPhases(yapss.Phases):
    only = yapss.phase(state=Block, control=Angle)


def _block_jacobian():
    """Return a continuous Jacobian target for a phase whose state has a block field."""
    from yapss._next.derivatives import ContinuousJacobian, Structure, phase_columns

    problem = yapss.Problem("blocks", phases=BlockPhases)

    @problem.register.objective
    def nothing(arg):
        return 0.0

    problem.phases.only.time.guess = (0.0, 1.0)
    spec = snapshot(problem)
    return ContinuousJacobian(Structure("jacobian"), phase_columns(spec.phases[0], spec.parameter))


def test_a_block_field_is_refused_for_now():
    """Block-field rows take an index on either side, which stage 1 does not build."""
    jacobian = _block_jacobian()
    with pytest.raises(AttributeError, match=r"block field, which derivatives supplied by hand"):
        jacobian.dynamics.r.u = 1.0


def test_a_block_field_is_refused_as_a_variable_too():
    """The refusal covers the column side as well as the row side."""
    from yapss._next.derivatives import ContinuousJacobian, Structure, phase_columns

    jacobian = _block_jacobian()
    with pytest.raises(AttributeError, match=r"block field, which derivatives supplied by hand"):
        jacobian.dynamics.m.r = 1.0


def test_a_scalar_beside_a_block_still_works():
    """Refusing the block does not refuse the scalars declared beside it."""
    jacobian = _block_jacobian()
    jacobian.dynamics.m.u = 1.0


# ----------------------------------------------------------------- what validate() requires


def test_every_derivative_the_method_needs_is_required(problem):
    """Including the ones that are zero: leaving one out would look like forgetting it."""
    with pytest.raises(ValueError) as info:
        problem.validate()
    message = str(info.value)
    for which in ("continuous_jacobian", "continuous_hessian"):
        assert f"has no {which} callback" in message
    for which in ("objective_gradient", "objective_hessian"):
        assert f"has no {which} callback" in message


def test_first_order_does_not_ask_for_the_hessians(problem):
    problem.derivatives.order = "first"
    with pytest.raises(ValueError) as info:
        problem.validate()
    message = str(info.value)
    assert "continuous_jacobian" in message
    assert "objective_gradient" in message
    assert "continuous_hessian" not in message
    assert "objective_hessian" not in message


def test_the_message_says_how_to_register(problem):
    with pytest.raises(ValueError, match=r"@ph\.register\.continuous_jacobian"):
        problem.validate()


# ------------------------------------------------------------------ the sparsity structure


def test_the_structure_may_not_change_between_calls():
    """What is written is the structure, so a name that appears later is refused."""
    problem = user_brachistochrone.setup()
    problem.ipopt_options.print_level = 0
    ph = problem.phases.slide
    calls = [0]

    @ph.register.continuous_jacobian(replace=True)
    def drifting(arg, jacobian):
        v, u = arg.state.v, arg.control.u
        jacobian.dynamics.x.v = np.cos(u)
        jacobian.dynamics.x.u = -v * np.sin(u)
        jacobian.dynamics.y.v = np.sin(u)
        jacobian.dynamics.y.u = v * np.cos(u)
        jacobian.dynamics.v.u = 32.174 * np.cos(u)
        calls[0] += 1
        if calls[0] > 1:
            jacobian.dynamics.v.v = 0.0
        return jacobian

    with pytest.raises(ValueError, match=r"jacobian\.dynamics\.v\.v, which it did not write"):
        problem.solve()


def test_a_dropped_name_is_refused_too():
    problem = user_brachistochrone.setup()
    problem.ipopt_options.print_level = 0
    ph = problem.phases.slide
    calls = [0]

    @ph.register.continuous_jacobian(replace=True)
    def shrinking(arg, jacobian):
        v, u = arg.state.v, arg.control.u
        jacobian.dynamics.x.v = np.cos(u)
        jacobian.dynamics.x.u = -v * np.sin(u)
        jacobian.dynamics.y.v = np.sin(u)
        jacobian.dynamics.y.u = v * np.cos(u)
        calls[0] += 1
        if calls[0] == 1:
            jacobian.dynamics.v.u = 32.174 * np.cos(u)
        return jacobian

    with pytest.raises(ValueError, match=r"did not write jacobian\.dynamics\.v\.u"):
        problem.solve()


# ------------------------------------------------------------------------- registration


def test_a_second_derivative_callback_is_refused(ph):

    @ph.register.continuous_jacobian
    def first(arg, jacobian):
        return jacobian

    with pytest.raises(ValueError, match=r"already has the continuous_jacobian callback"):
        ph.register.continuous_jacobian(first)


def test_replace_allows_a_second_one(ph):

    @ph.register.continuous_jacobian
    def first(arg, jacobian):
        return jacobian

    ph.register.continuous_jacobian(first, replace=True)


def test_a_derivative_callback_must_be_callable(problem):
    with pytest.raises(TypeError, match=r"continuous_jacobian callback must be callable"):
        problem.phases.slide.register.continuous_jacobian(3)


def test_the_registrations_are_offered_on_a_typo(problem):
    with pytest.raises(AttributeError, match=r"register\.continuous_jacobian"):
        problem.phases.slide.continuous_jacobain = None
