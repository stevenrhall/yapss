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

import yapss
from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot, validate_problem
from yapss._backend.auto import make_auto_functions
from yapss._backend.guess import make_initial_guess_nlp
from yapss._backend.mesh import Mesh
from yapss._backend.nlp import NLP
from yapss._backend.user import make_user_functions
from yapss.examples import brachistochrone_user_derivatives as user_brachistochrone
from yapss.examples.brachistochrone import setup as auto_brachistochrone
from yapss.examples.goddard_problem_1_phase import setup as user_goddard_1_phase
from yapss.examples.goddard_problem_3_phase import c, g, h0
from yapss.examples.goddard_problem_3_phase import setup as auto_goddard
from yapss.examples.goddard_problem_3_phase import sigma
from yapss.examples.newton import setup2 as auto_newton
from yapss.examples.orbit_raising import m_0, m_dot, mu
from yapss.examples.orbit_raising import setup as auto_orbit_raising
from yapss.examples.orbit_raising import thrust
from yapss.math import exp, sqrt

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
        nose = gradient.phases[ph]
        gradient[nose.integral.drag] = 1.0
        gradient[nose.initial.r] = 8 * arg[ph].initial.r
        return gradient

    @problem.register.objective_hessian
    def drag_hessian(_arg, hessian):
        r0 = hessian.phases[ph].initial.r
        hessian[r0, r0] = 8.0
        return hessian

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    return problem


def user_orbit_raising():
    """Return the orbit raising problem with its derivatives written by hand.

    Two things only this problem has. Its thrust acceleration depends explicitly on the
    independent variable, so its continuous derivatives have `time` columns and `time.time`
    and `u_r.time` cross terms -- the arms the released golden corpus keeps `orbit_raising`
    for. And its single discrete constraint is *nonlinear* in the final radius, so the
    discrete Hessian has an entry rather than being empty.
    """
    problem = auto_orbit_raising()
    ph = problem.phases.raise_

    @ph.register.continuous_jacobian
    def raising_jacobian(arg, jacobian):
        r, v_r, v_theta = arg.state.r, arg.state.v_r, arg.state.v_theta
        u_r, u_theta = arg.control.u_r, arg.control.u_theta
        mass = m_0 - m_dot * arg.time
        a = thrust / mass
        da = thrust * m_dot / mass**2
        jacobian.dynamics.r.v_r = 1.0
        jacobian.dynamics.theta.r = -v_theta / r**2
        jacobian.dynamics.theta.v_theta = 1 / r
        jacobian.dynamics.v_r.r = -(v_theta**2) / r**2 + 2 * mu / r**3
        jacobian.dynamics.v_r.v_theta = 2 * v_theta / r
        jacobian.dynamics.v_r.u_r = a
        jacobian.dynamics.v_r.time = da * u_r
        jacobian.dynamics.v_theta.r = v_r * v_theta / r**2
        jacobian.dynamics.v_theta.v_r = -v_theta / r
        jacobian.dynamics.v_theta.v_theta = -v_r / r
        jacobian.dynamics.v_theta.u_theta = a
        jacobian.dynamics.v_theta.time = da * u_theta
        jacobian.path.unit_thrust.u_r = 2 * u_r
        jacobian.path.unit_thrust.u_theta = 2 * u_theta
        return jacobian

    @ph.register.continuous_hessian
    def raising_hessian(arg, hessian):
        r, v_r, v_theta = arg.state.r, arg.state.v_r, arg.state.v_theta
        u_r, u_theta = arg.control.u_r, arg.control.u_theta
        mass = m_0 - m_dot * arg.time
        da = thrust * m_dot / mass**2
        dda = 2 * thrust * m_dot**2 / mass**3
        hessian.dynamics.theta.r.r = 2 * v_theta / r**3
        hessian.dynamics.theta.r.v_theta = -1 / r**2
        hessian.dynamics.v_r.r.r = 2 * v_theta**2 / r**3 - 6 * mu / r**4
        hessian.dynamics.v_r.r.v_theta = -2 * v_theta / r**2
        hessian.dynamics.v_r.v_theta.v_theta = 2 / r
        hessian.dynamics.v_r.u_r.time = da
        hessian.dynamics.v_r.time.time = dda * u_r
        hessian.dynamics.v_theta.r.r = -2 * v_r * v_theta / r**3
        hessian.dynamics.v_theta.r.v_r = v_theta / r**2
        hessian.dynamics.v_theta.r.v_theta = v_r / r**2
        hessian.dynamics.v_theta.v_r.v_theta = -1 / r
        hessian.dynamics.v_theta.u_theta.time = da
        hessian.dynamics.v_theta.time.time = dda * u_theta
        hessian.path.unit_thrust.u_r.u_r = 2.0
        hessian.path.unit_thrust.u_theta.u_theta = 2.0
        return hessian

    @problem.register.objective_gradient
    def largest_orbit_gradient(_arg, gradient):
        gradient[gradient.phases[ph].final.r] = 1.0
        return gradient

    @problem.register.objective_hessian
    def largest_orbit_hessian(_arg, hessian):
        return hessian

    @problem.register.discrete_jacobian
    def circular_jacobian(arg, jacobian):
        r = arg[ph].final.r
        final = jacobian.phases[ph].final
        jacobian.discrete.circular[final.v_theta] = 1.0
        jacobian.discrete.circular[final.r] = sqrt(mu) / (2 * r**1.5)
        return jacobian

    @problem.register.discrete_hessian
    def circular_hessian(arg, hessian):
        r = arg[ph].final.r
        rf = hessian.phases[ph].final.r
        hessian.discrete.circular[rf, rf] = -3 * sqrt(mu) / (4 * r**2.5)
        return hessian

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    return problem


def _rocket_terms(arg):
    """Return the rocket's state, its thrust, and the three drag terms the derivatives use."""
    h, v, mass = arg.state.h, arg.state.v, arg.state.m
    d2 = sigma * exp(-h / h0)
    d1 = d2 * v
    return v, mass, arg.control.thrust, d2, d1, d1 * v


def _rocket_jacobian(arg, jacobian):
    """Fill in the dynamics' first derivatives, which are the same in every phase."""
    _v, mass, thrust, _d2, d1, drag = _rocket_terms(arg)
    jacobian.dynamics.h.v = 1.0
    jacobian.dynamics.v.h = drag / (h0 * mass)
    jacobian.dynamics.v.v = -2 * d1 / mass
    jacobian.dynamics.v.m = -(thrust - drag) / mass**2
    jacobian.dynamics.v.thrust = 1 / mass
    jacobian.dynamics.m.thrust = -1 / c


def _rocket_hessian(arg, hessian):
    """Fill in the dynamics' second derivatives, which are the same in every phase."""
    _v, mass, thrust, d2, d1, drag = _rocket_terms(arg)
    hessian.dynamics.v.h.h = -drag / (h0**2 * mass)
    hessian.dynamics.v.h.v = 2 * d1 / (h0 * mass)
    hessian.dynamics.v.h.m = -drag / (h0 * mass**2)
    hessian.dynamics.v.v.v = -2 * d2 / mass
    hessian.dynamics.v.v.m = 2 * d1 / mass**2
    hessian.dynamics.v.m.m = 2 * (thrust - drag) / mass**3
    hessian.dynamics.v.m.thrust = -1 / mass**2


def user_goddard():
    """Return the three-phase Goddard rocket with its derivatives written by hand.

    The problem this stage exists for: eight discrete groups, each joining two of three
    phases, so every entry names its phase. They are linear, so the discrete Hessian is
    registered and writes nothing -- which is how "every entry of it is structurally zero" is
    said. The released version reaches the same entries as `jacobian[("f", 1), ("x", 0)]` and
    branches on `if p == 1` for the path; here the phases register separately, so the
    singular arc's extra rows live in its own callback and there is no branch.
    """
    problem = auto_goddard()
    boost, singular, coast = (
        problem.phases.boost,
        problem.phases.singular,
        problem.phases.coast,
    )

    def powered_jacobian(arg, jacobian):
        _rocket_jacobian(arg, jacobian)
        return jacobian

    def powered_hessian(arg, hessian):
        _rocket_hessian(arg, hessian)
        return hessian

    for ph in (boost, coast):
        ph.register.continuous_jacobian(powered_jacobian)
        ph.register.continuous_hessian(powered_hessian)

    @singular.register.continuous_jacobian
    def singular_jacobian(arg, jacobian):
        _rocket_jacobian(arg, jacobian)
        v, _mass, _thrust, _d2, d1, drag = _rocket_terms(arg)
        jacobian.path.switching.h = drag * (1 + v / c) / h0
        jacobian.path.switching.v = drag * (-3 / c) - 2 * d1
        jacobian.path.switching.m = g
        return jacobian

    @singular.register.continuous_hessian
    def singular_hessian(arg, hessian):
        _rocket_hessian(arg, hessian)
        v, _mass, _thrust, d2, d1, drag = _rocket_terms(arg)
        hessian.path.switching.h.h = -drag * (c + v) / (c * h0**2)
        hessian.path.switching.h.v = d1 * (2 * c + 3 * v) / (c * h0)
        hessian.path.switching.v.v = -2 * d2 * (c + 3 * v) / c
        return hessian

    @problem.register.objective_gradient
    def final_altitude_gradient(_arg, gradient):
        gradient[gradient.phases[coast].final.h] = 1.0
        return gradient

    @problem.register.objective_hessian
    def final_altitude_hessian(_arg, hessian):
        return hessian

    @problem.register.discrete_jacobian
    def linkage_jacobian(_arg, jacobian):
        """Each group is `later - earlier`, so its derivatives are +1 and -1 and nothing else."""
        d = jacobian.discrete
        b, s, c = (jacobian.phases[phase] for phase in (boost, singular, coast))
        d.boost_singular_h[s.initial.h] = 1.0
        d.boost_singular_h[b.final.h] = -1.0
        d.boost_singular_v[s.initial.v] = 1.0
        d.boost_singular_v[b.final.v] = -1.0
        d.boost_singular_m[s.initial.m] = 1.0
        d.boost_singular_m[b.final.m] = -1.0
        d.boost_singular_time[s.initial.time] = 1.0
        d.boost_singular_time[b.final.time] = -1.0
        d.singular_coast_h[c.initial.h] = 1.0
        d.singular_coast_h[s.final.h] = -1.0
        d.singular_coast_v[c.initial.v] = 1.0
        d.singular_coast_v[s.final.v] = -1.0
        d.singular_coast_m[c.initial.m] = 1.0
        d.singular_coast_m[s.final.m] = -1.0
        d.singular_coast_time[c.initial.time] = 1.0
        d.singular_coast_time[s.final.time] = -1.0
        return jacobian

    @problem.register.discrete_hessian
    def linkage_hessian(_arg, hessian):
        """The linkage constraints are linear, so writing nothing is the whole of it."""
        return hessian

    problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    return problem


def auto_goddard_1_phase():
    """Return the one-phase Goddard problem with automatic differentiation instead.

    Unlike the others here, this problem's derivative callbacks live in the example itself, so
    the hand-written side is `setup` unaltered and this is the one that has to be changed.
    """
    problem = user_goddard_1_phase()
    problem.derivatives.method = "auto"
    return problem


PROBLEMS = {
    "brachistochrone": (auto_brachistochrone, user_brachistochrone.setup),
    "newton": (auto_newton, user_newton),
    "orbit_raising": (auto_orbit_raising, user_orbit_raising),
    "goddard": (auto_goddard, user_goddard),
    "goddard_1_phase": (auto_goddard_1_phase, user_goddard_1_phase),
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


class Slide(yapss.State):
    x = yapss.scalar()
    v = yapss.scalar()


class Angle(yapss.Control):
    u = yapss.scalar()


class Block(yapss.State):
    """A state with one block field and one scalar, so each side can be tested alone."""

    r = yapss.vector(3)
    m = yapss.scalar()


class SlidePhase(yapss.Phase):
    state: Slide
    control: Angle


class Phases(yapss.Phases):
    slide: SlidePhase


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

    ph.time.initial = (0.0, 0.0)
    ph.time.guess = (0.0, 1.0)
    problem.derivatives.method = "user"
    return problem


def _targets(problem):
    """Return the four derivative targets, without running a solve.

    The targets are what the callbacks are handed, so reaching them directly is how the
    grammar is tested without a converging problem behind it.
    """
    from yapss._api.derivatives import (
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
        _ = targets["gradient"].phases["slide"]


def test_an_endpoint_derivative_names_an_end(targets, ph):
    with pytest.raises(AttributeError, match=r"'initial', 'final' or 'integral'"):
        _ = targets["gradient"].phases[ph].finel


def test_an_unknown_parameter_is_refused(targets):
    with pytest.raises(AttributeError, match=r"the problem has no 'beta'"):
        _ = targets["gradient"].parameter.beta


def test_a_variable_is_not_a_derivative(targets, ph):
    """Assigning to the variable is the mistake the old spelling invited; it is named."""
    with pytest.raises(AttributeError, match=r"is a variable, not a derivative"):
        targets["gradient"].phases[ph].final.time = 1.0


def test_a_column_of_another_problem_is_refused(targets):
    """A column carries the namespace it came from, so it cannot land in the wrong problem."""
    elsewhere = bare_problem()
    other = _targets(elsewhere)["gradient"]
    column = other.phases[elsewhere.phases.slide].final.time
    with pytest.raises(ValueError, match=r"belongs to another problem"):
        targets["gradient"][column] = 1.0


def test_a_gradient_takes_one_variable(targets, ph):
    final = targets["gradient"].phases[ph].final
    with pytest.raises(TypeError, match=r"A first derivative names one"):
        targets["gradient"][final.time, final.x] = 1.0


def test_an_objective_hessian_names_two_variables(targets, ph):
    """``hessian[i.x, f.x]`` relates two endpoint variables, each a value in its own right."""
    phase = targets["objective_hessian"].phases[ph]
    targets["objective_hessian"][phase.initial.x, phase.final.x] = 3.0


def test_an_objective_hessian_needs_its_second_variable(targets, ph):
    with pytest.raises(TypeError, match=r"A second derivative names two"):
        targets["objective_hessian"][targets["objective_hessian"].phases[ph].initial.x] = 3.0


# ------------------------------------------------------- what the discrete targets refuse


class Link(yapss.Discrete):
    gap = yapss.scalar()


def linked_problem():
    """Return a minimal problem that declares one discrete constraint."""
    problem = yapss.Problem("linked", phases=Phases, discrete=Link)
    ph = problem.phases.slide

    @ph.register.continuous
    def slide(arg, out):
        out.dynamics.x = arg.state.v
        out.dynamics.v = arg.control.u
        return out

    @problem.register.objective
    def final_time(arg):
        return arg[ph].final.time

    @problem.register.discrete
    def link(arg, out):
        out.discrete.gap = arg[ph].final.x
        return out

    ph.time.initial = (0.0, 0.0)
    ph.time.guess = (0.0, 1.0)
    problem.discrete.gap.bounds = (0.0, 0.0)
    problem.derivatives.method = "user"
    return problem


def _discrete_targets(problem):
    """Return the two discrete targets, without running a solve."""
    from yapss._api.derivatives import (
        DiscreteHessian,
        DiscreteJacobian,
        Structure,
        endpoint_columns,
    )

    spec = snapshot(problem)
    ends = endpoint_columns(spec.phases, spec.parameter, spec.discrete)
    return {
        "jacobian": DiscreteJacobian(Structure("jacobian"), ends),
        "hessian": DiscreteHessian(Structure("hessian"), ends),
    }


@pytest.fixture
def linked():
    return linked_problem()


@pytest.fixture
def discrete_targets(linked):
    return _discrete_targets(linked)


def test_a_discrete_derivative_names_a_group(discrete_targets, linked):
    jacobian = discrete_targets["jacobian"]
    jacobian.discrete.gap[jacobian.phases[linked.phases.slide].final.x] = 1.0


def test_an_unknown_group_is_refused_with_a_suggestion(discrete_targets, linked):
    with pytest.raises(AttributeError, match=r"has no group 'gapp'.*Did you mean 'gap'"):
        _ = discrete_targets["jacobian"].discrete.gapp


def test_a_group_needs_a_variable(discrete_targets):
    with pytest.raises(AttributeError, match=r"names a constraint but no variable"):
        discrete_targets["jacobian"].discrete.gap = 1.0


def test_a_discrete_derivative_names_discrete(discrete_targets):
    with pytest.raises(AttributeError, match=r"a derivative is written against"):
        _ = discrete_targets["jacobian"].dynamics


def test_a_discrete_hessian_relates_two_variables(discrete_targets, linked):
    hessian = discrete_targets["hessian"]
    final = hessian.phases[linked.phases.slide].final
    hessian.discrete.gap[final.x, final.v] = 2.0


def test_a_mirrored_discrete_hessian_pair_is_refused(discrete_targets, linked):
    hessian = discrete_targets["hessian"]
    final = hessian.phases[linked.phases.slide].final
    hessian.discrete.gap[final.x, final.v] = 2.0
    with pytest.raises(ValueError, match=r"same second derivative written both ways round"):
        hessian.discrete.gap[final.v, final.x] = 2.0


class TwoLinks(yapss.Discrete):
    gap = yapss.scalar()
    slack = yapss.scalar()


def test_the_same_pair_in_two_groups_is_two_entries():
    """The group's row is part of the key, so one pair written in two groups is not mirrored.

    Dropping the row from the canonical key would make the second write look like the first
    one reversed, and it would be refused as a mirrored pair. It is not: they are second
    derivatives of two different constraints.
    """
    from yapss._api.derivatives import DiscreteHessian, Structure, endpoint_columns

    problem = yapss.Problem("two", phases=Phases, discrete=TwoLinks)
    ph = problem.phases.slide

    @problem.register.objective
    def nothing(arg):
        return arg[ph].final.time

    ph.time.guess = (0.0, 1.0)
    spec = snapshot(problem)
    store = Structure("hessian")
    hessian = DiscreteHessian(store, endpoint_columns(spec.phases, spec.parameter, spec.discrete))
    final = hessian.phases[ph].final
    hessian.discrete.gap[final.x, final.v] = 2.0
    hessian.discrete.slack[final.x, final.v] = 3.0
    assert len(store.entries) == 2
    assert sorted(store.entries.values()) == [2.0, 3.0]


def test_the_discrete_callbacks_are_required(linked):
    with pytest.raises(ValueError) as info:
        linked.validate()
    message = str(info.value)
    assert "has no discrete_jacobian callback" in message
    assert "has no discrete_hessian callback" in message
    assert "of a problem with discrete constraints" in message


def test_a_problem_without_discrete_constraints_needs_neither(problem):
    """`bare_problem` declares none, so neither discrete callback is asked for."""
    with pytest.raises(ValueError) as info:
        problem.validate()
    message = str(info.value)
    assert "discrete_jacobian" not in message
    assert "discrete_hessian" not in message


# --------------------------------------------------- block fields: what the row index does


class Only(yapss.Phase):
    state: Block
    control: Angle


class BlockPhases(yapss.Phases):
    only: Only


def _block_jacobian():
    """Return a continuous Jacobian target for a phase whose state has a block field."""
    from yapss._api.derivatives import ContinuousJacobian, Structure, phase_columns

    problem = yapss.Problem("blocks", phases=BlockPhases)

    @problem.register.objective
    def nothing(arg):
        return 0.0

    problem.phases.only.time.guess = (0.0, 1.0)
    spec = snapshot(problem)
    return ContinuousJacobian(Structure("jacobian"), phase_columns(spec.phases[0], spec.parameter))


def test_a_block_row_without_an_index_says_so():
    """A block names as many rows as it has, so it names no single one."""
    jacobian = _block_jacobian()
    with pytest.raises(AttributeError, match=r"block field of 3 rows.*Give the row"):
        jacobian.dynamics.r.u = 1.0


def test_a_block_variable_without_an_index_says_so():
    jacobian = _block_jacobian()
    with pytest.raises(AttributeError, match=r"block field of 3 rows.*Give the row"):
        jacobian.dynamics.m.r = 1.0


def test_a_row_index_out_of_range_is_refused():
    jacobian = _block_jacobian()
    with pytest.raises(IndexError, match=r"out of range for a block field of 3 rows"):
        jacobian.dynamics.r[3].u = 1.0


def test_a_negative_row_index_is_refused():
    """Wrapping would give one row two spellings, and so one derivative two names."""
    jacobian = _block_jacobian()
    with pytest.raises(IndexError, match=r"out of range"):
        jacobian.dynamics.r[-1].u = 1.0


def test_a_row_index_must_be_an_integer():
    jacobian = _block_jacobian()
    with pytest.raises(TypeError, match=r"takes a row index, an integer from 0 to 2"):
        jacobian.dynamics.r["x"].u = 1.0


def test_a_boolean_is_not_a_row_index():
    """`True` is an int to Python, and would silently mean row 1."""
    jacobian = _block_jacobian()
    with pytest.raises(TypeError, match=r"takes a row index"):
        jacobian.dynamics.r[True].u = 1.0


def test_a_scalar_beside_a_block_takes_no_index():
    jacobian = _block_jacobian()
    jacobian.dynamics.m.u = 1.0


def test_a_block_row_is_suggested_on_a_typo():
    """A block field's name is offered like any other, though it is not in the scalar map."""
    jacobian = _block_jacobian()
    with pytest.raises(AttributeError, match=r"Did you mean 'r'"):
        _ = jacobian.dynamics.rr


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
