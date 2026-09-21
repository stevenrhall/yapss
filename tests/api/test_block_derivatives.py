"""Block fields in a derivative, which take a row index on whichever side they appear.

The problem here is synthetic, and deliberately so: no ported example has a block field in
every position a derivative can name, and none has a parameter at all. This one has a block
state, control, path, integral, parameter and discrete group, a scalar beside each, and a
continuous callback that reads the independent variable -- so every site that can take a row
index is exercised, crossed with the two derivative orders.

The check is the one `test_user_derivatives` uses: build the NLP twice from the same problem,
once from these callbacks and once from automatic differentiation, and compare the assembled
Jacobian and Hessian at an off-guess point.
"""

import numpy as np
import pytest

import yapss
from yapss._api.compile import _make_continuous_derivative, to_transcription_spec
from yapss._api.derivatives import ContinuousJacobian
from yapss._api.spec import snapshot
from yapss._backend.auto import make_auto_functions
from yapss._backend.guess import make_initial_guess_nlp
from yapss._backend.input_args import ContinuousStore, call_callback
from yapss._backend.mesh import Mesh
from yapss._backend.structure import get_nlp_dv_structure
from yapss.examples.delta_iii_ascent import (
    CD,
    MASS_FLOW,
    THRUST,
    R_e,
    S,
    h0,
    mu,
    omega_e,
    rho0,
)
from yapss.examples.delta_iii_ascent import setup as delta_iii
from yapss.math import exp

from .test_user_derivatives import SPECTRAL_METHODS, _build, _dense_hessian, _dense_jacobian


class State(yapss.State):
    r = yapss.vector(3)
    m = yapss.scalar()


class Control(yapss.Control):
    u = yapss.vector(2)


class Path(yapss.Path):
    limit = yapss.vector(2)


class Integral(yapss.Integral):
    cost = yapss.vector(2)


class Parameter(yapss.Parameter):
    beta = yapss.vector(2)
    gamma = yapss.scalar()


class Discrete(yapss.Discrete):
    gap = yapss.vector(3)
    total = yapss.scalar()


class Phases(yapss.Phases):
    only = yapss.phase(state=State, control=Control, path=Path, integral=Integral)


def _make(user):
    problem = yapss.Problem("blocks", phases=Phases, discrete=Discrete, parameter=Parameter)
    ph = problem.phases.only

    @ph.register.continuous
    def dynamics(arg, out):
        r, m, u = arg.state.r, arg.state.m, arg.control.u
        beta, gamma = arg.parameter.beta, arg.parameter.gamma
        t = arg.time
        out.dynamics.r = [r[1] * m, u[0] ** 2, beta[0] * r[0] + gamma * t]
        out.dynamics.m = -u[1] * m
        out.path.limit = [r[0] ** 2 + u[0] ** 2, m * u[1]]
        out.integrand.cost = [r[2] ** 2, beta[1] * u[0]]
        return out

    @problem.register.objective
    def objective(arg):
        e = arg[ph]
        return e.final.r[0] ** 2 + arg.parameter.beta[1] * e.initial.m + e.integral.cost[0]

    @problem.register.discrete
    def discrete(arg, out):
        e = arg[ph]
        out.discrete.gap = [e.final.r[i] - e.initial.r[i] for i in range(3)]
        out.discrete.total = e.final.r[0] * e.final.r[1] + arg.parameter.beta[0] ** 2
        return out

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 1.0)
    ph.time.guess = (0.0, 1.0)
    ph.state.r.bounds[:] = (-10.0, 10.0)
    ph.state.m.bounds = (0.5, 5.0)
    ph.control.u.bounds[:] = [(-2.0, 2.0)] * 2
    ph.path.limit.bounds[:] = [(-50.0, 50.0)] * 2
    ph.integral.cost.bounds[:] = [(-100.0, 100.0)] * 2
    problem.parameter.beta.bounds[:] = [(-3.0, 3.0)] * 2
    problem.parameter.gamma.bounds = (-3.0, 3.0)
    problem.discrete.gap.bounds[:] = (0.0, 0.0)
    problem.discrete.total.bounds = (0.0, 5.0)

    ph.state.r.guess[:] = [(0.3, 1.1), (0.4, 1.2), (0.5, 1.3)]
    ph.state.m.guess = (1.0, 2.0)
    ph.control.u.guess[:] = [(0.2, 0.7), (0.3, 0.8)]
    ph.integral.cost.guess[:] = [0.4, 0.6]
    problem.parameter.beta.guess[:] = [0.7, 0.9]
    problem.parameter.gamma.guess = 1.1

    if user:
        _derivatives(problem, ph)
        problem.derivatives.method = "user"
    problem.derivatives.order = "second"
    problem.ipopt_options.print_level = 0
    return problem


def _derivatives(problem, ph):
    @ph.register.continuous_jacobian
    def jac(arg, jacobian):
        r, m, u = arg.state.r, arg.state.m, arg.control.u
        beta, gamma = arg.parameter.beta, arg.parameter.gamma
        t = arg.time
        # dynamics.r[0] = r[1] * m
        jacobian.dynamics.r[0].r[1] = m
        jacobian.dynamics.r[0].m = r[1]
        # dynamics.r[1] = u[0] ** 2
        jacobian.dynamics.r[1].u[0] = 2 * u[0]
        # dynamics.r[2] = beta[0] * r[0] + gamma * t
        jacobian.dynamics.r[2].r[0] = beta[0]
        jacobian.dynamics.r[2].beta[0] = r[0]
        jacobian.dynamics.r[2].gamma = t
        jacobian.dynamics.r[2].time = gamma
        # dynamics.m = -u[1] * m
        jacobian.dynamics.m.m = -u[1]
        jacobian.dynamics.m.u[1] = -m
        # path.limit
        jacobian.path.limit[0].r[0] = 2 * r[0]
        jacobian.path.limit[0].u[0] = 2 * u[0]
        jacobian.path.limit[1].m = u[1]
        jacobian.path.limit[1].u[1] = m
        # integrand.cost
        jacobian.integrand.cost[0].r[2] = 2 * r[2]
        jacobian.integrand.cost[1].u[0] = beta[1]
        jacobian.integrand.cost[1].beta[1] = u[0]
        return jacobian

    @ph.register.continuous_hessian
    def hess(arg, hessian):
        hessian.dynamics.r[0].r[1].m = 1.0
        hessian.dynamics.r[1].u[0].u[0] = 2.0
        hessian.dynamics.r[2].r[0].beta[0] = 1.0
        hessian.dynamics.r[2].gamma.time = 1.0
        hessian.dynamics.m.m.u[1] = -1.0
        hessian.path.limit[0].r[0].r[0] = 2.0
        hessian.path.limit[0].u[0].u[0] = 2.0
        hessian.path.limit[1].m.u[1] = 1.0
        hessian.integrand.cost[0].r[2].r[2] = 2.0
        hessian.integrand.cost[1].u[0].beta[1] = 1.0
        return hessian

    @problem.register.objective_gradient
    def grad(arg, gradient):
        e = arg[ph]
        initial, final = gradient.phases[ph].initial, gradient.phases[ph].final
        gradient[final.r[0]] = 2 * e.final.r[0]
        gradient[initial.m] = arg.parameter.beta[1]
        gradient[gradient.parameter.beta[1]] = e.initial.m
        gradient[gradient.phases[ph].integral.cost[0]] = 1.0
        return gradient

    @problem.register.objective_hessian
    def ohess(_arg, hessian):
        initial, final = hessian.phases[ph].initial, hessian.phases[ph].final
        hessian[final.r[0], final.r[0]] = 2.0
        hessian[initial.m, hessian.parameter.beta[1]] = 1.0
        return hessian

    @problem.register.discrete_jacobian
    def djac(arg, jacobian):
        e = arg[ph]
        initial, final = jacobian.phases[ph].initial, jacobian.phases[ph].final
        for i in range(3):
            jacobian.discrete.gap[i][final.r[i]] = 1.0
            jacobian.discrete.gap[i][initial.r[i]] = -1.0
        jacobian.discrete.total[final.r[0]] = e.final.r[1]
        jacobian.discrete.total[final.r[1]] = e.final.r[0]
        jacobian.discrete.total[jacobian.parameter.beta[0]] = 2 * arg.parameter.beta[0]
        return jacobian

    @problem.register.discrete_hessian
    def dhess(_arg, hessian):
        final, beta = hessian.phases[ph].final, hessian.parameter.beta
        hessian.discrete.total[final.r[0], final.r[1]] = 1.0
        hessian.discrete.total[beta[0], beta[0]] = 2.0
        return hessian


def setup():
    """Return the problem with automatic differentiation."""
    return _make(user=False)


def setup_user():
    """Return the same problem with every derivative written by hand."""
    return _make(user=True)


@pytest.mark.parametrize("spectral_method", SPECTRAL_METHODS)
def test_block_derivatives_match_auto(spectral_method):
    """Every block site must assemble to what automatic differentiation assembles."""
    nlp_auto, z0 = _build(setup(), spectral_method)
    nlp_user, _ = _build(setup_user(), spectral_method)
    nz = len(z0)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(nz))
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(len(nlp_auto.constraints(z))))
    sigma = np.float64(1.3)

    jac_auto = _dense_jacobian(nlp_auto, np.asarray(nlp_auto.jacobian(z)), nz)
    jac_user = _dense_jacobian(nlp_user, np.asarray(nlp_user.jacobian(z)), nz)
    np.testing.assert_allclose(jac_user, jac_auto, rtol=0, atol=1e-13)

    hess_auto = _dense_hessian(nlp_auto, np.asarray(nlp_auto.hessian(z, lam, sigma)), nz)
    hess_user = _dense_hessian(nlp_user, np.asarray(nlp_user.hessian(z, lam, sigma)), nz)
    np.testing.assert_allclose(hess_user, hess_auto, rtol=0, atol=1e-13)


def test_it_solves_to_the_same_answer():
    auto, user = setup().solve(), setup_user().solve()
    assert user.converged
    assert user.objective == pytest.approx(auto.objective, rel=1e-12)


# ------------------------------------------------ the same indices, on real block physics
#
# The Delta III ascent is the problem whose state really is a position, a velocity and a
# mass, so its dynamics are 3x3 blocks with cross terms through the drag and the rotating
# atmosphere. Its *continuous* derivatives are written here and compared against `auto` key
# by key -- which checks the row indices on the shapes they exist for.
#
# Its discrete constraints are not written, and deliberately. Five of them are classical
# orbital elements, `arccos(dot(n_vec, e_vec) / (n * e))` and its siblings, which come to 30
# first and 105 second derivatives of arccosines of ratios of magnitudes. That is the work
# `development/prototype_discrete_derivatives.py` declines to do, calling it "the work that
# makes 'auto' the right method for this problem", and doing it here would test arithmetic
# rather than the API. The synthetic problem above covers the discrete block path instead.

SKEW = ((0.0, -omega_e, 0.0), (omega_e, 0.0, 0.0), (0.0, 0.0, 0.0))
"""d(omega x r)_i / dr_j for the Earth's rotation about z."""


def _make_delta_iii_jacobian(thrust):
    """Return the continuous Jacobian callback of a stage with the given thrust."""

    def stage_jacobian(arg, jacobian):
        r_vec, v_vec, m = arg.state.r, arg.state.v, arg.state.m
        u_vec = arg.control.u
        r = sum(r_vec[i] ** 2 for i in range(3)) ** 0.5
        rho = rho0 * exp(-(r - R_e) / h0)
        w = [v_vec[i] - sum(SKEW[i][j] * r_vec[j] for j in range(3)) for i in range(3)]
        speed = sum(w[i] ** 2 for i in range(3)) ** 0.5
        q = 0.5 * rho * speed * CD * S
        drag = [-q * w[i] for i in range(3)]

        for i in range(3):
            jacobian.dynamics.r[i].v[i] = 1.0

        d_rho = [-rho / h0 * r_vec[j] / r for j in range(3)]
        d_speed = [-sum(w[i] * SKEW[i][j] for i in range(3)) / speed for j in range(3)]
        d_q = [0.5 * CD * S * (d_rho[j] * speed + rho * d_speed[j]) for j in range(3)]
        for i in range(3):
            for j in range(3):
                jacobian.dynamics.v[i].r[j] = (
                    (3 * mu * r_vec[i] * r_vec[j] / r**5)
                    - (mu / r**3 if i == j else 0.0)
                    + (-w[i] * d_q[j] + q * SKEW[i][j]) / m
                )
        d_q_dv = [0.5 * rho * CD * S * w[j] / speed for j in range(3)]
        for i in range(3):
            for j in range(3):
                jacobian.dynamics.v[i].v[j] = -(w[i] * d_q_dv[j] + (q if i == j else 0.0)) / m
        for i in range(3):
            jacobian.dynamics.v[i].m = -(thrust * u_vec[i] + drag[i]) / m**2
            jacobian.dynamics.v[i].u[i] = thrust / m

        u_mag = sum(u_vec[i] ** 2 for i in range(3)) ** 0.5
        for i in range(3):
            jacobian.path.unit_thrust.u[i] = u_vec[i] / u_mag
            jacobian.path.radius.r[i] = r_vec[i] / r
        return jacobian

    return stage_jacobian


def test_delta_iii_continuous_jacobian_matches_auto():
    """Hand-written block derivatives must reach the entries `auto` finds, and agree on them."""
    spec_auto = to_transcription_spec(snapshot(delta_iii()))
    mesh = Mesh(spec_auto.phases)
    mesh.set_matrices("lgl")
    z0 = make_initial_guess_nlp(spec_auto, mesh)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(len(z0)))
    dv = get_nlp_dv_structure(spec_auto, np.float64)
    dv.z[:] = z

    user_problem = delta_iii()
    for stage, thrust in zip(user_problem.phases, THRUST, strict=True):
        stage.register.continuous_jacobian(_make_delta_iii_jacobian(thrust))

    def entries(function):
        store = ContinuousStore(spec_auto, dv=dv, dtype=np.float64, tau_u=mesh.tau_u)
        store._sync(z)
        arg = store.jacobian_arg
        call_callback(function, arg)
        return [dict(arg.phase[p].jacobian) for p in range(spec_auto.np)]

    auto = entries(make_auto_functions(spec_auto).continuous_jacobian)
    user = entries(
        _make_continuous_derivative(
            snapshot(user_problem), "jacobian", ContinuousJacobian, "jacobian"
        )
    )
    for phase, (by_auto, by_hand) in enumerate(zip(auto, user, strict=True)):
        assert set(by_hand) == set(by_auto), f"phase {phase} structure differs"
        for key, value in by_hand.items():
            np.testing.assert_allclose(value, by_auto[key], rtol=0, atol=1e-15)
