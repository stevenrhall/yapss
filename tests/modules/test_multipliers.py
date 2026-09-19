"""
Continuous-time multipliers against a problem with a known solution.

Minimize x(tf) subject to xdot = u, x(0) = 0, and -1 <= u <= 1. The optimal control is
u = -1 throughout, the costate is identically 1, and stationarity of the Hamiltonian
H = lambda * u in u gives an active-bound multiplier of magnitude 1 -- for every final
time. The same bound expressed as a path constraint h = u must give the same multiplier.

Durations other than 2 discriminate: at tf - t0 = 2 the half-duration is 1 and any power
of it is invisible, which is how the scaling of both multipliers went wrong through 0.2.2.
"""

import numpy as np
import pytest

from yapss import _legacy as yapss
from yapss._backend.mesh import Mesh

DURATIONS = [2.0, 4.0, 0.5]
METHODS = ["lg", "lgr", "lgl"]


def _setup(tf: float, *, as_path: bool) -> yapss.Problem:
    problem = yapss.Problem(name="bound multiplier", nx=[1], nu=[1], nh=[1 if as_path else 0])

    def objective(arg):
        arg.objective = arg.phase[0].final_state[0]

    def continuous(arg):
        for p in arg.phase_list:
            u = arg.phase[p].control[0]
            arg.phase[p].dynamics[:] = (u,)
            if as_path:
                arg.phase[p].path[:] = (u,)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = tf
    bounds.initial_state.lower = bounds.initial_state.upper = [0.0]
    if as_path:
        bounds.path.lower, bounds.path.upper = [-1.0], [1.0]
    else:
        bounds.control.lower, bounds.control.upper = [-1.0], [1.0]
    problem.guess.phase[0].time = [0.0, tf]
    problem.guess.phase[0].state = [[0.0, -tf]]
    problem.guess.phase[0].control = [[-1.0, -1.0]]
    problem.mesh.phase[0].collocation_points = (5, 5)
    problem.mesh.phase[0].fraction = (0.5, 0.5)
    problem.ipopt_options.print_level = 0
    return problem


@pytest.mark.parametrize("spectral_method", METHODS)
@pytest.mark.parametrize("tf", DURATIONS)
def test_bound_multipliers_are_densities_in_time(spectral_method: str, tf: float) -> None:
    control_problem = _setup(tf, as_path=False)
    path_problem = _setup(tf, as_path=True)
    for problem in (control_problem, path_problem):
        problem.spectral_method = spectral_method
    control_solution = control_problem.solve()
    path_solution = path_problem.solve()

    for solution in (control_solution, path_solution):
        assert solution.nlp_info.ipopt_status == 0
        np.testing.assert_allclose(solution.phase[0].costate, 1.0, atol=1e-6)

    control_multiplier = control_solution.phase[0].control_multiplier
    path_multiplier = path_solution.phase[0].path_multiplier
    # magnitude 1, independent of tf: dH/du + mu = 0 with lambda = 1
    np.testing.assert_allclose(np.abs(control_multiplier), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.abs(path_multiplier), 1.0, atol=1e-6)
    # the same constraint, however expressed, has the same multiplier
    np.testing.assert_allclose(control_multiplier, path_multiplier, atol=1e-6)


def test_zero_duration_phase_multipliers_are_nan() -> None:
    """A density in time on a phase of zero duration is undefined, and says so quietly."""
    problem = _setup(1.0, as_path=True)
    problem.bounds.phase[0].control.lower = [-1.0]
    problem.bounds.phase[0].control.upper = [1.0]
    problem.bounds.phase[0].final_time.lower = 0.0
    problem.bounds.phase[0].final_time.upper = 0.0
    with np.errstate(all="raise"):
        solution = problem.solve()
    assert np.all(np.isnan(solution.phase[0].control_multiplier))
    assert np.all(np.isnan(solution.phase[0].path_multiplier))
    assert np.all(np.isfinite(solution.phase[0].costate))


def test_zero_duration_phase_multipliers_are_nan_whatever_ipopt_returns(monkeypatch) -> None:
    """NaN must not depend on the NLP multipliers happening to be exactly zero.

    The bound and the path constraint above duplicate each other, so how Ipopt splits the
    multiplier between them is build-dependent: conda-forge Ipopt 3.14.19 on macOS returned
    nonzero path-row values, which divided by the zero duration to -inf. Forcing every NLP
    multiplier nonzero makes the case deterministic on any build.
    """
    from yapss._backend import solver

    make_solution_object = solver.make_solution_object

    def nonzero_multipliers(problem, mesh, nlp, nlp_info, origin=None):
        for key in ("mult_g", "mult_x_L", "mult_x_U"):
            nlp_info[key] = np.linspace(0.5, 1.5, len(nlp_info[key]))
        nlp_info["mult_x_U"] = -nlp_info["mult_x_U"]
        return make_solution_object(problem, mesh, nlp, nlp_info, origin)

    monkeypatch.setattr(solver, "make_solution_object", nonzero_multipliers)
    problem = _setup(1.0, as_path=True)
    problem.bounds.phase[0].control.lower = [-1.0]
    problem.bounds.phase[0].control.upper = [1.0]
    problem.bounds.phase[0].final_time.lower = 0.0
    problem.bounds.phase[0].final_time.upper = 0.0
    with np.errstate(all="raise"):
        solution = problem.solve()
    assert np.all(np.isnan(solution.phase[0].control_multiplier))
    assert np.all(np.isnan(solution.phase[0].path_multiplier))


# ----------------------------------------------------------------------------------
# A multiplier is a derivative of the optimal objective with respect to the thing it
# belongs to, so a finite difference around a converged solve is an independent check
# of the whole conversion. For a per-point row the prediction is an *integral* of the
# reported density, sum_k h w_k mu_k, which is what the transcription's h and w_k
# normalization -- wrong until 0.2.3 -- has to get right. Stating it as an integral
# also keeps the check convention-neutral: it constrains a sum over the grid, never
# the split of a multiplier at a collocated endpoint.
#
# Established by the study of 2026-09-17 (development/MULTIPLIER_STUDY.md), which
# measured these identities over 480 configurations at a worst relative error of 1e-8.
# ----------------------------------------------------------------------------------

SCALE_KNOBS = {
    "objective": 4.0,
    "state": [2.5, 0.4],
    "control": [3.0],
    "dynamics": [0.7, 5.0],
    "integral": [8.0],
    "path": [0.25],
    "time": 2.0,
    "parameter": [6.0],
    "discrete": [0.125],
}


def _kitchen_sink(
    spectral_method: str = "lgr",
    *,
    scaled: bool = False,
    sense: str = "minimize",
    collocation_points: tuple[int, ...] = (6,),
) -> yapss.Problem:
    """Every reported multiplier kind, active at once, with nothing degenerate.

    The initial time is fixed and the final time free with an active duration bound, so
    the three time-related rows stay linearly independent; fixing t0 and tf *and*
    bounding the duration would make Ipopt's split among them arbitrary.
    """
    problem = yapss.Problem(name="kitchen sink", nx=[2], nu=[1], nq=[1], nh=[1], ns=1, nd=1)

    # under "maximize" the objective expression is negated too, so the same point is
    # optimal and the two solves are directly comparable
    sign = -1.0 if sense == "maximize" else 1.0

    def objective(arg):
        phase = arg.phase[0]
        arg.objective = sign * (
            phase.integral[0] - 2.0 * phase.final_state[0] + 0.5 * arg.parameter[0]
        )

    def continuous(arg):
        for p in arg.phase_list:
            x0, x1 = arg.phase[p].state
            (u,) = arg.phase[p].control
            (s,) = arg.parameter
            arg.phase[p].dynamics[0] = x1 + 0.1 * arg.phase[p].time
            arg.phase[p].dynamics[1] = u + s
            arg.phase[p].integrand[0] = 0.5 * (u * u + 0.2 * x0 * x0)
            arg.phase[p].path[0] = u + 0.3 * x0

    def discrete(arg):
        arg.discrete[0] = arg.phase[0].final_state[1] + arg.parameter[0]

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    problem.functions.discrete = discrete

    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower, bounds.final_time.upper = 0.5, 10.0
    bounds.duration.lower, bounds.duration.upper = 0.1, 2.0
    bounds.initial_state.lower = bounds.initial_state.upper = [1.0, 0.0]
    bounds.path.lower, bounds.path.upper = [-1.5], [0.6]
    problem.bounds.parameter.lower, problem.bounds.parameter.upper = [-1.0], [-0.05]
    problem.bounds.discrete.lower = problem.bounds.discrete.upper = [0.2]

    problem.guess.phase[0].time = [0.0, 2.0]
    problem.guess.phase[0].state = [[1.0, 1.0], [0.0, 0.0]]
    problem.guess.phase[0].control = [[0.0, 0.0]]
    problem.guess.parameter = [0.0]

    problem.mesh.phase[0].collocation_points = collocation_points
    problem.mesh.phase[0].fraction = tuple(
        1.0 / len(collocation_points) for _ in collocation_points
    )
    problem.spectral_method = spectral_method
    problem.sense = sense
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.tol = 1e-12

    if scaled:
        problem.scale.objective = SCALE_KNOBS["objective"]
        scale_phase = problem.scale.phase[0]
        scale_phase.state = SCALE_KNOBS["state"]
        scale_phase.control = SCALE_KNOBS["control"]
        scale_phase.dynamics = SCALE_KNOBS["dynamics"]
        scale_phase.integral = SCALE_KNOBS["integral"]
        scale_phase.path = SCALE_KNOBS["path"]
        scale_phase.time = SCALE_KNOBS["time"]
        problem.scale.parameter = SCALE_KNOBS["parameter"]
        problem.scale.discrete = SCALE_KNOBS["discrete"]
    return problem


def _reported_multipliers(solution: yapss.Solution) -> dict[str, np.ndarray]:
    phase = solution.phase[0]
    return {
        "costate": phase.costate,
        "control_multiplier": phase.control_multiplier,
        "path_multiplier": phase.path_multiplier,
        "integral_multiplier": phase.integral_multiplier,
        "initial_time_multiplier": np.array([phase.initial_time_multiplier]),
        "final_time_multiplier": np.array([phase.final_time_multiplier]),
        "duration_multiplier": np.array([phase.duration_multiplier]),
        "discrete_multiplier": solution.discrete_multiplier,
        "parameter_multiplier": solution.parameter_multiplier,
        "hamiltonian": phase.hamiltonian,
    }


@pytest.mark.parametrize("spectral_method", METHODS)
def test_multipliers_do_not_depend_on_problem_scaling(spectral_method: str) -> None:
    """Ipopt returns unscaled multipliers under user-scaling, so scale.* must not show.

    Nothing else in the suite sets a non-unit problem.scale.* and then reads a
    multiplier; if Ipopt ever returned them scaled, every reported multiplier would be
    wrong by a factor whenever the user scaled the problem, silently.
    """
    unit = _kitchen_sink(spectral_method).solve()
    scaled = _kitchen_sink(spectral_method, scaled=True).solve()
    assert unit.nlp_info.ipopt_status == 0
    assert scaled.nlp_info.ipopt_status == 0
    np.testing.assert_allclose(scaled.objective, unit.objective, rtol=1e-7)
    unit_multipliers = _reported_multipliers(unit)
    for name, value in _reported_multipliers(scaled).items():
        np.testing.assert_allclose(
            value, unit_multipliers[name], rtol=1e-5, atol=1e-7, err_msg=name
        )


def _integrate(solution: yapss.Solution, density: np.ndarray) -> float:
    """Integrate a reported density over phase 0: sum_k h w_k mu_k."""
    phase = solution.phase[0]
    mesh = Mesh(solution.problem._to_spec().phases)
    mesh.set_matrices(solution.problem.spectral_method)
    half_duration = (phase.final_time - phase.initial_time) / 2
    return float(half_duration * (np.asarray(mesh.w[0]) * np.asarray(density)).sum())


def _shift_continuous_row(problem: yapss.Problem, name: str, index: int, eps: float) -> None:
    """Add eps to one row of a continuous-callback output, after the user's callback."""
    original = problem.functions.continuous

    def wrapped(arg):
        original(arg)
        out = getattr(arg.phase[0], name)
        out[index] = out[index] + eps

    problem.functions.continuous = wrapped


def _shift_discrete_row(problem: yapss.Problem, index: int, eps: float) -> None:
    original = problem.functions.discrete

    def wrapped(arg):
        original(arg)
        arg.discrete[index] = arg.discrete[index] + eps

    problem.functions.discrete = wrapped


def _shift_bound(problem: yapss.Problem, which: str, eps: float) -> None:
    phase_bounds = problem.bounds.phase[0]
    if which == "initial_time":
        target, attrs = phase_bounds.initial_time, ("lower", "upper")
    elif which == "duration_upper":
        target, attrs = phase_bounds.duration, ("upper",)
    elif which == "path_upper":
        target, attrs = phase_bounds.path, ("upper",)
    elif which == "parameter_upper":
        target, attrs = problem.bounds.parameter, ("upper",)
    elif which == "discrete":
        target, attrs = problem.bounds.discrete, ("lower", "upper")
    else:  # pragma: no cover - guards the parametrization below
        raise ValueError(which)
    for attr in attrs:
        current = getattr(target, attr)
        if isinstance(current, (list, tuple)) or np.ndim(current) > 0:
            setattr(target, attr, [float(v) + eps for v in current])
        else:
            setattr(target, attr, float(current) + eps)


# (name, how to perturb, what the multipliers predict dJ/deps to be)
#
# Adding eps to a constraint-function row moves the objective by +(the multiplier's
# action on that row); shifting a bound up by eps moves it by -(the multiplier).
SENSITIVITY_KNOBS = [
    (
        "dynamics[0] -> costate[0]",
        lambda p, eps: _shift_continuous_row(p, "dynamics", 0, eps),
        lambda s: _integrate(s, s.phase[0].costate[0]),
    ),
    (
        "dynamics[1] -> costate[1]",
        lambda p, eps: _shift_continuous_row(p, "dynamics", 1, eps),
        lambda s: _integrate(s, s.phase[0].costate[1]),
    ),
    (
        "integrand[0] -> integral multiplier",
        lambda p, eps: _shift_continuous_row(p, "integrand", 0, eps),
        lambda s: float(s.phase[0].integral_multiplier[0]) * s.phase[0].duration,
    ),
    (
        "path[0] -> path multiplier",
        lambda p, eps: _shift_continuous_row(p, "path", 0, eps),
        lambda s: _integrate(s, s.phase[0].path_multiplier[0]),
    ),
    (
        "discrete[0] -> discrete multiplier",
        lambda p, eps: _shift_discrete_row(p, 0, eps),
        lambda s: float(s.discrete_multiplier[0]),
    ),
    (
        "initial time bound",
        lambda p, eps: _shift_bound(p, "initial_time", eps),
        lambda s: -s.phase[0].initial_time_multiplier,
    ),
    (
        "duration upper bound",
        lambda p, eps: _shift_bound(p, "duration_upper", eps),
        lambda s: -s.phase[0].duration_multiplier,
    ),
    (
        "path upper bound",
        lambda p, eps: _shift_bound(p, "path_upper", eps),
        lambda s: -_integrate(s, s.phase[0].path_multiplier[0]),
    ),
    (
        "parameter upper bound",
        lambda p, eps: _shift_bound(p, "parameter_upper", eps),
        lambda s: -float(s.parameter_multiplier[0]),
    ),
    (
        "discrete bound",
        lambda p, eps: _shift_bound(p, "discrete", eps),
        lambda s: -float(s.discrete_multiplier[0]),
    ),
]


@pytest.mark.parametrize("spectral_method", METHODS)
def test_every_multiplier_kind_is_the_objective_sensitivity(spectral_method: str) -> None:
    """Each reported multiplier predicts dJ/d(its own perturbation).

    Independent of the transcription: it compares a reported number with a finite
    difference of the objective over two further solves, and needs no analytic solution.
    """
    base = _kitchen_sink(spectral_method).solve()
    assert base.nlp_info.ipopt_status == 0
    eps = 1e-5
    for name, perturb, predict in SENSITIVITY_KNOBS:
        objectives = []
        for step in (+eps, -eps):
            problem = _kitchen_sink(spectral_method)
            perturb(problem, step)
            solution = problem.solve()
            assert solution.nlp_info.ipopt_status == 0, name
            objectives.append(solution.objective)
        derivative = (objectives[0] - objectives[1]) / (2 * eps)
        predicted = predict(base)
        np.testing.assert_allclose(derivative, predicted, rtol=1e-5, err_msg=name)


@pytest.mark.parametrize("spectral_method", METHODS)
def test_interior_costate_converges_to_the_closed_form(spectral_method: str) -> None:
    """Minimize (1/2) int_0^1 (x^2 + u^2) dt, xdot = u, x(0) = 1, x(1) free.

    The Riccati solution gives lambda(t) = tanh(1 - t) x(t) with
    x(t) = cosh(1 - t) / cosh(1), so the costate has a closed form at every point --
    a check on the costate itself, not merely on a sensitivity identity.
    """
    problem = yapss.Problem(name="LQ", nx=[1], nu=[1], nq=[1])

    def objective(arg):
        arg.objective = arg.phase[0].integral[0]

    def continuous(arg):
        for p in arg.phase_list:
            (x,) = arg.phase[p].state
            (u,) = arg.phase[p].control
            arg.phase[p].dynamics[0] = u
            arg.phase[p].integrand[0] = 0.5 * (x * x + u * u)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower = bounds.initial_state.upper = [1.0]
    problem.guess.phase[0].time = [0.0, 1.0]
    problem.guess.phase[0].state = [[1.0, 1.0]]
    problem.guess.phase[0].control = [[0.0, 0.0]]
    problem.mesh.phase[0].collocation_points = (12,)
    problem.mesh.phase[0].fraction = (1.0,)
    problem.spectral_method = spectral_method
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.tol = 1e-12

    solution = problem.solve()
    assert solution.nlp_info.ipopt_status == 0
    time = solution.phase[0].time_c
    exact = np.tanh(1.0 - time) * (np.cosh(1.0 - time) / np.cosh(1.0))
    np.testing.assert_allclose(solution.phase[0].costate[0], exact, atol=1e-8)


@pytest.mark.parametrize("spectral_method", METHODS)
def test_hamiltonian_needs_no_path_term(spectral_method: str) -> None:
    """The Hamiltonian excludes the path term because that term is zero on-shell.

    Written in the standard form h - bound <= 0, the augmented Hamiltonian is
    lambda.f + nu.g + mu.(h - bound), and complementary slackness makes the last term
    zero along the solution: the multiplier is zero where the constraint is inactive,
    the residual is zero where it is active. The two agree in value, not in derivative.
    On an autonomous problem with a free, interior final time the necessary conditions
    are that the augmented Hamiltonian is constant and that it vanishes at tf, and the
    reported Hamiltonian satisfies both -- with the path constraint active over the
    whole arc.
    """
    upper = 0.6
    problem = yapss.Problem(name="autonomous", nx=[2], nu=[1], nq=[1], nh=[1])

    def objective(arg):
        phase = arg.phase[0]
        arg.objective = phase.integral[0] - 2.0 * phase.final_state[0]

    def continuous(arg):
        for p in arg.phase_list:
            x0, x1 = arg.phase[p].state
            (u,) = arg.phase[p].control
            arg.phase[p].dynamics[0] = x1
            arg.phase[p].dynamics[1] = u
            arg.phase[p].integrand[0] = 0.5 * (u * u + 0.2 * x0 * x0) + 0.2
            arg.phase[p].path[0] = u + 0.3 * x0

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower, bounds.final_time.upper = 0.2, 20.0
    bounds.initial_state.lower = bounds.initial_state.upper = [1.0, 0.0]
    bounds.path.lower, bounds.path.upper = [-1.5], [upper]
    problem.guess.phase[0].time = [0.0, 2.0]
    problem.guess.phase[0].state = [[1.0, 1.0], [0.0, 0.0]]
    problem.guess.phase[0].control = [[0.0, 0.0]]
    problem.mesh.phase[0].collocation_points = (12,)
    problem.mesh.phase[0].fraction = (1.0,)
    problem.spectral_method = spectral_method
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.tol = 1e-12

    solution = problem.solve()
    assert solution.nlp_info.ipopt_status == 0
    phase = solution.phase[0]
    # the final time is free and strictly inside its bounds, and the path constraint is
    # active throughout -- otherwise the two conditions below are vacuous
    assert 0.2 < phase.final_time < 20.0
    assert np.all(np.abs(phase.path_multiplier[0]) > 1e-8)

    hamiltonian = phase.hamiltonian
    assert hamiltonian.max() - hamiltonian.min() < 1e-6
    np.testing.assert_allclose(hamiltonian[-1], 0.0, atol=1e-6)

    # adding mu.(h - bound) changes nothing: complementary slackness zeroes it on-shell
    augmented = hamiltonian + (phase.path_multiplier * (phase.path - upper)).sum(axis=0)
    np.testing.assert_allclose(augmented, hamiltonian, atol=1e-6)


def test_the_sensitivity_identities_are_the_same_under_maximize() -> None:
    """The reported multipliers are the multipliers of the user's problem.

    Under sense="maximize" both the objective and the multipliers change sign, so the
    identities of the test above hold verbatim rather than flipped.
    """
    base = _kitchen_sink(sense="maximize").solve()
    assert base.nlp_info.ipopt_status == 0
    eps = 1e-5
    for name, perturb, predict in SENSITIVITY_KNOBS:
        objectives = []
        for step in (+eps, -eps):
            problem = _kitchen_sink(sense="maximize")
            perturb(problem, step)
            solution = problem.solve()
            assert solution.nlp_info.ipopt_status == 0, name
            objectives.append(solution.objective)
        derivative = (objectives[0] - objectives[1]) / (2 * eps)
        np.testing.assert_allclose(derivative, predict(base), rtol=1e-5, err_msg=name)
