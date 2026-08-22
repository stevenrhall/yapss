"""

Golden regression tests for the NLP derivative assembly.

The NLP Hessian is assembled by two functions that must agree positionally: the
structure builder emits (row, col) index pairs, and the evaluator writes values at a
manually-advanced cursor, iterating the same term lists in the same order. A mismatch
does not crash -- the downstream sparse fold sums the wrong entries -- so refactoring
this code needs a net that catches *any* behavioral change, not just gross ones.

These tests pin the exact outputs of all five NLP callbacks (objective, gradient,
constraints, jacobian, hessian) and both structure functions, for a matrix of problems
chosen so that together they exercise every term kind the assembler handles:

===========================  =======================================================
problem                      exercises
===========================  =======================================================
brachistochrone              defect ("f") terms only, single phase
orbit_raising                time-varying dynamics (the only example with any): t,t
                             and t,u Hessian terms from a real problem
isoperimetric                integral ("g") terms, path ("h") terms, discrete
dynamic_soaring              parameter ("s") variable terms, path, discrete
goddard_problem_3_phase      multiple phases, phase-linkage discrete constraints
hs071                        np == 0: objective/discrete Hessian path alone
===========================  =======================================================

crossed with all three spectral methods and both derivative backends. Evaluation
points are deterministic (no RNG -- numpy does not guarantee Generator streams across
versions).

To regenerate the golden file after an *intentional* change in behavior:

    python tests/modules/test_nlp_hessian_golden.py

which adds any missing cases while keeping the existing pinned values (pass ``--all``
to overwrite everything, only for an intentional behavior change). Never regenerate to
make a refactor pass: a refactor must reproduce these values.

``test_hessian_matches_finite_difference`` is different in kind: it checks the current
assembly against finite differences of the gradient and constraint Jacobian, so it
validates that the pinned values are *right*, not merely unchanged.

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yapss._private.auto import make_auto_functions
from yapss._private.central_difference import make_cd_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss._private.user import make_user_functions
from yapss.examples import (
    brachistochrone,
    dynamic_soaring,
    goddard_problem_3_phase,
    hs071,
    isoperimetric,
    orbit_raising,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "nlp_golden.json"


def kitchen_sink_setup():
    """A deliberately non-autonomous problem exercising every assembler arm.

    Every example problem is autonomous, so none of them reaches the time-derivative
    branches of the Hessian assembly (the ``t,t``, ``{x|u|s},t``, and chain-rule
    Jacobian ``t`` terms). This problem makes the dynamics, integrands, and path
    constraint depend explicitly on time, crossed with every variable kind, and its
    objective and discrete constraints touch every discrete variable. The coverage
    check in ``test_golden_cases_reach_every_assembler_branch`` fails if an arm of the
    assembler stops being exercised.
    """
    import yapss

    problem = yapss.Problem(name="kitchen_sink", nx=[2], nu=[1], nq=[2], nh=[1], ns=2, nd=2)

    def objective(arg):
        phase = arg.phase[0]
        s = arg.parameter
        arg.objective = (
            phase.final_time * phase.integral[0]
            + phase.initial_time**2 * s[0]
            + phase.final_state[0] * phase.initial_state[1]
            + s[1] * phase.integral[1] ** 2
        )

    def continuous(arg):
        s = arg.parameter
        for p in arg.phase_list:
            x = arg.phase[p].state
            (u,) = arg.phase[p].control
            t = arg.phase[p].time
            arg.phase[p].dynamics[:] = [
                x[0] * x[1] * u + t**2 * x[0] + s[0] * t * u,
                t**3 + s[1] * x[1] ** 2 + u**2 * t,
            ]
            arg.phase[p].integrand[:] = [
                t**2 * x[0] * u + s[0] * s[1] * t,
                x[1] * u * t + s[1] ** 2,
            ]
            arg.phase[p].path[:] = [t**2 * u * s[0] + x[0] ** 2 * t + x[1] * s[1]]

    def discrete(arg):
        phase = arg.phase[0]
        s = arg.parameter
        arg.discrete[:] = [
            phase.final_time**2 * s[0] + phase.initial_state[0] * phase.integral[1],
            phase.initial_time * phase.final_state[1] + s[1] ** 2 * phase.integral[0],
        ]

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    problem.functions.discrete = discrete

    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower, bounds.initial_time.upper = 0.0, 0.4
    bounds.final_time.lower, bounds.final_time.upper = 1.0, 2.0
    bounds.state.lower[:], bounds.state.upper[:] = -5.0, 5.0
    bounds.control.lower[:], bounds.control.upper[:] = -2.0, 2.0
    bounds.path.lower[:], bounds.path.upper[:] = -50.0, 50.0
    problem.bounds.parameter.lower[:], problem.bounds.parameter.upper[:] = -3.0, 3.0
    problem.bounds.discrete.lower[:], problem.bounds.discrete.upper[:] = -50.0, 50.0

    guess = problem.guess.phase[0]
    guess.time = [0.2, 1.5]
    guess.state = [[1.0, 2.0], [-1.0, 1.0]]
    guess.control = [[0.5, -0.5]]
    guess.integral = [1.0, 2.0]
    problem.guess.parameter = [1.5, -0.7]
    return problem


def kitchen_sink_user_derivatives(problem):
    """Attach hand-derived derivative callbacks for the "user" method.

    Each Hessian pair is supplied once, per the documented convention. Continuous
    entries are broadcast over the time grid because constant f-term entries must be
    arrays. The user-vs-auto agreement test validates this algebra against CasADi
    before the values are pinned.
    """

    def objective_gradient(arg):
        phase = arg.phase[0]
        s = arg.parameter
        gradient = arg.gradient
        gradient[0, "tf", 0] = phase.integral[0]
        gradient[0, "q", 0] = phase.final_time
        gradient[0, "t0", 0] = 2.0 * phase.initial_time * s[0]
        gradient[0, "s", 0] = phase.initial_time**2
        gradient[0, "xf", 0] = phase.initial_state[1]
        gradient[0, "x0", 1] = phase.final_state[0]
        gradient[0, "s", 1] = phase.integral[1] ** 2
        gradient[0, "q", 1] = 2.0 * s[1] * phase.integral[1]

    def objective_hessian(arg):
        phase = arg.phase[0]
        s = arg.parameter
        hessian = arg.hessian
        hessian[(0, "tf", 0), (0, "q", 0)] = 1.0
        hessian[(0, "t0", 0), (0, "t0", 0)] = 2.0 * s[0]
        hessian[(0, "t0", 0), (0, "s", 0)] = 2.0 * phase.initial_time
        hessian[(0, "xf", 0), (0, "x0", 1)] = 1.0
        hessian[(0, "s", 1), (0, "q", 1)] = 2.0 * phase.integral[1]
        hessian[(0, "q", 1), (0, "q", 1)] = 2.0 * s[1]

    def continuous_jacobian(arg):
        s = arg.parameter
        for p in arg.phase_list:
            x = arg.phase[p].state
            (u,) = arg.phase[p].control
            t = arg.phase[p].time
            one = 0.0 * t + 1.0
            jacobian = arg.phase[p].jacobian
            # f0 = x0*x1*u + t**2*x0 + s0*t*u
            jacobian[("f", 0), ("x", 0)] = x[1] * u + t**2
            jacobian[("f", 0), ("x", 1)] = x[0] * u
            jacobian[("f", 0), ("u", 0)] = x[0] * x[1] + s[0] * t
            jacobian[("f", 0), ("t", 0)] = 2.0 * t * x[0] + s[0] * u
            jacobian[("f", 0), ("s", 0)] = t * u
            # f1 = t**3 + s1*x1**2 + u**2*t
            jacobian[("f", 1), ("x", 1)] = 2.0 * s[1] * x[1]
            jacobian[("f", 1), ("u", 0)] = 2.0 * u * t
            jacobian[("f", 1), ("t", 0)] = 3.0 * t**2 + u**2
            jacobian[("f", 1), ("s", 1)] = x[1] ** 2
            # g0 = t**2*x0*u + s0*s1*t
            jacobian[("g", 0), ("x", 0)] = t**2 * u
            jacobian[("g", 0), ("u", 0)] = t**2 * x[0]
            jacobian[("g", 0), ("t", 0)] = 2.0 * t * x[0] * u + s[0] * s[1]
            jacobian[("g", 0), ("s", 0)] = s[1] * t
            jacobian[("g", 0), ("s", 1)] = s[0] * t
            # g1 = x1*u*t + s1**2
            jacobian[("g", 1), ("x", 1)] = u * t
            jacobian[("g", 1), ("u", 0)] = x[1] * t
            jacobian[("g", 1), ("t", 0)] = x[1] * u
            jacobian[("g", 1), ("s", 1)] = 2.0 * s[1] * one
            # h0 = t**2*u*s0 + x0**2*t + x1*s1
            jacobian[("h", 0), ("x", 0)] = 2.0 * x[0] * t
            jacobian[("h", 0), ("x", 1)] = s[1] * one
            jacobian[("h", 0), ("u", 0)] = t**2 * s[0]
            jacobian[("h", 0), ("t", 0)] = 2.0 * t * u * s[0] + x[0] ** 2
            jacobian[("h", 0), ("s", 0)] = t**2 * u
            jacobian[("h", 0), ("s", 1)] = x[1] * one

    def continuous_hessian(arg):
        s = arg.parameter
        for p in arg.phase_list:
            x = arg.phase[p].state
            (u,) = arg.phase[p].control
            t = arg.phase[p].time
            one = 0.0 * t + 1.0
            hessian = arg.phase[p].hessian
            # f0 = x0*x1*u + t**2*x0 + s0*t*u
            hessian[("f", 0), ("x", 0), ("x", 1)] = u * one
            hessian[("f", 0), ("x", 0), ("u", 0)] = x[1] * one
            hessian[("f", 0), ("x", 1), ("u", 0)] = x[0] * one
            hessian[("f", 0), ("x", 0), ("t", 0)] = 2.0 * t
            hessian[("f", 0), ("t", 0), ("t", 0)] = 2.0 * x[0]
            hessian[("f", 0), ("s", 0), ("t", 0)] = u * one
            hessian[("f", 0), ("s", 0), ("u", 0)] = t
            hessian[("f", 0), ("t", 0), ("u", 0)] = s[0] * one
            # f1 = t**3 + s1*x1**2 + u**2*t
            hessian[("f", 1), ("t", 0), ("t", 0)] = 6.0 * t
            hessian[("f", 1), ("x", 1), ("x", 1)] = 2.0 * s[1] * one
            hessian[("f", 1), ("s", 1), ("x", 1)] = 2.0 * x[1]
            hessian[("f", 1), ("u", 0), ("u", 0)] = 2.0 * t
            hessian[("f", 1), ("u", 0), ("t", 0)] = 2.0 * u
            # g0 = t**2*x0*u + s0*s1*t
            hessian[("g", 0), ("x", 0), ("u", 0)] = t**2
            hessian[("g", 0), ("x", 0), ("t", 0)] = 2.0 * t * u
            hessian[("g", 0), ("u", 0), ("t", 0)] = 2.0 * t * x[0]
            hessian[("g", 0), ("t", 0), ("t", 0)] = 2.0 * x[0] * u
            hessian[("g", 0), ("s", 0), ("s", 1)] = t
            hessian[("g", 0), ("s", 0), ("t", 0)] = s[1] * one
            hessian[("g", 0), ("s", 1), ("t", 0)] = s[0] * one
            # g1 = x1*u*t + s1**2
            hessian[("g", 1), ("x", 1), ("u", 0)] = t
            hessian[("g", 1), ("x", 1), ("t", 0)] = u * one
            hessian[("g", 1), ("u", 0), ("t", 0)] = x[1] * one
            hessian[("g", 1), ("s", 1), ("s", 1)] = 2.0 * one
            # h0 = t**2*u*s0 + x0**2*t + x1*s1
            hessian[("h", 0), ("t", 0), ("t", 0)] = 2.0 * u * s[0]
            hessian[("h", 0), ("t", 0), ("u", 0)] = 2.0 * t * s[0]
            hessian[("h", 0), ("s", 0), ("t", 0)] = 2.0 * t * u
            hessian[("h", 0), ("s", 0), ("u", 0)] = t**2
            hessian[("h", 0), ("x", 0), ("x", 0)] = 2.0 * t
            hessian[("h", 0), ("x", 0), ("t", 0)] = 2.0 * x[0]
            hessian[("h", 0), ("x", 1), ("s", 1)] = one

    def discrete_jacobian(arg):
        phase = arg.phase[0]
        s = arg.parameter
        jacobian = arg.jacobian
        # d0 = tf**2*s0 + x0_0*q1
        jacobian[0, (0, "tf", 0)] = 2.0 * phase.final_time * s[0]
        jacobian[0, (0, "s", 0)] = phase.final_time**2
        jacobian[0, (0, "x0", 0)] = phase.integral[1]
        jacobian[0, (0, "q", 1)] = phase.initial_state[0]
        # d1 = t0*xf_1 + s1**2*q0
        jacobian[1, (0, "t0", 0)] = phase.final_state[1]
        jacobian[1, (0, "xf", 1)] = phase.initial_time
        jacobian[1, (0, "s", 1)] = 2.0 * s[1] * phase.integral[0]
        jacobian[1, (0, "q", 0)] = s[1] ** 2

    def discrete_hessian(arg):
        phase = arg.phase[0]
        s = arg.parameter
        hessian = arg.hessian
        hessian[0, (0, "tf", 0), (0, "tf", 0)] = 2.0 * s[0]
        hessian[0, (0, "tf", 0), (0, "s", 0)] = 2.0 * phase.final_time
        hessian[0, (0, "x0", 0), (0, "q", 1)] = 1.0
        hessian[1, (0, "t0", 0), (0, "xf", 1)] = 1.0
        hessian[1, (0, "s", 1), (0, "s", 1)] = 2.0 * phase.integral[0]
        hessian[1, (0, "s", 1), (0, "q", 0)] = 2.0 * s[1]

    functions = problem.functions
    functions.objective_gradient = objective_gradient
    functions.objective_hessian = objective_hessian
    functions.continuous_jacobian = continuous_jacobian
    functions.continuous_hessian = continuous_hessian
    functions.discrete_jacobian = discrete_jacobian
    functions.discrete_hessian = discrete_hessian


class _KitchenSinkModule:
    setup = staticmethod(kitchen_sink_setup)


# collocation points per phase; at least one problem uses two mesh intervals, since the
# LGL noncollocated-point count (nc - K + 1) depends on the interval count K
MESHES = {
    "kitchen_sink": [[3, 4]],
    "orbit_raising": [[4]],
    "brachistochrone": [[3, 4]],
    "isoperimetric": [[3, 4]],
    "dynamic_soaring": [[4]],
    "goddard_problem_3_phase": [[3], [3], [3]],
    "hs071": [],
}

PROBLEMS = {
    "kitchen_sink": _KitchenSinkModule,
    "orbit_raising": orbit_raising,
    "brachistochrone": brachistochrone,
    "isoperimetric": isoperimetric,
    "dynamic_soaring": dynamic_soaring,
    "goddard_problem_3_phase": goddard_problem_3_phase,
    "hs071": hs071,
}

SPECTRAL_METHODS = ("lg", "lgr", "lgl")
DERIVATIVE_METHODS = ("auto", "central-difference")

# problems whose "user"-method derivative callbacks exist: the examples define their
# own, and the kitchen sink's are attached by kitchen_sink_user_derivatives
USER_METHOD_PROBLEMS = ("kitchen_sink", "brachistochrone", "goddard_problem_3_phase")

# the "user" cases are appended after the original matrix so that regenerating the
# golden file extends it without rewriting the existing entries
CASES = [
    (name, spectral, method)
    for name in PROBLEMS
    for spectral in (SPECTRAL_METHODS if MESHES[name] else ("lgl",))
    for method in DERIVATIVE_METHODS
] + [(name, spectral, "user") for name in USER_METHOD_PROBLEMS for spectral in SPECTRAL_METHODS]


def build_nlp(name: str, spectral_method: str, derivative_method: str) -> tuple[NLP, dict]:
    """Build the NLP for one case, with deterministic evaluation inputs."""
    problem = PROBLEMS[name].setup()
    if name == "kitchen_sink" and derivative_method == "user":
        kitchen_sink_user_derivatives(problem)
    for p, points in enumerate(MESHES[name]):
        problem.mesh.phase[p].collocation_points = points
        problem.mesh.phase[p].fraction = [1.0 / len(points)] * len(points)
    problem.spectral_method = spectral_method
    problem.derivatives.method = derivative_method
    problem.derivatives.order = "second"
    problem.validate()

    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)
    if derivative_method == "auto":
        functions = make_auto_functions(problem)
    elif derivative_method == "user":
        functions = make_user_functions(problem, z0, mesh.tau_u)
    else:
        functions = make_cd_functions(problem, z0, mesh.tau_u)
    nlp = NLP(problem, functions, mesh)

    # deterministic, structure-free perturbation off the guess so that no term
    # evaluates at a symmetric or zero point that could mask an assembly error
    nz = len(z0)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(nz))
    nc = len(nlp.constraints(z))
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(nc))
    inputs = {"z": z, "lam": lam, "objective_factor": np.float64(1.3)}
    return nlp, inputs


def evaluate_case(name: str, spectral_method: str, derivative_method: str) -> dict:
    """Return every pinned quantity for one case, as plain lists."""
    nlp, inputs = build_nlp(name, spectral_method, derivative_method)
    z, lam = inputs["z"], inputs["lam"]
    jrow, jcol = nlp.jacobianstructure()
    hrow, hcol = nlp.hessianstructure()
    return {
        "objective": float(nlp.objective(z)),
        "gradient": np.asarray(nlp.gradient(z)).tolist(),
        "constraints": np.asarray(nlp.constraints(z)).tolist(),
        "jacobian_structure": [list(map(int, jrow)), list(map(int, jcol))],
        "jacobian": np.asarray(nlp.jacobian(z)).tolist(),
        "hessian_structure": [list(map(int, hrow)), list(map(int, hcol))],
        "hessian": np.asarray(nlp.hessian(z, lam, inputs["objective_factor"])).tolist(),
    }


def case_key(name: str, spectral_method: str, derivative_method: str) -> str:
    return f"{name}/{spectral_method}/{derivative_method}"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():  # pragma: no cover
        pytest.fail(f"golden file missing; generate it with: python {__file__}")
    return json.loads(GOLDEN_PATH.read_text())


# Tolerances for comparison against the pinned values, per (field kind, method).
#
# Exact derivatives ("auto", "user") are pinned tightly: across numpy versions and
# platforms they differ only by the last bit of the callback evaluations.
#
# Central-difference values cannot be pinned that tightly. A finite difference
# amplifies the ulp-level evaluation differences between platforms and numpy versions
# (different libm and SIMD kernels) by the reciprocal of the step, so the pinned
# values are one platform's roundoff realization and another platform's differs by
# up to the intrinsic finite-difference error. That error, measured against the
# exact derivatives and normalized by the field's magnitude, is up to 3e-7 for the
# Hessian (dynamic soaring, whose function values are large relative to its Hessian)
# and 3e-10 for the Jacobian across the golden problems. The tolerances below sit an
# order of magnitude above those measurements. A CI run on Linux/x86 against values
# generated on macOS/arm64 differed by 1.9e-7 on exactly the predicted entry.
#
# All comparisons are normalized by the pinned field's magnitude; every assembly
# mutation this suite was validated against -- down to a 0.04% scale error -- fails
# these by orders of magnitude.
TOLERANCES = {
    ("value", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("value", "user"): {"rtol": 1e-10, "atol": 1e-12},
    ("value", "central-difference"): {"rtol": 1e-10, "atol": 1e-12},
    ("first", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("first", "user"): {"rtol": 1e-10, "atol": 1e-12},
    ("first", "central-difference"): {"rtol": 1e-7, "atol": 1e-8},
    ("second", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("second", "user"): {"rtol": 1e-10, "atol": 1e-12},
    ("second", "central-difference"): {"rtol": 1e-5, "atol": 3e-6},
}
FIELD_KIND = {
    "objective": "value",
    "constraints": "value",
    "gradient": "first",
    "jacobian": "first",
    "hessian": "second",
}


@pytest.mark.parametrize(("name", "spectral_method", "derivative_method"), CASES)
def test_nlp_callbacks_match_golden(name, spectral_method, derivative_method, golden):
    """All NLP callback outputs must be unchanged from the pinned values.

    Structures are compared exactly; values per the tolerance table above.
    """
    expected = golden[case_key(name, spectral_method, derivative_method)]
    actual = evaluate_case(name, spectral_method, derivative_method)

    assert actual["jacobian_structure"] == expected["jacobian_structure"]
    assert actual["hessian_structure"] == expected["hessian_structure"]
    for field in ("objective", "gradient", "constraints", "jacobian", "hessian"):
        expected_values = np.asarray(expected[field])
        scale = max(1.0, np.max(np.abs(expected_values)) if expected_values.size else 1.0)
        np.testing.assert_allclose(
            np.asarray(actual[field]) / scale,
            expected_values / scale,
            **TOLERANCES[FIELD_KIND[field], derivative_method],
            err_msg=f"{field} changed for {case_key(name, spectral_method, derivative_method)}",
        )


def fold_jacobian_dense(nlp: NLP, values: np.ndarray, nz: int) -> np.ndarray:
    """Reconstruct the dense Jacobian from the sparse triple, summing duplicates."""
    row, col = nlp.jacobianstructure()
    dense = np.zeros((max(row) + 1 if row else 0, nz))
    for r, c, v in zip(row, col, values, strict=True):
        dense[r, c] += v
    return dense


def fold_hessian_dense(nlp: NLP, values: np.ndarray, nz: int) -> np.ndarray:
    """Reconstruct the dense symmetric Hessian from the sparse triple.

    The raw structure may contain duplicate (row, col) entries and entries in either
    triangle; Ipopt receives it as one triangle of a symmetric matrix after
    ``simplify_hessian`` folds it. Reproduce that fold: accumulate each entry into
    canonical (max, min) order, then mirror.
    """
    row, col = nlp.hessianstructure()
    lower = np.zeros((nz, nz))
    for r, c, v in zip(row, col, values, strict=True):
        i, j = (r, c) if r >= c else (c, r)
        lower[i, j] += v
    return lower + lower.T - np.diag(np.diag(lower))


@pytest.mark.parametrize(
    ("name", "spectral_method"),
    [(n, s) for n in PROBLEMS for s in (SPECTRAL_METHODS if MESHES[n] else ("lgl",))],
)
def test_hessian_matches_finite_difference(name, spectral_method):
    """The assembled Hessian must equal the derivative of the assembled gradient.

    This is the independent check the golden values rest on: it compares the Hessian of
    the Lagrangian against central differences of ``sigma * gradient(z) + J(z)^T lam``
    at the NLP boundary, using the "auto" backend so the first derivatives are exact.
    It validates the assembly itself -- structure/value correspondence included --
    against the objective and constraints, not against a previous version of the code.
    """
    nlp, inputs = build_nlp(name, spectral_method, "auto")
    z, lam = inputs["z"], inputs["lam"]
    sigma = float(inputs["objective_factor"])
    nz = len(z)
    jrow, jcol = nlp.jacobianstructure()

    def lagrangian_gradient(point: np.ndarray) -> np.ndarray:
        grad = sigma * np.asarray(nlp.gradient(point), dtype=float)
        jac = np.asarray(nlp.jacobian(point), dtype=float)
        for r, c, v in zip(jrow, jcol, jac, strict=True):
            grad[c] += lam[r] * v
        return grad

    step = 1e-6
    fd = np.zeros((nz, nz))
    for k in range(nz):
        zp, zm = z.copy(), z.copy()
        zp[k] += step
        zm[k] -= step
        fd[:, k] = (lagrangian_gradient(zp) - lagrangian_gradient(zm)) / (2 * step)

    assembled = fold_hessian_dense(
        nlp,
        np.asarray(nlp.hessian(z, lam, inputs["objective_factor"]), dtype=float),
        nz,
    )
    scale = max(1.0, np.abs(fd).max())
    np.testing.assert_allclose(assembled / scale, (fd + fd.T) / 2 / scale, atol=5e-7)


def main(regenerate_all: bool = False) -> None:
    """Add missing cases to the golden file; keep the existing pinned values.

    Existing entries are preserved deliberately: their authority comes from having
    been generated by the implementation that predates the assembly refactors, and
    regenerating them would replace that provenance with the current code's output
    (bit-identical up to summation order, but no longer independent). Pass ``--all``
    to overwrite everything -- only for an intentional behavior change, committed
    together with the change that explains it.
    """
    data: dict = {}
    if GOLDEN_PATH.exists() and not regenerate_all:
        data = json.loads(GOLDEN_PATH.read_text())
    added = 0
    for case in CASES:
        key = case_key(*case)
        if key not in data:
            data[key] = evaluate_case(*case)
            added += 1
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(data, indent=1))
    print(f"wrote {GOLDEN_PATH} ({len(data)} cases, {added} added)")


if __name__ == "__main__":
    import sys

    main(regenerate_all="--all" in sys.argv)


FD_CASES = [(n, s) for n in PROBLEMS for s in (SPECTRAL_METHODS if MESHES[n] else ("lgl",))]


@pytest.mark.parametrize(("name", "spectral_method"), FD_CASES)
def test_central_difference_full_matches_central_difference(name, spectral_method):
    """The "-full" variant must produce the same derivatives as sparsity-probed CD.

    "central-difference-full" differs from "central-difference" only in the
    first-derivative structure builders: it assumes every function depends on every
    variable rather than probing sparsity at the initial guess. Everything downstream
    is shared code fed a denser structure, so rather than pinning a second set of
    near-identical golden values, this checks the two variants against each other:

    * the probed structure must be a subset of the full structure (a full builder
      that drops a variable class fails here), and
    * the dense Jacobian and Hessian must agree to the finite-difference noise
      floor. The extra entries of the full structure are central differences of
      structurally-zero derivatives, which cancel only to roundoff -- about
      eps/DELTA2**2 ~ 1e-9 of the function magnitude for the Hessian -- and fold
      into coordinates shared with real terms, so agreement is not bitwise.

    This is the only coverage the "-full" structure builders have; the golden matrix
    deliberately excludes the method.
    """
    nlp_cd, inputs = build_nlp(name, spectral_method, "central-difference")
    nlp_full, _ = build_nlp(name, spectral_method, "central-difference-full")
    z, lam = inputs["z"], inputs["lam"]
    nz = len(z)

    # structure containment, as coordinate sets
    for structure_of in (NLP.jacobianstructure, NLP.hessianstructure):
        cd_pairs = set(zip(*structure_of(nlp_cd), strict=True))
        full_pairs = set(zip(*structure_of(nlp_full), strict=True))
        missing = cd_pairs - full_pairs
        assert not missing, (
            f"{structure_of.__name__}: full structure is missing {len(missing)} "
            f"coordinate(s) present in the probed structure: {sorted(missing)[:5]}"
        )

    # dense value agreement, normalized so the absolute tolerance tracks the
    # problem's magnitude; a dropped or mis-scaled term fails by many orders
    jac_cd = fold_jacobian_dense(nlp_cd, np.asarray(nlp_cd.jacobian(z)), nz)
    jac_full = fold_jacobian_dense(nlp_full, np.asarray(nlp_full.jacobian(z)), nz)
    jac_scale = max(1.0, np.abs(jac_cd).max())
    np.testing.assert_allclose(jac_full / jac_scale, jac_cd / jac_scale, rtol=1e-8, atol=1e-9)

    sigma = inputs["objective_factor"]
    hess_cd = fold_hessian_dense(nlp_cd, np.asarray(nlp_cd.hessian(z, lam, sigma)), nz)
    hess_full = fold_hessian_dense(nlp_full, np.asarray(nlp_full.hessian(z, lam, sigma)), nz)
    hess_scale = max(1.0, np.abs(hess_cd).max())
    np.testing.assert_allclose(
        hess_full / hess_scale,
        hess_cd / hess_scale,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("name", "spectral_method"),
    [(n, sp) for n in USER_METHOD_PROBLEMS for sp in SPECTRAL_METHODS],
)
def test_user_derivatives_match_auto(name, spectral_method):
    """User-supplied analytic derivatives must agree with automatic differentiation.

    Both paths are exact, so the dense Jacobian and Hessian must match to roundoff.
    This validates the hand-derived callbacks -- including the kitchen sink's, written
    for this suite -- independently of the pinned values, and it is the only check
    that the "user" structure deduction handles time-derivative terms: every example
    problem with user derivatives is autonomous.
    """
    nlp_user, inputs = build_nlp(name, spectral_method, "user")
    nlp_auto, _ = build_nlp(name, spectral_method, "auto")
    z, lam = inputs["z"], inputs["lam"]
    nz = len(z)

    jac_user = fold_jacobian_dense(nlp_user, np.asarray(nlp_user.jacobian(z)), nz)
    jac_auto = fold_jacobian_dense(nlp_auto, np.asarray(nlp_auto.jacobian(z)), nz)
    scale = max(1.0, np.abs(jac_auto).max())
    np.testing.assert_allclose(jac_user / scale, jac_auto / scale, rtol=1e-11, atol=1e-12)

    sigma = inputs["objective_factor"]
    hess_user = fold_hessian_dense(nlp_user, np.asarray(nlp_user.hessian(z, lam, sigma)), nz)
    hess_auto = fold_hessian_dense(nlp_auto, np.asarray(nlp_auto.hessian(z, lam, sigma)), nz)
    scale = max(1.0, np.abs(hess_auto).max())
    np.testing.assert_allclose(hess_user / scale, hess_auto / scale, rtol=1e-11, atol=1e-12)
