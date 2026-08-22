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

and commit the resulting ``data/nlp_golden.json`` along with the change that explains
it. Never regenerate to make a refactor pass: a refactor must reproduce these values.

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

CASES = [
    (name, spectral, method)
    for name in PROBLEMS
    for spectral in (SPECTRAL_METHODS if MESHES[name] else ("lgl",))
    for method in DERIVATIVE_METHODS
]


def build_nlp(name: str, spectral_method: str, derivative_method: str) -> tuple[NLP, dict]:
    """Build the NLP for one case, with deterministic evaluation inputs."""
    problem = PROBLEMS[name].setup()
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
# The golden data was generated on one numpy version, and numpy releases change SIMD
# kernels for the transcendental functions, so callback evaluations differ across
# versions at the last bit. Exact ("auto") derivatives inherit only that ulp-level
# difference; finite differences amplify it by 1/step -- about eps/DELTA1 ~ 3e-11 of
# the function magnitude for first derivatives and eps/DELTA2**2 ~ 1e-9 for second
# derivatives -- so the central-difference fields carry the noise floor with margin.
# All comparisons are normalized by the pinned field's magnitude; every assembly
# mutation this suite was validated against fails these by many orders of magnitude.
TOLERANCES = {
    ("value", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("value", "central-difference"): {"rtol": 1e-10, "atol": 1e-12},
    ("first", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("first", "central-difference"): {"rtol": 1e-8, "atol": 1e-9},
    ("second", "auto"): {"rtol": 1e-10, "atol": 1e-12},
    ("second", "central-difference"): {"rtol": 1e-6, "atol": 1e-7},
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


def main() -> None:
    """Regenerate the golden file from the current implementation."""
    data = {case_key(*case): evaluate_case(*case) for case in CASES}
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(data, indent=1))
    print(f"wrote {GOLDEN_PATH} ({len(data)} cases)")


if __name__ == "__main__":
    main()


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
