"""Regression tests for derivative assembly at the flat NLP boundary."""

from __future__ import annotations

import numpy as np
import pytest

from yapss._private.auto import make_auto_functions
from yapss._private.bounds import get_nlp_constraint_function_bounds
from yapss._private.central_difference import make_cd_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss._private.structure import get_nlp_cf_structure
from yapss.examples import delta_iii_ascent, orbit_raising


def _make_orbit_raising_nlp(method: str) -> tuple[NLP, np.ndarray]:
    problem = orbit_raising.setup()
    problem.mesh.phase[0].collocation_points = [3]
    problem.mesh.phase[0].fraction = [1.0]
    problem.derivatives.method = method
    problem.derivatives.order = "second"
    problem.spectral_method = "lgl"
    problem.validate()

    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    x = make_initial_guess_nlp(problem, mesh)
    if method == "auto":
        functions = make_auto_functions(problem)
    else:
        functions = make_cd_functions(problem, x, mesh.tau_u)
    return NLP(problem, functions, mesh), x


def _continuous_values(nlp: NLP, x: np.ndarray, order: int) -> tuple[np.ndarray, ...]:
    continuous = nlp.eval_continuous(x, order)
    return tuple(
        np.concatenate(
            (
                phase.dynamics.ravel(),
                phase.integrand.ravel(),
                phase.path.ravel(),
            ),
        ).copy()
        for phase in continuous.phase
    )


@pytest.mark.parametrize("order", [1, 2])
def test_central_difference_derivatives_restore_continuous_values(order: int) -> None:
    """Derivative scratch evaluations must not leak into rolled-up chain-rule terms."""
    nlp, x = _make_orbit_raising_nlp("central-difference")

    after_derivatives = _continuous_values(nlp, x, order)
    unperturbed = _continuous_values(nlp, x, 0)

    for actual, expected in zip(after_derivatives, unperturbed, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_first_hessian_call_uses_current_phase_times() -> None:
    """Repeated Hessian calls at one point must not depend on callback history."""
    nlp, x = _make_orbit_raising_nlp("auto")
    constraint_upper, _ = get_nlp_constraint_function_bounds(nlp.problem)
    multipliers = 1.0 + np.arange(len(constraint_upper), dtype=np.float64) % 5 / 5

    first = nlp.hessian(x, multipliers, np.float64(1.0)).copy()
    second = nlp.hessian(x, multipliers, np.float64(1.0)).copy()

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("spectral_method", ["lg", "lgr"])
def test_lg_lgr_do_not_allocate_zero_mode_constraints(spectral_method: str) -> None:
    """Do not pass the inactive research zero-mode rows to Ipopt."""
    problem = delta_iii_ascent.setup()
    problem.spectral_method = spectral_method

    structure = get_nlp_cf_structure(problem, np.float64)
    assert all(not hasattr(phase, "zero_mode") for phase in structure.phase)

    constraint_upper, constraint_lower = get_nlp_constraint_function_bounds(problem)
    free = np.isneginf(constraint_lower) & np.isposinf(constraint_upper)
    assert not np.any(free)
