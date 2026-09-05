"""Tests for the shared continuous-function evaluator at the NLP boundary.

The constraint, Jacobian, and Hessian callbacks share one `ContinuousEvaluator`, so
the user's continuous function and its derivatives are computed once per point no
matter which callbacks Ipopt asks for, or in what order. These tests pin two things:
that the values served are the same as a fresh evaluation would give, in every call
order and across cache misses and hits; and that the sharing actually happens, by
counting calls into the user-facing functions.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from yapss._private.auto import make_auto_functions
from yapss._private.bounds import get_nlp_constraint_function_bounds
from yapss._private.central_difference import make_cd_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss.examples import orbit_raising

METHODS = ["auto", "central-difference"]


def _build_nlp(method: str) -> tuple[NLP, np.ndarray]:
    problem = orbit_raising.setup()
    problem.mesh.phase[0].collocation_points = [4]
    problem.mesh.phase[0].fraction = [1.0]
    problem.derivatives.method = method
    problem.derivatives.order = "second"
    problem.spectral_method = "lgr"
    problem.validate()

    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)
    if method == "auto":
        functions = make_auto_functions(problem)
    else:
        functions = make_cd_functions(problem, z0, mesh.tau_u)
    return NLP(problem, functions, mesh), z0


def _points(nlp: NLP, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = len(z0)
    z1 = z0 + 0.01 * np.sin(1.0 + np.arange(nz))
    z2 = z0 + 0.02 * np.cos(2.0 + np.arange(nz))
    constraint_upper, _ = get_nlp_constraint_function_bounds(nlp.problem)
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(len(constraint_upper)))
    return z1, z2, lam


def _fresh(method: str, quantity: str, z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Evaluate one quantity on a brand-new NLP, so nothing can be served from cache."""
    nlp, _ = _build_nlp(method)
    if quantity == "constraints":
        return nlp.constraints(z).copy()
    if quantity == "jacobian":
        return nlp.jacobian(z).copy()
    return nlp.hessian(z, lam, np.float64(1.3)).copy()


def _call(nlp: NLP, quantity: str, z: np.ndarray, lam: np.ndarray) -> np.ndarray:
    if quantity == "constraints":
        return nlp.constraints(z).copy()
    if quantity == "jacobian":
        return nlp.jacobian(z).copy()
    return nlp.hessian(z, lam, np.float64(1.3)).copy()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("order", list(permutations(["constraints", "jacobian", "hessian"])))
def test_every_call_order_matches_fresh_evaluation(method: str, order: tuple[str, ...]) -> None:
    """Cache hits, upgrades, and misses must all give what a fresh NLP gives.

    Each permutation exercises a different cache path: Hessian first is a miss that
    computes everything at once; constraints first followed by the Hessian is an
    upgrade; the return to ``z1`` after ``z2`` is a miss that must not see ``z2``.
    """
    nlp, z0 = _build_nlp(method)
    z1, z2, lam = _points(nlp, z0)

    for z in (z1, z2, z1):
        for quantity in order:
            actual = _call(nlp, quantity, z, lam)
            expected = _fresh(method, quantity, z, lam)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{quantity} at {order}")


@pytest.mark.parametrize("method", METHODS)
def test_repeated_calls_at_one_point_are_stable(method: str) -> None:
    """A second request for any quantity at the cached point returns the same values."""
    nlp, z0 = _build_nlp(method)
    z1, _, lam = _points(nlp, z0)

    for quantity in ("constraints", "jacobian", "hessian"):
        first = _call(nlp, quantity, z1, lam)
        second = _call(nlp, quantity, z1, lam)
        np.testing.assert_array_equal(first, second)


class _Counter:
    def __init__(self, function):  # noqa: ANN001
        self.function = function
        self.calls = 0

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
        self.calls += 1
        return self.function(*args, **kwargs)


def _count(nlp: NLP) -> tuple[_Counter, _Counter, _Counter]:
    functions = nlp.functions
    counters = (
        _Counter(functions.continuous),
        _Counter(functions.continuous_jacobian),
        _Counter(functions.continuous_hessian),
    )
    functions.continuous, functions.continuous_jacobian, functions.continuous_hessian = counters
    return counters


def test_one_iterate_evaluates_each_order_once() -> None:
    """Ipopt's usual g, jac_g, h sequence at one point costs one call per order."""
    nlp, z0 = _build_nlp("auto")
    z1, z2, lam = _points(nlp, z0)
    continuous, jacobian, hessian = _count(nlp)

    nlp.constraints(z1)
    nlp.jacobian(z1)
    nlp.hessian(z1, lam, np.float64(1.0))
    assert (continuous.calls, jacobian.calls, hessian.calls) == (1, 1, 1)

    # a line-search trial point asks for the function values only
    nlp.constraints(z2)
    assert (continuous.calls, jacobian.calls, hessian.calls) == (2, 1, 1)

    # returning to the accepted point is a miss: the trial point displaced it
    nlp.hessian(z1, lam, np.float64(1.0))
    assert (continuous.calls, jacobian.calls, hessian.calls) == (3, 2, 2)


def test_hessian_first_computes_lower_orders_once() -> None:
    """A Hessian request at a new point evaluates the function and Jacobian exactly once."""
    nlp, z0 = _build_nlp("auto")
    z1, _, lam = _points(nlp, z0)
    continuous, jacobian, hessian = _count(nlp)

    nlp.hessian(z1, lam, np.float64(1.0))
    nlp.jacobian(z1)
    nlp.constraints(z1)
    assert (continuous.calls, jacobian.calls, hessian.calls) == (1, 1, 1)


def test_cache_key_is_the_value_not_the_buffer() -> None:
    """The same buffer with new contents must miss; a different buffer with the same contents must hit."""
    nlp, z0 = _build_nlp("auto")
    z1, z2, _ = _points(nlp, z0)
    continuous, _, _ = _count(nlp)

    buffer = z1.copy()
    nlp.constraints(buffer)
    nlp.constraints(z1.copy())
    assert continuous.calls == 1

    buffer[:] = z2
    nlp.constraints(buffer)
    assert continuous.calls == 2


def test_failed_evaluation_is_not_cached() -> None:
    """An exception in a user callback must not leave its order marked as computed."""
    nlp, z0 = _build_nlp("auto")
    z1, _, _ = _points(nlp, z0)
    continuous, jacobian, _ = _count(nlp)

    original = jacobian.function
    fail_once = {"armed": True}

    def flaky(arg):  # noqa: ANN001, ANN202
        if fail_once["armed"]:
            fail_once["armed"] = False
            msg = "user callback failed"
            raise RuntimeError(msg)
        return original(arg)

    nlp.functions.continuous_jacobian = _Counter(flaky)

    with pytest.raises(RuntimeError, match="user callback failed"):
        nlp.jacobian(z1)
    assert continuous.calls == 1

    expected = _fresh("auto", "jacobian", z1, np.zeros(0))
    np.testing.assert_array_equal(nlp.jacobian(z1), expected)
    # the function values were computed once and kept; only the Jacobian was retried
    assert continuous.calls == 1
