"""Test `yapss._standin` as a plain CasADi user would: no Problem, no SXW, no callback.

The import-boundary test is the property that makes the package liftable: nothing in it
imports from YAPSS outside itself.
"""

import ast
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

import yapss._standin
from yapss._standin import Function, external, tracing

PACKAGE = Path(yapss._standin.__file__).parent


def test_the_package_imports_nothing_else_from_yapss():
    offenders = []
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] if node.level == 0 else []
            else:
                continue
            offenders += [
                f"{path.name}: {name}"
                for name in names
                if name.startswith("yapss") and not name.startswith("yapss._standin")
            ]
    assert offenders == []


# the model: drag from a density "auto" cannot trace, and the same density traced

RHO0, H0 = 1.225, 8500.0


def numpy_density(h):
    return RHO0 * np.exp(-h / H0)


def traced_pair():
    """Return (variables, traced drag, stand-in drag, uses)."""
    h, v = ca.SX.sym("h"), ca.SX.sym("v")
    variables = ca.vertcat(h, v)
    exact = 0.5 * RHO0 * ca.exp(-h / H0) * v**2
    density = external(numpy_density, scale=H0, vectorized=True)
    with tracing() as uses:
        drag = 0.5 * density(h) * v**2
    return variables, exact, drag, uses


POINTS = np.array([[0.0, 2000.0, 8500.0, 20000.0], [100.0, 150.0, 200.0, 250.0]])


def test_a_plain_sx_argument_makes_a_stand_in():
    variables, exact, drag, uses = traced_pair()
    assert len(uses) == 1
    assert ca.depends_on(drag, uses[0].symbols)


def test_the_function_evaluates_at_many_points():
    variables, exact, drag, uses = traced_pair()
    F = Function("drag", variables, drag, uses)
    reference = ca.Function("exact", [variables], [exact])(POINTS).full()
    np.testing.assert_allclose(F.eval(POINTS), reference, rtol=1e-12)


def test_the_jacobian_and_hessian_are_the_traced_ones():
    variables, exact, drag, uses = traced_pair()
    F = Function("drag", variables, drag, uses)
    jac = ca.Function("jacobian_exact", [variables], [ca.jacobian(exact, variables)])
    hes = ca.Function("hessian_exact", [variables], [ca.hessian(exact, variables)[0]])
    width = POINTS.shape[1]
    expected_jac = jac(POINTS).full().reshape(1, width, 2).transpose(1, 0, 2)
    expected_hes = hes(POINTS).full().reshape(2, width, 2).transpose(1, 0, 2)
    # the density is differenced in h, so agreement is at the differencing level
    np.testing.assert_allclose(F.jacobian(POINTS), expected_jac, rtol=1e-7)
    np.testing.assert_allclose(F.hessian(POINTS), expected_hes, rtol=1e-4)


def test_hessian_weights_are_applied_per_point():
    variables, exact, drag, uses = traced_pair()
    outputs = ca.vertcat(drag, variables[1] ** 2)
    F = Function("two", variables, outputs, uses)
    weights = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
    full = F.hessian(POINTS, weights)
    drag_only = F.hessian(POINTS, np.array([[1.0], [0.0]]))
    # the second output's Hessian is 2 in the (v, v) entry and nothing else
    extra = full - drag_only
    np.testing.assert_allclose(extra[:, 1, 1], 2.0 * weights[1])
    np.testing.assert_allclose(extra[:, 0, :], 0.0)


def test_symbols_outside_a_trace_are_refused():
    density = external(numpy_density)
    with pytest.raises(RuntimeError, match="outside a trace"):
        density(ca.SX.sym("h"))


def test_a_vector_argument_is_refused():
    density = external(numpy_density)
    with tracing(), pytest.raises(ValueError, match="scalar arguments"):
        density(ca.SX.sym("h", 2))


def test_numbers_are_just_the_function():
    density = external(numpy_density, vectorized=True)
    h = np.array([0.0, 8500.0])
    np.testing.assert_allclose(density(h), numpy_density(h))
    assert density(0.0) == pytest.approx(RHO0)
