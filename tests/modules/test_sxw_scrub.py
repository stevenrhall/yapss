"""

Tests from the SXW scrub of 2026-09-02: the operand-type matrix, and the functions that
numpy evaluates by taking a truth value.

``yapss.math`` operates on four kinds of operand -- a number, a float ndarray, a scalar
``SXW``, and an ``SXArray`` -- and every combination must give the finite-difference answer
in the ``"auto"`` container. Two further kinds used to leak in: a plain object ndarray of
``SXW`` (whose comparisons numpy coerced to ``bool``) and an ``SXW`` wrapping a casadi matrix
(from any scalar-times-array product). Both are refused or normalized now.

The ``pipeline`` tests are the ones that matter: they transcribe the same problem under
``"auto"`` and ``"central-difference"`` and compare the discrete constraints. Before the
scrub, ``clip``, ``where``, ``all``, ``any``, and a Python ``if`` on a symbol each produced a
silently different NLP.

"""

import casadi as ca
import numpy as np
import pytest

import yapss
from yapss import math as ym
from yapss._private.auto import make_auto_functions
from yapss._private.central_difference import make_cd_functions
from yapss._private.guess import make_initial_guess_nlp
from yapss._private.mesh import Mesh
from yapss._private.nlp import NLP
from yapss.math.wrapper import SXW, SXArray, sx_array

# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------

XF = np.array([2.0, -1.0, 3.0])
TF = 1.5
S0 = 0.5


def symbolic_operands():
    """Return (final_state, final_time, parameter[0]) as the arg objects build them."""
    xf = ca.SX.sym("xf", 3)
    tf = ca.SX.sym("tf")
    s = ca.SX.sym("s")
    return (sx_array([SXW(xf[i]) for i in range(3)]), SXW(tf), SXW(s)), (xf, tf, s)


def evaluate(body):
    """Evaluate `body` numerically, and symbolically then compiled; return both."""
    expected = np.atleast_1d(np.asarray(body(XF.copy(), TF, S0), dtype=float)).flatten()
    (xf, tf, s), symbols = symbolic_operands()
    out = body(xf, tf, s)
    items = [out] if isinstance(out, SXW) else np.atleast_1d(out).flatten().tolist()
    expression = ca.vertcat(*[SXW(item)._value for item in items])
    actual = np.asarray(ca.Function("f", list(symbols), [expression])(XF, TF, S0)).flatten()
    return expected, actual


# ------------------------------------------------------------------------------------
# the operand-type matrix
# ------------------------------------------------------------------------------------

W = SXW(ca.SX.sym("w"))
A = sx_array([SXW(ca.SX.sym("a0")), SXW(ca.SX.sym("a1")), SXW(ca.SX.sym("a2"))])
O = np.array([SXW(ca.SX.sym("o0")), SXW(ca.SX.sym("o1")), SXW(ca.SX.sym("o2"))], dtype=object)
F = np.array([1.0, 2.0, 3.0])
N = 2.0


@pytest.mark.parametrize(
    ("left", "right", "expected_type", "expected_shape"),
    [
        (W, N, SXW, None),
        (N, W, SXW, None),
        (W, W, SXW, None),
        (W, F, SXArray, (3,)),
        (F, W, SXArray, (3,)),
        (W, A, SXArray, (3,)),
        (A, W, SXArray, (3,)),
        (W, O, SXArray, (3,)),
        (O, W, SXArray, (3,)),
        (A, N, SXArray, (3,)),
        (N, A, SXArray, (3,)),
        (A, F, SXArray, (3,)),
        (F, A, SXArray, (3,)),
        (A, A, SXArray, (3,)),
    ],
    ids=lambda v: (
        getattr(type(v), "__name__", str(v)) if not isinstance(v, (tuple, type)) else str(v)
    ),
)
@pytest.mark.parametrize("operation", [np.add, np.multiply, np.subtract, np.true_divide, np.power])
def test_binary_operand_matrix(left, right, operation, expected_type, expected_shape):
    """Every operand combination gives a scalar SXW or an SXArray of scalar SXW.

    A scalar SXW times a 3-element array used to produce one SXW wrapping a casadi 3x1
    matrix -- values right, container wrong, and fatal three layers later inside casadi.
    """
    result = operation(left, right)
    assert type(result) is expected_type
    if expected_shape is not None:
        assert result.shape == expected_shape
        for item in result.flat:
            assert isinstance(item, SXW)
            assert item._value.shape == (1, 1)


def test_two_dimensional_object_array_with_a_scalar():
    """A scalar SXW broadcasts over a 2-D object array (it used to raise TypeError)."""
    o22 = np.array([[SXW(ca.SX.sym("a")), SXW(ca.SX.sym("b"))]] * 2, dtype=object)
    result = W * o22
    assert type(result) is SXArray
    assert result.shape == (2, 2)
    assert str(result[0, 1]) == "SXW(SX((w*b)))"


def test_operator_and_ufunc_agree_on_a_scalar():
    """``a * b`` and ``np.multiply(a, b)`` are the same operation, whatever the operands."""
    for left, right in [(W, A), (A, W), (W, F), (W, N)]:
        by_operator = left * right
        by_ufunc = np.multiply(left, right)
        assert type(by_operator) is type(by_ufunc)
        assert [str(x) for x in np.atleast_1d(by_operator).flat] == [
            str(x) for x in np.atleast_1d(by_ufunc).flat
        ]


@pytest.mark.parametrize(
    "operation",
    [
        lambda x: x + 1,
        lambda x: 1 + x,
        lambda x: -x,
        lambda x: x**2,
        lambda x: x[0:2],
        lambda x: x[[0, 2]],
        lambda x: x.copy(),
        lambda x: x.reshape(3, 1),
        lambda x: np.sin(x),
        lambda x: ym.sqrt(abs(x)),
        lambda x: ym.maximum(x, 1.0),
        lambda x: ym.arctan2(x, x),
        lambda x: ym.clip(x, 0.0, 1.0),
        lambda x: ym.where(x > 0, x, 0.0),
        lambda x: np.concatenate([x, x]),
        lambda x: x.cumsum(),
        lambda x: (x < 1),
        lambda x: (x < 1) & (x > -1),
        lambda x: ~(x < 1),
    ],
)
def test_sxarray_is_closed(operation):
    """An operation on an SXArray gives an SXArray, never a plain object ndarray.

    A plain object ndarray of SXW is the type whose comparisons numpy coerces to bool.
    """
    assert type(operation(A)) is SXArray


# ------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------


def test_sxw_refuses_a_non_scalar():
    """An SXW holds one scalar; a casadi matrix is refused where it is produced."""
    with pytest.raises(TypeError, match="exactly one scalar"):
        SXW(ca.SX.sym("m", 2))
    with pytest.raises(TypeError, match="exactly one scalar"):
        SXW(np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "take_truth_value",
    [
        lambda w: bool(w),
        lambda w: w if w > 1 else -w,
        lambda w: max(w, 1.0),
        lambda w: sorted([w, w]),
        lambda w: w in [SXW(ca.SX.sym("other"))],
        lambda w: w and 1.0,
        lambda w: not w,
    ],
)
def test_truth_value_is_refused(take_truth_value):
    """Every Python construct that takes a truth value raises, as casadi's SX does.

    Without ``__bool__`` each of these silently took the ``True`` branch under ``"auto"``
    and evaluated the real condition under the finite-difference methods.
    """
    with pytest.raises(TypeError, match="truth value of a symbolic value"):
        take_truth_value(W)


def test_truth_value_of_a_one_element_sxarray_is_refused():
    with pytest.raises(TypeError, match="truth value of a symbolic value"):
        bool(A[0:1])


@pytest.mark.parametrize("function", ["isnan", "isfinite", "matmul"])
def test_ufunc_without_an_implementation_is_refused(function):
    """A ufunc the table does not cover raises, rather than reaching casadi's own interop."""
    with pytest.raises(ym.UnsupportedMathFunctionError, match="no symbolic implementation"):
        getattr(np, function)(W, W) if function == "matmul" else getattr(np, function)(W)


# Rounding is exact, including at the ties. Every half-integer in a range, its two
# floating-point neighbors (where floor(x + 0.5) schemes fail: the addition rounds up
# when x is half an ulp below the tie), and the classic decimals cases (2.675 rounds to
# 2.68, not the builtin's 2.67: 2.675 * 100 is exactly 267.5 in double) -- all must match
# numpy exactly.
HALVES = np.arange(-5.5, 6.0, 1.0)
ROUNDING_POINTS = np.concatenate(
    [
        HALVES,
        np.nextafter(HALVES, np.inf),
        np.nextafter(HALVES, -np.inf),
        np.nextafter(np.nextafter(HALVES, np.inf), np.inf),
        np.array([0.0, -0.0, 1e-300, -1e-300, 0.49999999999999994, 2**52, -(2**52), 2**53]),
    ]
)
DECIMALS_POINTS = np.array([2.675, 1.005, 0.125, 0.375, -2.675, 1234.5, -1234.5, 0.045, 1e-3])


def rounding_kinds(values):
    """Yield the same values as a scalar SXW list, an object array, and an SXArray."""
    symbols = ca.SX.sym("v", len(values))
    wrapped = [SXW(symbols[i]) for i in range(len(values))]
    yield "scalar", symbols, wrapped
    yield "objarr", symbols, np.array(wrapped, dtype=object)
    yield "sxarr", symbols, sx_array(wrapped)


def evaluate_rounding(function, values, **kwargs):
    results = {}
    for kind, symbols, argument in rounding_kinds(values):
        if kind == "scalar":
            out = [function(item, **kwargs) for item in argument]
        else:
            out = list(np.atleast_1d(function(argument, **kwargs)))
        expression = ca.vertcat(*[SXW(item)._value for item in out])
        results[kind] = np.asarray(ca.Function("f", [symbols], [expression])(values)).flatten()
    return results


@pytest.mark.parametrize("name", ["rint", "round"])
def test_rounding_matches_numpy_at_halves_and_their_neighbors(name):
    """rint and round are bit-exact against numpy at every tie and its ulp neighbors."""
    expected = getattr(np, name)(ROUNDING_POINTS)
    for kind, actual in evaluate_rounding(getattr(ym, name), ROUNDING_POINTS).items():
        np.testing.assert_array_equal(actual, expected, err_msg=f"{name} on {kind}")


@pytest.mark.parametrize("decimals", [-2, -1, 1, 2, 3])
def test_round_with_decimals_matches_numpy(decimals):
    """round(x, decimals) reproduces numpy's scale-rint-unscale exactly, negatives included."""
    expected = np.round(DECIMALS_POINTS, decimals)
    for kind, actual in evaluate_rounding(ym.round, DECIMALS_POINTS, decimals=decimals).items():
        np.testing.assert_array_equal(actual, expected, err_msg=f"round on {kind}")


def test_builtin_round_on_a_symbol():
    """round(w) matches the builtin on floats; round(w, n) is refused.

    The builtin with ``ndigits`` rounds the exact decimal value of the float
    (``round(2.675, 2) == 2.67``), which numpy's scale-rint-unscale does not reproduce
    (``np.round(2.675, 2) == 2.68``) and neither can a symbol. Refusing it keeps the
    finite-difference and auto paths from silently disagreeing.
    """
    w = SXW(ca.SX.sym("w"))
    f = ca.Function("f", [w._value], [round(w)._value])
    for value in (2.5, 3.5, -0.5, 2.675, 0.125, 0.49999999999999994):
        assert float(f(value)) == round(value)
    with pytest.raises(ym.UnsupportedMathFunctionError, match="exact decimal"):
        round(w, 2)


def test_bare_casadi_value_is_refused():
    with pytest.raises(TypeError, match="bare casadi SX"):
        ym.arctan2(ca.SX.sym("x"), 1.0)


# ------------------------------------------------------------------------------------
# the functions numpy gets wrong by taking a truth value
# ------------------------------------------------------------------------------------

CASES = {
    "clip on a scalar": lambda xf, tf, s: ym.clip(tf, 0.0, 1.0),
    "clip on an array": lambda xf, tf, s: ym.clip(xf, 0.0, 2.5),
    "clip with lo > hi": lambda xf, tf, s: ym.clip(xf, 2.5, 0.0),
    "where": lambda xf, tf, s: ym.where(xf > 0, xf, 0.0),
    "where with scalar branches": lambda xf, tf, s: ym.where(tf > 2, tf, s),
    "max": lambda xf, tf, s: ym.max(xf),
    "min": lambda xf, tf, s: ym.min(xf),
    "amax": lambda xf, tf, s: ym.amax(xf),
    "amin": lambda xf, tf, s: ym.amin(xf),
    "max of a scalar": lambda xf, tf, s: ym.max(tf),
    "all true": lambda xf, tf, s: ym.all(xf > -5),
    "all false": lambda xf, tf, s: ym.all(xf > 0),
    "any true": lambda xf, tf, s: ym.any(xf > 0),
    "any false": lambda xf, tf, s: ym.any(xf > 5),
    "sum of a scalar": lambda xf, tf, s: ym.sum(tf),
    "sum of scalar times array": lambda xf, tf, s: ym.sum(tf * xf**2),
    "scalar times array": lambda xf, tf, s: tf * xf,
    "array times scalar plus scalar": lambda xf, tf, s: xf * s + tf,
    "abs of a scalar": lambda xf, tf, s: ym.abs(tf),
    "square of a scalar": lambda xf, tf, s: ym.square(tf),
    "reciprocal of a scalar": lambda xf, tf, s: ym.reciprocal(tf),
    "negative of a scalar": lambda xf, tf, s: ym.negative(tf),
    "norm": lambda xf, tf, s: ym.linalg.norm(xf),
    "mask times array": lambda xf, tf, s: (xf > 0) * xf,
}


@pytest.mark.parametrize("label", list(CASES))
def test_callback_body_agrees_between_paths(label):
    """The same body gives the same values on floats and on symbols."""
    expected, actual = evaluate(CASES[label])
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_reductions_of_an_sxarray_are_scalars():
    """np.sum / mean / norm of an SXArray give an SXW, not a 0-d SXArray."""
    for value in (ym.sum(A), A.sum(), ym.mean(A), ym.linalg.norm(A), ym.dot(A, A), A @ A):
        assert type(value) is SXW, type(value)


def test_reduction_with_axis_is_refused_on_a_symbol():
    with pytest.raises(TypeError, match="only a full reduction"):
        ym.max(A, axis=0)


def test_reductions_on_real_input_are_numpy():
    """The real path is untouched, keyword arguments included."""
    values = np.array([[1.0, 5.0], [3.0, 2.0]])
    assert np.array_equal(ym.max(values, axis=0), np.max(values, axis=0))
    assert np.array_equal(ym.clip(values, 2.0, 4.0), np.clip(values, 2.0, 4.0))
    assert np.array_equal(ym.where(values > 2, values, 0.0), np.where(values > 2, values, 0.0))
    assert ym.all(values > 0) == np.all(values > 0)


# ------------------------------------------------------------------------------------
# through the transcription
# ------------------------------------------------------------------------------------


def make_problem(nd, discrete_body):
    problem = yapss.Problem(name="scrub", nx=[2], nu=[1], nd=nd)

    def objective(arg):
        arg.objective = arg.phase[0].final_time

    def continuous(arg):
        for q in arg.phase_list:
            x = arg.phase[q].state
            (u,) = arg.phase[q].control
            arg.phase[q].dynamics[:] = [x[1], u]

    def discrete(arg):
        arg.discrete[:] = discrete_body(arg.phase[0].final_state, arg.phase[0].final_time)

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    problem.functions.discrete = discrete
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower, bounds.final_time.upper = 0.5, 3.0
    bounds.state.lower[:], bounds.state.upper[:] = -5.0, 5.0
    bounds.control.lower[:], bounds.control.upper[:] = -2.0, 2.0
    problem.bounds.discrete.lower[:], problem.bounds.discrete.upper[:] = -50.0, 50.0
    guess = problem.guess.phase[0]
    guess.time = [0.0, 1.5]
    # final state [2, -1]: a "> 0" condition is false in one slot and a clip is active
    guess.state = [[0.4, 2.0], [-1.0, -1.0]]
    guess.control = [[0.5, -0.5]]
    problem.mesh.phase[0].collocation_points = [3, 4]
    problem.mesh.phase[0].fraction = [0.5, 0.5]
    problem.spectral_method = "lgr"
    problem.derivatives.order = "second"
    return problem


def discrete_constraints(problem, method):
    problem.derivatives.method = method
    problem.validate()
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    z0 = make_initial_guess_nlp(problem, mesh)
    if method == "auto":
        functions = make_auto_functions(problem)
    else:
        functions = make_cd_functions(problem, z0, mesh.tau_u)
    nlp = NLP(problem, functions, mesh)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(len(z0)))
    return np.asarray(nlp.constraints(z))[-problem.nd :]


PIPELINE_CASES = {
    "clip scalar": (1, lambda xf, tf: [ym.clip(xf[0], -0.5, 0.5)]),
    "clip array": (2, lambda xf, tf: ym.clip(xf, -0.5, 0.5)),
    "max": (1, lambda xf, tf: [ym.max(xf)]),
    "min": (1, lambda xf, tf: [ym.min(xf)]),
    "where": (2, lambda xf, tf: ym.where(xf > 0, xf, 0.0)),
    "all": (1, lambda xf, tf: [ym.all(xf > 0)]),
    "any": (1, lambda xf, tf: [ym.any(xf > 5)]),
    "scalar times array": (2, lambda xf, tf: tf * xf),
    "sum of scalar times array": (1, lambda xf, tf: [ym.sum(tf * xf)]),
}


@pytest.mark.parametrize("label", list(PIPELINE_CASES))
def test_pipeline_agrees_between_auto_and_central_difference(label):
    """The transcribed discrete constraints agree between derivative methods.

    Each of these produced a silently different NLP under ``"auto"`` before the scrub
    (or, for the scalar-times-array cases, failed inside casadi's derivative code).
    """
    nd, body = PIPELINE_CASES[label]
    by_cd = discrete_constraints(make_problem(nd, body), "central-difference")
    by_auto = discrete_constraints(make_problem(nd, body), "auto")
    assert by_cd.shape == (nd,)
    np.testing.assert_allclose(by_auto, by_cd, rtol=1e-9, atol=1e-9)


def test_python_if_on_a_symbol_fails_loudly_in_the_pipeline():
    """A Python ``if`` on a symbol raises under "auto" instead of taking one branch.

    This was the last silent instance of the defect class 0.2.2 fixed: the constraint
    below transcribed to +tf under "auto" and -tf under central differences.
    """
    problem = make_problem(1, lambda xf, tf: [tf if tf > 2 else -tf])
    assert discrete_constraints(problem, "central-difference").shape == (1,)
    with pytest.raises(TypeError, match="truth value of a symbolic value"):
        discrete_constraints(make_problem(1, lambda xf, tf: [tf if tf > 2 else -tf]), "auto")
