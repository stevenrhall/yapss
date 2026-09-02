"""

Conformance tests for yapss.math against numpy.

Every name exported by ``yapss.math`` is a promise: user callbacks may call it and get
the same answer whether the argument is a float array (the ``"central-difference"``
derivative methods) or an SXW-wrapped casadi symbol (the ``"auto"`` method). A name that
disagrees between the two paths is the worst kind of bug in YAPSS -- the problem is
transcribed differently depending on how derivatives are computed, and nothing raises.

These tests hold that line two ways:

* ``test_agrees_with_numpy`` evaluates each exported ufunc symbolically and compares
  against numpy on floats.
* ``test_every_exported_name_is_classified`` fails if a name is added to
  ``yapss.math.__all__`` without deciding which category it belongs to, which is how the
  gaps below went unnoticed.

Entries in ``KNOWN_BROKEN`` are ``xfail(strict=True)``: fixing one makes the suite fail
until it is removed from the list.

"""

import casadi as ca
import numpy as np
import pytest

from yapss import math
from yapss.math.wrapper import SXW, SXArray, sx_array

# Number of sample points per function.
N = 9

# Sampling domain per argument position. Defaults to DEFAULT_DOMAIN; entries here narrow
# it to where the function is real-valued, or away from poles.
DEFAULT_DOMAIN = (-2.0, 3.0)
DOMAIN = {
    "arccos": [(-0.9, 0.9)],
    "arcsin": [(-0.9, 0.9)],
    "arctanh": [(-0.9, 0.9)],
    "arccosh": [(1.1, 4.0)],
    "log": [(0.1, 4.0)],
    "log10": [(0.1, 4.0)],
    "log2": [(0.1, 4.0)],
    "log1p": [(-0.5, 4.0)],
    "sqrt": [(0.1, 4.0)],
    "reciprocal": [(0.5, 4.0)],
    # positive base, modest exponent
    "power": [(0.5, 3.0), (0.5, 3.0)],
    "pow": [(0.5, 3.0), (0.5, 3.0)],
    "float_power": [(0.5, 3.0), (0.5, 3.0)],
    # nonzero divisor
    "divide": [DEFAULT_DOMAIN, (0.5, 3.0)],
    "true_divide": [DEFAULT_DOMAIN, (0.5, 3.0)],
    "floor_divide": [DEFAULT_DOMAIN, (0.5, 3.0)],
    "mod": [DEFAULT_DOMAIN, (0.5, 3.0)],
    "remainder": [DEFAULT_DOMAIN, (0.5, 3.0)],
    "fmod": [DEFAULT_DOMAIN, (0.5, 3.0)],
}

# Names in __all__ that are out of scope for elementwise symbolic conformance, with the
# reason. These are not defects; they are things a user callback has no business calling
# on a symbolic state.
OUT_OF_SCOPE = {
    "pi": "not a function",
    "all": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "any": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "max": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "min": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "sum": "array reduction, not elementwise",
    "round": "takes a decimals parameter; tested at the halves in test_sxw_scrub.py",
    "clip": "three arguments; symbolic dispatch tested in test_sxw_scrub.py",
    "where": "three arguments; symbolic dispatch tested in test_sxw_scrub.py",
    "matmul": "not elementwise",
    "gcd": "integer domain",
    "lcm": "integer domain",
    "invert": "integer domain",
    "left_shift": "integer domain",
    "right_shift": "integer domain",
    "ldexp": "requires an integer second argument",
    "frexp": "two outputs",
    "modf": "two outputs",
    "divmod": "two outputs",
}

# Exported names that YAPSS deliberately refuses in a callback, because they have no
# symbolic equivalent. A symbolic argument raises UnsupportedMathFunctionError, as it did
# before 0.2.2; a real argument warns and evaluates until 0.3.0, when it raises too -- see
# test_rejected_names_warn_on_real_input, which is the test to flip then.
REJECTED = ("nextafter", "signbit", "spacing")

# Exported names that do not round-trip through SXW. Entries are xfail(strict=True), so
# fixing one fails the suite until it is removed from this list. Empty is the goal
# state, not an invitation to leave it empty: record a defect here rather than deleting
# the test that found it.
KNOWN_BROKEN: dict[str, str] = {}


def elementwise_ufunc_names():
    """Return the exported names that should agree with numpy elementwise."""
    names = []
    for name in math.__all__:
        if name in OUT_OF_SCOPE or name in REJECTED:
            continue
        ufunc = getattr(np, name, None)
        if isinstance(ufunc, np.ufunc) and ufunc.nout == 1:
            names.append(name)
    return sorted(names)


# Explicit sample points for functions whose interesting behaviour is at values a
# linspace will not land on exactly -- zero for the logical functions, and coincident
# operands for the comparisons. `logical_not` passed this suite while being broken
# because the default sampling never hit exactly 0.
SAMPLES = {
    "logical_not": [np.array([-2.0, -1.0, 0.0, 1.0, 2.0])],
    "logical_and": [
        np.array([0.0, 0.0, 1.0, -1.0, 2.0]),
        np.array([0.0, 3.0, 0.0, 1.0, -2.0]),
    ],
    "logical_or": [
        np.array([0.0, 0.0, 1.0, -1.0, 2.0]),
        np.array([0.0, 3.0, 0.0, 1.0, -2.0]),
    ],
}
# all four sign combinations: `fmod` truncates and takes the sign of the dividend,
# while `mod` and `remainder` floor and take the sign of the divisor. A positive-only
# divisor cannot tell a correct implementation from one that confuses the two.
for _name in ("mod", "remainder", "fmod", "floor_divide"):
    SAMPLES[_name] = [
        np.array([7.0, -7.0, 7.0, -7.0, 2.5, -2.5]),
        np.array([3.0, 3.0, -3.0, -3.0, 1.5, -1.5]),
    ]

# large arguments, where the naive log(exp(x) + exp(y)) overflows but numpy does not
SAMPLES["logaddexp"] = [
    np.array([0.0, 1.0, 500.0, -500.0, 800.0]),
    np.array([1.0, 0.0, 800.0, 500.0, -800.0]),
]
SAMPLES["logaddexp2"] = [
    np.array([0.0, 1.0, 500.0, -500.0, 1000.0]),
    np.array([1.0, 0.0, 1000.0, 500.0, -1000.0]),
]

for _name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"):
    SAMPLES[_name] = [
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        np.array([-1.0, 1.0, 0.0, 2.0, -3.0]),  # equal in places, either side elsewhere
    ]


def sample_points(name, nin):
    """Return `nin` arrays of sample points appropriate to the function's domain."""
    if name in SAMPLES:
        return SAMPLES[name]
    domains = DOMAIN.get(name, [DEFAULT_DOMAIN] * nin)
    if len(domains) < nin:
        domains = list(domains) + [DEFAULT_DOMAIN] * (nin - len(domains))
    points = []
    for k in range(nin):
        lo, hi = domains[k]
        # offset each argument slightly so the two operands are never identical, which
        # would hide an implementation that returns one of its arguments unchanged
        shift = 0.1 * k * (hi - lo)
        points.append(np.linspace(lo + shift, hi - shift, N))
    return points


# The three ways a symbol reaches yapss.math, each with its own dispatch path:
#   scalar  -- a bare SXW, as final_time, parameter[0], or state[0][0] is. Goes through
#              SXW.__array_ufunc__.
#   objarr  -- a plain object ndarray of SXW, which numpy hands to the element methods
#              SXW.__getattr__ supplies.
#   sxarr   -- an SXArray, as state[i], control[i], and final_state are. Same as objarr,
#              plus the comparison overrides.
# Before 0.2.3 only objarr was tested here, and sixteen names failed on scalar.
INPUT_KINDS = ("scalar", "objarr", "sxarr")


def evaluate_symbolically(name, points, kind="objarr"):
    """Evaluate ``yapss.math.<name>`` on SXW symbols of one input kind; return floats."""
    n = len(points[0])
    symbols = [ca.SX.sym(f"v{k}", n) for k in range(len(points))]
    function = getattr(math, name)
    if kind == "scalar":
        result = [function(*[SXW(symbols[k][i]) for k in range(len(points))]) for i in range(n)]
    else:
        build = sx_array if kind == "sxarr" else (lambda items: np.array(items, dtype=object))
        arrays = [build([SXW(symbols[k][i]) for i in range(n)]) for k in range(len(points))]
        result = np.atleast_1d(function(*arrays))
        if kind == "sxarr":
            assert isinstance(result, SXArray), f"{name} on an SXArray returned {type(result)}"
    expression = ca.vertcat(*[SXW(item)._value for item in result])
    casadi_function = ca.Function("f", symbols, [expression])
    return np.asarray(casadi_function(*points)).flatten().astype(float)


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("name", elementwise_ufunc_names())
def test_agrees_with_numpy(name, kind, request):
    """Check that the SXW path computes the same value as numpy on floats."""
    if name in KNOWN_BROKEN:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=KNOWN_BROKEN[name]))

    nin = getattr(getattr(np, name), "nin", 1)  # round is not a ufunc
    points = sample_points(name, nin)
    expected = np.asarray(getattr(np, name)(*points), dtype=float)
    actual = evaluate_symbolically(name, points, kind)

    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True), (
        f"yapss.math.{name} disagrees with numpy:\n"
        f"  args     = {[np.round(p, 4) for p in points]}\n"
        f"  expected = {np.round(expected, 6)}\n"
        f"  actual   = {np.round(actual, 6)}"
    )


def test_every_exported_name_is_classified():
    """Every name in ``yapss.math.__all__`` must be tested or explicitly out of scope.

    This is the check that was missing: it is why `exp2` could be wrong and `cbrt` could
    raise without anything noticing.
    """
    tested = set(elementwise_ufunc_names())
    classified = tested | set(OUT_OF_SCOPE) | set(REJECTED)
    unclassified = sorted(set(math.__all__) - classified)
    assert not unclassified, (
        f"{len(unclassified)} name(s) exported by yapss.math are neither covered by "
        f"test_agrees_with_numpy nor listed in OUT_OF_SCOPE or REJECTED: "
        f"{unclassified}. Classify each, or make it conform."
    )


def test_known_broken_names_are_all_exported():
    """Guard against KNOWN_BROKEN going stale as names leave ``__all__``."""
    stale = sorted(set(KNOWN_BROKEN) - set(math.__all__))
    assert not stale, f"KNOWN_BROKEN lists names no longer exported by yapss.math: {stale}"


@pytest.mark.parametrize("name", REJECTED)
def test_rejected_names_raise_on_symbolic_input(name):
    """A rejected name must raise on a symbolic argument, naming itself and numpy."""
    function = getattr(math, name)
    nin = getattr(getattr(np, name), "nin", 1)  # round is not a ufunc
    symbolic = np.array([SXW(ca.SX.sym("v"))], dtype=object)

    with pytest.raises(math.UnsupportedMathFunctionError) as excinfo:
        function(*(symbolic,) * nin)

    message = str(excinfo.value)
    assert name in message
    assert f"numpy.{name}" in message


@pytest.mark.parametrize("name", REJECTED)
def test_rejected_names_warn_on_real_input(name):
    """Real arguments warn and still evaluate; flip this to raise in 0.3.0.

    These worked on real arrays through 0.2.1, so a patch release may not take them
    away. The end state is rejection on both paths, so that a formulation cannot come
    to depend on which derivative method is selected.
    """
    function = getattr(math, name)
    numpy_function = getattr(np, name)
    arguments = (np.array([1.0, 2.0]),) * getattr(numpy_function, "nin", 1)

    with pytest.warns(math.UnsupportedMathFunctionWarning) as record:
        result = function(*arguments)

    assert np.array_equal(result, numpy_function(*arguments))
    message = str(record[0].message)
    assert name in message
    assert "0.3.0" in message


def test_unsupported_error_is_a_type_error():
    """numpy raises TypeError for these today; keep that catchable."""
    assert issubclass(math.UnsupportedMathFunctionError, TypeError)


def test_unsupported_warning_is_a_future_warning():
    """DeprecationWarning is suppressed by default outside __main__; this must not be."""
    assert issubclass(math.UnsupportedMathFunctionWarning, FutureWarning)
