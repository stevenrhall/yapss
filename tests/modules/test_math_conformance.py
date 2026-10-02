"""

Conformance tests for yapss.math against numpy.

Every name exported by ``yapss.math`` is a promise: user callbacks may call it and get
the same answer whether the argument is a float array (the ``"central-difference"``
derivative methods) or an SXW-wrapped casadi symbol (the ``"auto"`` method). A name that
disagrees between the two paths is the worst kind of bug in YAPSS -- the problem is
transcribed differently depending on how derivatives are computed, and nothing raises.

These tests hold that line three ways:

* ``test_agrees_with_numpy`` evaluates each exported ufunc symbolically and compares
  against numpy on floats.
* ``test_real_path_is_numpy`` checks that each one, on real arguments, is exactly numpy's
  function: the same values and dtype on floats and integers, or the same exception.
* ``test_every_exported_name_is_classified`` fails if a name is added to
  ``yapss.math.__all__`` without deciding which category it belongs to, which is how the
  gaps below went unnoticed.

Entries in ``KNOWN_BROKEN`` are ``xfail(strict=True)``: fixing one makes the suite fail
until it is removed from the list.

"""

import casadi as ca
import numpy as np
import pytest

import yapss
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
    "acos": [(-0.9, 0.9)],
    "asin": [(-0.9, 0.9)],
    "atanh": [(-0.9, 0.9)],
    "acosh": [(1.1, 4.0)],
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
    "e": "a constant; test_the_constants_are_numpys",
    "euler_gamma": "a constant; test_the_constants_are_numpys",
    "inf": "a constant; test_the_constants_are_numpys",
    "nan": "a constant; test_the_constants_are_numpys",
    "pi": "a constant; test_the_constants_are_numpys",
    "all": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "any": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "max": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "min": "array reduction; symbolic fold tested in test_sxw_scrub.py",
    "amax": "array reduction; alias of max",
    "amin": "array reduction; alias of min",
    "sum": "array reduction, not elementwise",
    "round": "takes a decimals parameter; tested at the halves in test_sxw_scrub.py",
    "clip": "three arguments; symbolic dispatch tested in test_sxw_scrub.py",
    "where": "three arguments; symbolic dispatch tested in test_sxw_scrub.py",
    "matmul": "not elementwise",
    "invert": "integer domain",
}

# Exported names that YAPSS deliberately refuses in a callback, because they have no
# symbolic equivalent. Every argument raises UnsupportedMathFunctionError; a real argument
# warned and evaluated through 0.2.x.
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
#   numpy   -- numpy's own function on an SXArray, which goes through
#              SXArray.__array_ufunc__ rather than yapss.math at all. Before 0.2.3 the
#              two-argument ufuncs raised here.
#   numpy_scalar -- numpy's own function on a bare SXW, through SXW.__array_ufunc__. With
#              the numpy kind, this is why a callback may call numpy's version of any of
#              these names under "auto"; yapss.math marks the promised set.
# Before 0.2.3 only objarr was tested here, and sixteen names failed on scalar.
INPUT_KINDS = ("scalar", "objarr", "sxarr", "numpy", "numpy_scalar")


def evaluate_symbolically(name, points, kind="objarr"):
    """Evaluate ``yapss.math.<name>`` on SXW symbols of one input kind; return floats."""
    n = len(points[0])
    symbols = [ca.SX.sym(f"v{k}", n) for k in range(len(points))]
    function = getattr(np if kind.startswith("numpy") else math, name)
    if kind in ("scalar", "numpy_scalar"):
        result = [function(*[SXW(symbols[k][i]) for k in range(len(points))]) for i in range(n)]
    else:
        build = (
            sx_array
            if kind in ("sxarr", "numpy")
            else (lambda items: np.array(items, dtype=object))
        )
        arrays = [build([SXW(symbols[k][i]) for i in range(n)]) for k in range(len(points))]
        result = np.atleast_1d(function(*arrays))
        if kind in ("sxarr", "numpy"):
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


def outcome(function, arguments):
    """Return what `function` gives on `arguments`: an array, or the type of what it raises."""
    try:
        with np.errstate(all="ignore"):
            return np.asarray(function(*arguments))
    except Exception as exception:
        return type(exception)


# integers reach a callback from a user's own constants, and a function that is numpy's on
# floats can still differ on them, in value or in dtype
INTEGERS = np.array([-2, -1, 0, 1, 2, 3])


@pytest.mark.parametrize("dtype", ["float", "int"])
@pytest.mark.parametrize("name", elementwise_ufunc_names())
def test_real_path_is_numpy(name, dtype):
    """Check that ``yapss.math.<name>`` on real arguments gives exactly numpy's result."""
    nin = getattr(getattr(np, name), "nin", 1)
    if dtype == "float":
        points = sample_points(name, nin)
    else:
        points = [INTEGERS, INTEGERS[::-1]][:nin]
    expected = outcome(getattr(np, name), points)
    actual = outcome(getattr(math, name), points)

    if isinstance(expected, type):
        assert actual is expected, f"numpy.{name} raises {expected.__name__}; got {actual}"
        return
    assert not isinstance(actual, type), f"yapss.math.{name} raised {actual.__name__}"
    assert actual.dtype == expected.dtype, f"yapss.math.{name} returned {actual.dtype}"
    assert np.array_equal(actual, expected, equal_nan=True), (
        f"yapss.math.{name} disagrees with numpy on {points}:\n"
        f"  expected = {expected}\n"
        f"  actual   = {actual}"
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
def test_rejected_names_raise_on_real_input(name):
    """Real arguments raise as symbolic ones do, so no formulation depends on the method.

    They warned and evaluated through 0.2.x, having worked through 0.2.1.
    """
    function = getattr(math, name)
    arguments = (np.array([1.0, 2.0]),) * getattr(getattr(np, name), "nin", 1)

    with pytest.raises(math.UnsupportedMathFunctionError) as excinfo:
        function(*arguments)

    message = str(excinfo.value)
    assert name in message
    assert f"numpy.{name}" in message


def test_unsupported_error_is_a_type_error():
    """numpy raises TypeError for these today; keep that catchable."""
    assert issubclass(math.UnsupportedMathFunctionError, TypeError)


@pytest.mark.parametrize("module", [yapss, math], ids=["yapss", "yapss.math"])
def test_the_removed_warning_says_what_replaced_it(module):
    with pytest.raises(AttributeError, match="removed in 0.3.0.*now raise"):
        _ = module.UnsupportedMathFunctionWarning


# ------------------------------------------------------------------------------------------
# what the module provides
# ------------------------------------------------------------------------------------------

# public names besides the promised functions: the error they raise, and the submodules
NOT_FUNCTIONS = {"UnsupportedMathFunctionError", "functions", "wrapper"}


def test_the_module_provides_only_its_promise():
    """A successful import from yapss.math is the promise, so nothing else is importable."""
    public = {name for name in dir(math) if not name.startswith("_")}
    assert public - NOT_FUNCTIONS == set(math.__all__)


@pytest.mark.parametrize("name", ["e", "euler_gamma", "inf", "nan", "pi"])
def test_the_constants_are_numpys(name):
    """A script written as ``import yapss.math as np`` keeps ``np.inf`` and the rest."""
    assert name in math.__all__
    assert getattr(math, name) is getattr(np, name)


def test_numpy_has_no_constant_the_module_lacks():
    """Every float numpy exports at its top level is a constant the module provides."""
    constants = {
        name for name in dir(np) if not name.startswith("_") and type(getattr(np, name)) is float
    }
    assert constants <= set(math.__all__)


@pytest.mark.parametrize("name", ["linspace", "ndarray", "gcd", "divmod"])
def test_a_numpy_name_says_where_to_import_it(name):
    with pytest.raises(AttributeError, match=f"{name!r} is not one of them.*import it from numpy"):
        getattr(math, name)
    assert not hasattr(math, name)


def test_importing_a_numpy_name_fails():
    with pytest.raises(ImportError, match="cannot import name 'linspace'"):
        from yapss.math import linspace  # noqa: F401


def test_an_unknown_name_is_a_plain_attribute_error():
    with pytest.raises(AttributeError, match=r"^module 'yapss.math' has no attribute 'sine'$"):
        _ = math.sine


def test_numpy_where_refuses_a_symbolic_condition():
    """``where`` is not a ufunc, so numpy's asks the condition for a truth value first.

    It is the one promised name whose numpy version does not work under "auto"; the
    refusal names the one that does.
    """
    x = SXW(ca.SX.sym("x"))
    with pytest.raises(TypeError, match="yapss.math.where"):
        np.where(x > 0, x, 0.0)
    assert isinstance(math.where(x > 0, x, 0.0), SXW)


# ------------------------------------------------------------------------------------------
# NaN propagation, which the central-difference sparsity probe relies on
# ------------------------------------------------------------------------------------------

# Functions and arguments whose NaN is legitimately not propagated: a boolean result has no
# derivative to hide; copysign's result depends on the sign of its second argument, not its
# value; and heaviside's second argument is its value at exactly zero, which the samples avoid.
# IEEE power(1, nan) and power(nan, 0) are 1, but the sample points avoid them too; that case
# is the probe's to handle, not a function's.
BOOLEAN = {
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "invert",
}
ABSORBS_NAN = {("copysign", 1), ("heaviside", 1)}


def nan_cases():
    """Return (name, position) for every argument of every elementwise function."""
    cases = []
    for name in elementwise_ufunc_names():
        if name in BOOLEAN:
            continue
        for position in range(getattr(np, name).nin):
            if (name, position) not in ABSORBS_NAN:
                cases.append((name, position))
    return cases


@pytest.mark.parametrize(("name", "position"), nan_cases())
def test_nan_in_any_argument_gives_nan(name, position):
    """A NaN in any argument is NaN in the result, so the probe sees the dependency."""
    points = [np.array(p, dtype=float) for p in sample_points(name, getattr(np, name).nin)]
    points[position][:] = np.nan
    with np.errstate(all="ignore"):
        result = getattr(math, name)(*points)
    assert np.all(np.isnan(result)), f"yapss.math.{name} dropped a NaN in argument {position}"


@pytest.mark.parametrize("name", ["fmax", "fmin"])
def test_fmax_and_fmin_propagate_nan_unlike_numpy(name):
    """numpy's fmax and fmin ignore NaN; yapss.math's are maximum and minimum."""
    assert getattr(np, name)(1.0, np.nan) == 1.0
    assert np.isnan(getattr(math, name)(1.0, np.nan))


CLIP_VALUES = np.linspace(-2.0, 2.0, 9)
SYMBOLIC_BOUND = -0.5  # the value a symbolic lower bound takes


@pytest.mark.parametrize("shape", ["scalar", "array"])
@pytest.mark.parametrize("bounds", [(-1.0, 1.0), (None, 1.0), (-1.0, None), ("symbol", 1.0)])
@pytest.mark.parametrize("function", [np.clip, math.clip], ids=["numpy", "yapss.math"])
def test_clip_on_a_symbol_agrees_with_numpy(function, bounds, shape):
    """numpy's clip and yapss.math's give numpy's values on a symbol, with a bound of None.

    numpy's clip calls its argument's ``clip`` method, which a single symbol must have, or
    numpy compares through its object loop and a symbol has no truth value.
    """
    symbols = ca.SX.sym("x", len(CLIP_VALUES))
    bound = ca.SX.sym("s")
    lo, hi = bounds
    symbolic_lo = SXW(bound) if lo == "symbol" else lo
    if shape == "scalar":
        result = [function(SXW(symbols[i]), symbolic_lo, hi) for i in range(len(CLIP_VALUES))]
    else:
        values = sx_array([SXW(symbols[i]) for i in range(len(CLIP_VALUES))])
        result = list(function(values, symbolic_lo, hi))
    expression = ca.vertcat(*[SXW(item)._value for item in result])
    actual = np.asarray(
        ca.Function("f", [symbols, bound], [expression])(CLIP_VALUES, SYMBOLIC_BOUND)
    )

    expected = np.clip(CLIP_VALUES, SYMBOLIC_BOUND if lo == "symbol" else lo, hi)
    assert np.array_equal(actual.flatten(), expected)
