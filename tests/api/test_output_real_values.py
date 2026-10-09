"""A callback output row takes real numbers; a complex or string value is refused where written.

Without the check, under "auto" a complex scalar failed inside CasADi while the derivatives were
built, naming neither the output nor the row; under the central-difference methods it was handed
to the transcription's float storage, which dropped the imaginary part with only a NumPy warning,
and the solve went on with the wrong values.
"""

import re
import warnings

import numpy as np
import pytest

from yapss.examples import brachistochrone_minimal
from yapss.math import sin

METHODS = ["auto", "central-difference"]


def solve_with(value, method="central-difference"):
    """Solve the minimal brachistochrone with `value(v)` written to out.dynamics.x."""
    problem = brachistochrone_minimal.setup()
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg, out):
        v, u = arg.state.v, arg.control.u
        out.dynamics.x = value(v)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = brachistochrone_minimal.G0 * sin(u)

    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.max_iter = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        return problem.solve()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("value", "got"),
    [
        (np.complex128(1 + 1j), "a complex128, np.complex128(1+1j)"),
        (0j, "a complex, 0j"),
        ("0", "a str, '0'"),
        (np.str_("1"), "a str_"),
    ],
    ids=["numpy-complex", "complex", "str", "numpy-str"],
)
def test_a_row_refuses_what_is_not_real(value, got, method):
    message = f"out.dynamics.x: a row takes real numbers, got {got}"
    with pytest.raises(TypeError, match=re.escape(message)):
        solve_with(lambda v: value, method)


def test_a_complex_array_is_refused_rather_than_stored_without_its_imaginary_part():
    with pytest.raises(TypeError, match="got an array of dtype complex128"):
        solve_with(lambda v: v * (1 + 0j))


@pytest.mark.parametrize(
    "value",
    [np.int64(1), np.float32(1.0), np.array(1.0)],
    ids=["int64", "float32", "0-d array"],
)
def test_real_values_are_still_accepted(value):
    solve_with(lambda v: value)


def test_a_boolean_array_is_still_accepted():
    solve_with(lambda v: np.asarray(v) > -1.0)
