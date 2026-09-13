"""

Test the yapss._private.ipopt_options module.

"""

import warnings

import numpy as np
import pytest

from yapss import IpoptConvergenceWarning
from yapss._private.ipopt_options import IpoptOptions
from yapss._private.solver import IpoptOptionSettingWarning
from yapss.examples.rosenbrock import setup


@pytest.mark.filterwarnings("ignore::yapss._private.solver.IpoptOptionSettingWarning")
def test_ipopt_options():
    """Test the ipopt_options module."""

    # Test with a non-existent option
    ocp = setup()
    ocp.ipopt_options.not_a_real_option = 1
    msg = r"^Ipopt refused option 'not_a_real_option' with value 1: .*not applied"
    with pytest.warns(IpoptOptionSettingWarning, match=msg):
        ocp.solve()

    # A wrong kind for a documented option is refused at assignment (see below); an
    # unknown option can only be judged by Ipopt, so a refused value there still warns
    ocp = setup()
    ocp.ipopt_options.not_a_real_option = "not_a_float"
    msg = r"^Ipopt refused option 'not_a_real_option' with value 'not_a_float'"
    with pytest.warns(IpoptOptionSettingWarning, match=msg):
        ocp.solve()

    # A value of None should be ignored and should not trigger a warning
    ocp = setup()
    ocp.ipopt_options.tol = None
    ocp.solve()


# ------------------------------------------------------------------------------------
# option kinds, checked at assignment
#
# Ipopt refuses a value sent to the wrong registry (Integer, Number, String), and through
# 0.2.2 the backend chose the registry from the Python type of the value: a Python int for
# max_wall_time went to the Integer registry and was refused, and a NumPy integer was
# refused outright, both demoted to a warning at solve time with the option dropped.


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("max_iter", 7, 7),
        ("max_iter", np.int64(7), 7),
        ("max_wall_time", 60, 60.0),
        ("max_wall_time", 60.5, 60.5),
        ("tol", np.float32(0.5), 0.5),
        ("tol", np.int32(1), 1.0),
        ("linear_solver", "mumps", "mumps"),
        ("not_annotated", np.int16(3), 3),
        ("not_annotated", np.float64(2.5), 2.5),
        ("not_annotated", "text", "text"),
    ],
)
def test_option_values_are_stored_as_the_python_kind(name, value, expected):
    options = IpoptOptions()
    setattr(options, name, value)
    stored = options.get_options()[name]
    assert stored == expected
    assert type(stored) is type(expected)


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("max_iter", 2.0, "Integer option 'max_iter' takes an int"),
        ("max_iter", "7", "Integer option 'max_iter' takes an int"),
        ("max_iter", True, "does not take a bool"),
        ("tol", "1e-8", "Number option 'tol' takes a float or int"),
        ("tol", False, "does not take a bool"),
        ("linear_solver", 3, "String option 'linear_solver' takes a str"),
        ("mu_strategy", True, "does not take a bool"),
        ("not_annotated", [1], "takes an int, float, or str"),
        ("not_annotated", True, "does not take a bool"),
    ],
)
def test_wrong_kind_is_refused_at_assignment(name, value, match):
    options = IpoptOptions()
    before = options.get_options()
    with pytest.raises(TypeError, match=match):
        setattr(options, name, value)
    assert options.get_options() == before


def test_numpy_integer_max_iter_is_applied():
    """The value reaches Ipopt: one iteration, then the iteration-limit status."""
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iter = np.int64(1)
    with pytest.warns(IpoptConvergenceWarning, match="Maximum Number of Iterations"):
        solution = ocp.solve()
    assert solution.nlp_info.ipopt_status == -1


def test_int_valued_number_option_does_not_warn():
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_wall_time = 60
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        ocp.solve()


def test_refused_option_warning_points_at_the_caller():
    """The warning is attributed to the line that called solve(), not to YAPSS."""
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iter = -5  # valid kind, out of Ipopt's range
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ocp.solve()
    refused = [w for w in caught if issubclass(w.category, IpoptOptionSettingWarning)]
    assert len(refused) == 1
    assert refused[0].filename == __file__
    assert "max_iter" in str(refused[0].message)
    assert "not applied" in str(refused[0].message)
