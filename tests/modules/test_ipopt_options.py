"""

Test the yapss._private.ipopt_options module.

"""

import warnings

import numpy as np
import pytest

from yapss import IpoptConvergenceWarning, IpoptOptionSettingWarning
from yapss._private.ipopt_option_specs import IPOPT_DOC_VERSION
from yapss._private.ipopt_options import IpoptOptions, explain_refusal
from yapss.examples.rosenbrock import setup


@pytest.mark.filterwarnings("ignore::yapss._private.solver.IpoptOptionSettingWarning")
def test_ipopt_options():
    """Test the ipopt_options module."""

    # a name close to a real option is a misspelling: refused where it is written
    ocp = setup()
    with pytest.raises(AttributeError, match="Did you mean 'max_iter'"):
        ocp.ipopt_options.max_iters = 1

    # a name nothing like a known one may be an option from another Ipopt build, so it
    # warns and is passed on for Ipopt to judge
    with pytest.warns(IpoptOptionSettingWarning, match="not among the options documented"):
        ocp.ipopt_options.not_a_real_option = 1

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
        ("mumps_mpi_communicator", np.int16(3), 3),
        ("bound_relax_factor", np.float64(2.5), 2.5),
        ("hsllib", "text", "text"),
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
        ("hsllib", [1], "takes a str"),
        ("hsllib", True, "does not take a bool"),
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


@pytest.mark.parametrize(("name", "value"), [("output_file", "log.txt"), ("file_print_level", 5)])
def test_the_file_log_options_are_supported(name, value):
    """Ipopt's docs disclaim these, but the C interface honors them, so YAPSS lists them.

    The documentation's "only works when read from the ipopt.opt options file" is not true
    of the interface YAPSS uses: `test_ipopt_defaults.py` sets both and reads back the log
    Ipopt wrote. They are annotated like any other option, so an IDE suggests them.
    """
    options = IpoptOptions()
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        setattr(options, name, value)
    assert options.get_options()[name] == value
    assert name in IpoptOptions.__annotations__


def test_an_option_this_ipopt_build_lacks_warns_at_the_solve():
    """`file_append` arrived after Ipopt 3.14.11, which the pinned casadi wheel bundles.

    The name and the value are both documented, so setting it says nothing; Ipopt then
    refuses it ("It is not a valid option") and YAPSS reports the one explanation that
    fits -- this build does not provide it. A conda build may, which is why it is not an
    error. This is the real case for that branch, on the wheel this suite runs against.
    """
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        ocp.ipopt_options.file_append = "yes"  # silent: the table has it
    with pytest.warns(IpoptOptionSettingWarning, match="does not provide it"):
        ocp.solve()


def test_refused_option_warning_points_at_the_caller():
    """The warning is attributed to the line that called solve(), not to YAPSS.

    Reached by writing the instance dictionary directly, which is what `get_options` reads:
    the setter now refuses an undocumented name, and a documented option with a value out of
    range raises rather than warns, so this is the remaining path to the warning.
    """
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.__dict__["not_a_real_option"] = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ocp.solve()
    refused = [w for w in caught if issubclass(w.category, IpoptOptionSettingWarning)]
    assert len(refused) == 1
    assert refused[0].filename == __file__
    assert "No option of that name is documented" in str(refused[0].message)


@pytest.mark.parametrize(
    ("name", "value", "is_error", "fragment"),
    [
        ("max_iter", -1, True, "0 <= value"),
        ("mu_strategy", "adaptiv", True, "one of monotone, adaptive"),
        ("linear_solver", "ma27", False, "does not provide it"),
        ("hsllib", "libhsl.so", False, "does not provide it"),
        ("no_such_option", 1, False, "No option of that name"),
    ],
)
def test_a_refusal_is_classified_against_what_ipopt_documents(name, value, is_error, fragment):
    """Ipopt says only that it refused; the documented range or value list says why.

    A free-form string option such as `hsllib` has no value list, so any value counts as
    documented and a refusal can only mean the build lacks it.
    """
    message, error = explain_refusal(name, value, "invalid option or value")
    assert error is is_error
    assert fragment in message
    assert IPOPT_DOC_VERSION in message  # never stated as certain
    assert "console output" in message


def test_a_value_outside_the_documented_range_raises_naming_the_range():
    """Ipopt reports only that it refused; the documented range says why, hedged."""
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iter = -5
    with pytest.raises(ValueError) as info:
        ocp.solve()
    message = str(info.value)
    assert "max_iter" in message
    assert "0 <= value" in message
    assert "console output" in message  # every verdict sends the user to Ipopt itself
