"""

Test the yapss._backend.ipopt_options module.

"""

import warnings

import numpy as np
import pytest

from yapss._backend.ipopt_options import IpoptOptions, refusal_message
from yapss._backend.mseipopt import library
from yapss._legacy import IpoptConvergenceWarning, IpoptOptionSettingWarning
from yapss._legacy.examples.rosenbrock import setup


@pytest.mark.filterwarnings("ignore::yapss._backend.solver.IpoptOptionSettingWarning")
def test_ipopt_options():
    """Test the ipopt_options module."""

    # which options exist is Ipopt's to judge, so a name is neither refused nor warned about
    # where it is written: a misspelling and an option from another build alike are passed on
    ocp = setup()
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        ocp.ipopt_options.max_iters = 1
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


# Ipopt 3.14.13 added `file_append` (Ipopt ChangeLog, #720). The casadi wheel bundles 3.14.11
# and conda-forge ships a later Ipopt, so which of the two tests below runs depends on the
# build loaded, not on the environment: a version check stays right if the wheel moves on.
FILE_APPEND_VERSION = (3, 14, 13)


def _loaded_ipopt_version() -> tuple[int, int, int] | None:
    path = library.initialize_ipopt()
    info = library.read_ipopt_header(path)
    return None if info is None else info.version


def _label(version: tuple[int, int, int] | None) -> str:
    return "version unknown" if version is None else ".".join(map(str, version))


def _set_file_append(ocp):
    ocp.ipopt_options.print_level = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        ocp.ipopt_options.file_append = "yes"  # silent: the table has it


def test_an_option_this_ipopt_build_lacks_warns_at_the_solve():
    """On an Ipopt older than the option, the solve reports that this build lacks it.

    The name and the value are both documented, so setting it says nothing; Ipopt then
    refuses it ("It is not a valid option") and YAPSS reports the one explanation that
    fits -- this build might not provide it. This is the real case for that hint.
    """
    version = _loaded_ipopt_version()
    if version is None or version >= FILE_APPEND_VERSION:
        pytest.skip(f"the loaded Ipopt, {_label(version)}, provides file_append or has no version")
    ocp = setup()
    _set_file_append(ocp)
    with pytest.warns(IpoptOptionSettingWarning, match="might not provide it"):
        ocp.solve()


def test_an_option_this_ipopt_build_has_is_accepted_at_the_solve():
    """On an Ipopt that has the option, the same setting solves without a warning."""
    version = _loaded_ipopt_version()
    if version is None or version < FILE_APPEND_VERSION:
        pytest.skip(f"the loaded Ipopt, {_label(version)}, lacks file_append or has no version")
    ocp = setup()
    _set_file_append(ocp)
    with warnings.catch_warnings():
        warnings.simplefilter("error", IpoptOptionSettingWarning)
        ocp.solve()


def test_refused_option_warning_points_at_the_caller():
    """The warning is attributed to the line that called solve(), not to YAPSS, and is one.

    An unknown name is not warned about where it is written, so the refusal at the solve is
    the only warning it produces.
    """
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.not_a_real_option = 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ocp.solve()
    refused = [w for w in caught if issubclass(w.category, IpoptOptionSettingWarning)]
    assert len(refused) == 1
    assert refused[0].filename == __file__
    assert "Ipopt refused option 'not_a_real_option' with value 1" in str(refused[0].message)


@pytest.mark.parametrize(
    ("name", "value", "hint"),
    [
        ("max_iters", 1, "Did you mean 'max_iter'?"),
        ("max_iter", -1, "might not provide it, or might not accept that value"),
        ("mu_strategy", "adaptiv", "might not provide it, or might not accept that value"),
        ("linear_solver", "ma27", "might not provide it, or might not accept that value"),
        ("no_such_option", 1, None),
    ],
)
def test_a_refusal_message_hints_only_what_the_table_knows(name, value, hint):
    """Ipopt says only that it refused; the table adds a hint, never a verdict.

    A name close to a documented one may be a misspelling; a documented option may be missing
    from this build or refuse the value; a name the table knows nothing about gets no hint.
    """
    message = refusal_message(name, value)
    assert message.startswith(f"Ipopt refused option '{name}' with value {value!r}.")
    assert "not applied, and the solve continues with Ipopt's default" in message
    assert "Check Ipopt's console output above for the exact cause." in message
    if hint is None:
        assert "Did you mean" not in message
        assert "might not" not in message
    else:
        assert hint in message


def test_a_value_ipopt_refuses_warns_and_the_solve_continues():
    """A value is Ipopt's to judge: its refusal warns, and the solve uses Ipopt's default."""
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iter = -5
    with pytest.warns(IpoptOptionSettingWarning, match="Ipopt refused option 'max_iter'"):
        solution = ocp.solve()
    assert solution.converged


def test_a_misspelled_option_warns_at_the_solve_with_a_suggestion():
    ocp = setup()
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iters = 50
    with pytest.warns(IpoptOptionSettingWarning, match="Did you mean 'max_iter'"):
        ocp.solve()
