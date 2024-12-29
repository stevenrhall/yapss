"""

Test the yapss._private.ipopt_options module.

"""

import pytest

from yapss._private.solver import IpoptOptionSettingWarning
from yapss.examples.rosenbrock import setup


@pytest.mark.filterwarnings("ignore::yapss._private.solver.IpoptOptionSettingWarning")
def test_ipopt_options():
    """Test the ipopt_options module."""

    # Test with a non-existent option
    ocp = setup()
    ocp.ipopt_options.not_a_real_option = 1
    msg = r"^Failed to set option 'not_a_real_option' with value '1'"
    with pytest.warns(IpoptOptionSettingWarning, match=msg):
        ocp.solve()

    # Test with wrong type (value for tol must be a float)
    ocp = setup()
    ocp.ipopt_options.tol = "not_a_float"
    msg = r"^Failed to set option 'tol' with value 'not_a_float'"
    with pytest.warns(IpoptOptionSettingWarning, match=msg):
        ocp.solve()

    # A value of None should be ignored and should not trigger a warning
    ocp = setup()
    ocp.ipopt_options.tol = None
    ocp.solve()
