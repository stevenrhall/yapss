"""Messages about Ipopt options carry what the user needs, not a pointer to Ipopt's console."""

from __future__ import annotations

import warnings

import pytest

from tests.support.legacy.examples import rosenbrock
from yapss.examples import hs071


def test_invalid_option_status_lists_the_options_set() -> None:
    """Status -12: Ipopt took the option but could not use it, so the message lists them."""
    problem = hs071.setup()
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.linear_solver = "ma27"  # HSL, loaded at run time; absent here
    try:
        problem.solve()
    except ValueError as error:
        message = str(error)
    else:
        pytest.skip("this Ipopt can load MA27, so the option is valid here")
    assert "Status -12" in message
    assert "output above" not in message
    assert "print_level is 1 or more" in message
    assert "linear_solver = 'ma27'" in message


def test_returned_value_message_names_no_type() -> None:
    """The back end's message, which only the legacy front end reaches; 0.4 words its own."""
    problem = rosenbrock.setup()
    problem.ipopt_options.print_level = 0

    def objective(arg):
        return arg.parameter[0]

    problem.functions.objective = objective
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="callback returned a value; a callback assigns") as e:
            problem.solve()
    assert "SXW" not in str(e.value)
