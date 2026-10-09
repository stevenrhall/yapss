"""Ipopt's Integer options hold a 32-bit signed integer; a Python int beyond that is refused.

Before, ``max_iter = 2**32`` reached Ipopt as 0, its low 32 bits, and the solve stopped before
its first iteration with only a non-convergence warning to show for it.
"""

from __future__ import annotations

import pytest

from yapss.examples import hs071


@pytest.mark.parametrize("value", [-(2**31), 0, 3000, 2**31 - 1])
def test_in_range_is_accepted(value: int) -> None:
    problem = hs071.setup()
    problem.ipopt_options.max_iter = value
    assert problem.ipopt_options.max_iter == value


@pytest.mark.parametrize("value", [2**31, 2**32, 2**32 + 50, -(2**31) - 1, 10**12])
def test_out_of_range_is_refused_at_assignment(value: int) -> None:
    problem = hs071.setup()
    with pytest.raises(
        ValueError, match=r"greater than or equal to -2\*\*31 and less than 2\*\*31"
    ):
        problem.ipopt_options.max_iter = value


def test_number_option_refuses_an_integer_too_large_for_a_float() -> None:
    problem = hs071.setup()
    with pytest.raises(ValueError, match="too large"):
        problem.ipopt_options.max_cpu_time = 10**400
