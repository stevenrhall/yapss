"""

The step sizes of a central difference, from the precision of what is differenced.

A central first difference has a truncation error that grows as the square of the step and a
rounding error that grows as the precision over the step; a central second difference has the
same truncation error and a rounding error that grows as the precision over the square of the
step. Balancing the two gives a step proportional to the cube root of the precision for the
first and to its fourth root for the second. Each is relative: it multiplies the scale of the
variable that is stepped.

"""

from __future__ import annotations

__all__ = ["EPS", "difference_steps"]

# EPS should be 2 ** -53, but calculate to be sure
exponent: int = 1
while 1 - 2 ** float(-exponent) < 1:
    exponent += 1
exponent -= 1
EPS: float = 2 ** (-exponent)


def difference_steps(eps: float = EPS) -> tuple[float, float]:
    """Return the relative steps for central first and second differences.

    Parameters
    ----------
    eps : float, optional
        The relative precision of the function being differenced. The default is that of
        double-precision arithmetic.

    Returns
    -------
    tuple of float
        The step for a first difference and the step for a second difference.
    """
    return (3 * eps) ** (1 / 3), (3 * eps) ** (1 / 4)
