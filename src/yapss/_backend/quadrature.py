"""

Provides functions and classes for LGL and LGR integration and differentiation.

* The functions :meth:`lgl` and :meth:`lgr` compute the Legendre-Gauss_Lobatto (LGL)
  and Legendre-Gauss-Radau (LGR) collocation points, quadrature weights, and
  derivative matrix. The functions are memoized to avoid repeated calculations.

* :class:`Mesh` instances represent the mesh structure of the NLP

The arithmetic is on NumPy arrays of mpmath numbers. In a product of such an array and an mpmath
number, the array goes on the left: that is NumPy's elementwise loop. With the mpmath number on
the left, mpmath first tries to convert the whole array, formatting it as text, and only then
hands the product back to NumPy, which in the Legendre recurrences doubled the time taken.

"""

# future imports
from __future__ import annotations

__all__ = ["lgl", "lgr"]

# standard imports
from functools import wraps
from typing import TYPE_CHECKING

# third party imports
import mpmath
import numpy as np
from mpmath import mp

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    # third party imports
    from numpy.typing import NDArray

    # typing
    Array = NDArray[np.float64]
    MPArray = NDArray[np.object_]

    LG_func = Callable[[int], tuple[Array, ...]]
    MP_func = Callable[[int], tuple[MPArray, MPArray, MPArray, MPArray]]

DIGITS = 30
"""The working precision of the computation, in decimal digits.

Twenty digits already give every node, weight and matrix entry to within half an ulp of a
60-digit reference for all three methods up to 100 points, and 22 exactly; 30 leaves margin and
costs little, since the time goes to Python-level arithmetic on mpmath numbers, not to digits.
"""


def memoize(f: MP_func) -> LG_func:
    """Memoize the `lgl`, `lgr` and `lg` functions, computing each at `DIGITS` digits.

    `f` computes a rule in mpmath and returns it as object arrays. The decorated function
    returns the rule rounded once to float, as copies, since the arrays are mutable. The
    rule itself, unrounded, is kept too and returned by the decorated function's ``mp``
    attribute, as read-only object arrays, so that `Mesh` can compose a mesh from it and
    round once at the end. The precision is set for the computation only, with
    `mpmath.mp.workdps`, so that importing YAPSS leaves mpmath's global precision, which is
    the user's, as it was.

    Parameters
    ----------
    f : Callable[[int], tuple[NDArray, NDArray, NDArray, NDArray]]
        decorated function (`lgl`, `lgr` or `lg`), returning mpmath object arrays

    Returns
    -------
    Callable[[int], tuple[NDArray, ...]]
        memoized function
    """
    memo_mp: dict[int, tuple[MPArray, ...]] = {}
    memo: dict[int, tuple[Array, ...]] = {}

    def rule_mp(n: int) -> tuple[MPArray, ...]:
        if n not in memo_mp:
            with mp.workdps(DIGITS):
                arrays = tuple(np.array(item, dtype=object) for item in f(n))
            for array in arrays:
                array.setflags(write=False)
            memo_mp[n] = arrays
        return memo_mp[n]

    @wraps(f)
    def function(n: int) -> tuple[Array, ...]:
        if n not in memo:
            memo[n] = tuple(np.array(item, dtype=float) for item in rule_mp(n))
        return tuple(item.copy() for item in memo[n])

    function.mp = rule_mp  # type: ignore[attr-defined]
    return function


@memoize
def lgl(
    n: int,
) -> tuple[MPArray, MPArray, MPArray, MPArray]:
    r"""Compute the LGL collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Must have :math:`n \ge 2`

    Returns
    -------
    t : NDArray
        The :math:`n` collocation points on the interval [-1,1]
    w : NDArray
        The :math:`n` quadrature weights
    d : NDArray
        The :math:`n \times n` derivative matrix that relates the values of a degree `n-1`
        polynomial at the collocation points to the derivative of the polynomial at those
        points
    d0 : NDArray
        The derivative of the polynomial
    """
    min_collocation_points = 2
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 2."
        raise ValueError(msg)

    t: MPArray = -np.array(
        [mpmath.cos(mp.mpf(np.pi * i / (n - 1))) for i in range(n)],
        dtype=object,
    )

    p: MPArray = np.zeros([n, n + 1], dtype=object)
    for i in range(n):
        for j in range(n + 1):
            p[i, j] = mp.mpf(0)

    # compute the legendre polynomials at the collocation points t, and update t using the
    # newton-raphson method until convergence.
    for _ in range(10):
        # legendre polynomials evaluated at t. p_{n}(t) is in column n+1.
        p[:, 0] = 1
        p[:, 1] = t

        for j in range(2, n + 1):
            # the array on the left of each product: see the module docstring
            fj = mp.mpf(j)
            p[:, j] = (t * p[:, j - 1] * (2 * fj - 1) - p[:, j - 2] * (fj - 1)) / fj

        # do newton raphson step to drive t to zero of g_{n}(t)
        t = t - (p[:, n] - p[:, n - 2]) / ((2 * n - 1) * p[:, n - 1])

    # quadrature weights
    w = 2 / (n * (n - 1) * p[:, n - 1] ** 2)

    # derivative matrix

    d: MPArray = np.zeros([n, n], dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                if i == 0:
                    d[i, i] = mp.mpf(-n * (n - 1) / 4)
                elif i == n - 1:
                    d[i, i] = mp.mpf(+n * (n - 1) / 4)
                else:
                    d[i, i] = mp.mpf(0)
            else:
                d[i, j] = p[i, n - 1] / p[j, n - 1] / (t[i] - t[j])

    d0 = p[:, n - 1] * (mp.mpf(n * (n - 1)) / 4)
    return t, w, d, d0


@memoize
def lgr(
    n: int,
) -> tuple[MPArray, MPArray, MPArray, MPArray]:
    r"""Compute the LGR collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Must have :math:`n \ge 1`

    Returns
    -------
    t : NDArray
        The :math:`n` collocation points on the interval [-1,1]
    w : NDArray
        The :math:`n` quadrature weights
    d : NDArray
        The :math:`n \times n` derivative matrix that relates the values of a degree
        `n-1` polynomial at the collocation points to the derivative of the polynomial at
        those points
    b : NDArray
    """
    min_collocation_points = 1
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 1."
        raise ValueError(msg)

    # TODO: Describe the array b
    # initial guess of collocation points
    t: MPArray = -np.array(
        [mpmath.cos(mp.mpf(2 * np.pi * i / (2 * n - 1))) for i in range(n)],
        dtype=object,
    )

    # initialize the matrix of legendre polynomials
    p: MPArray = np.zeros([n, n + 1], dtype=object)

    # compute the legendre polynomials at the collocation points t, and update t using the
    # newton-raphson method until convergence.
    for _ in range(10):
        # legendre polynomials evaluated at t. p_{n}(t) is in column n+1.
        p[:, 0] = 1
        p[:, 1] = t

        for j in range(2, n + 1):
            p[:, j] = ((2 * j - 1) * t * p[:, j - 1] - (j - 1) * p[:, j - 2]) / j

        # do newton raphson step to drive t to zero of g_{n}(t)
        t = t - (t - 1) * (p[:, n] + p[:, n - 1]) / (n * (p[:, n] - p[:, n - 1]))

    # quadrature weights
    w = (1 - t) / n**2 / p[:, n - 1] ** 2

    # derivative matrix
    d: MPArray = np.zeros([n, n + 1], dtype=object)

    for i in range(n):
        for j in range(n + 1):
            d[i, j] = mp.mpf(0)
            if i == 0 and j == 0:
                d[i, j] = mp.mpf(-(n**2 + 1) / 4)
            elif i == j:
                d[i, j] = -1 / 2 / (1 - t[i])
            elif j < n:
                d[i, j] = p[i, n - 1] / p[j, n - 1] / (t[i] - t[j])
            else:
                d[i, j] = (p[i, n] - p[i, n - 1]) * n / 2 / (t[i] - 1)

    # left eigenvector
    b: MPArray = np.zeros([n + 1], dtype=object)
    ti: MPArray = np.zeros([n + 1], dtype=object)
    ti[:-1] = t
    ti[-1] = mp.mpf(1)

    for i in range(n + 1):
        b[i] = mp.mpf(1)
        for j in range(n + 1):
            if i != j:
                b[i] /= ti[j] - ti[i]

    return t, w, d, b


@memoize
def lg(
    n: int,
) -> tuple[MPArray, MPArray, MPArray, MPArray]:
    r"""Compute the LG collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Must have :math:`n \ge 1`

    Returns
    -------
    t : NDArray
        The :math:`n` collocation points on the interval [-1,1]
    w : NDArray
        The :math:`n` quadrature weights
    d : NDArray
        The :math:`n \times n` derivative matrix that relates the values of a degree `n-1`
        polynomial at the collocation points to the derivative of the polynomial at those
        points
    d0 : NDArray
        The derivative of the polynomial
    """
    min_collocation_points = 1
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 1."
        raise ValueError(msg)

    t: MPArray = -np.array(
        [mpmath.cos(mp.mpf(np.pi * (4 * i + 3) / (4 * n + 2))) for i in range(n)],
        dtype=object,
    )

    p: MPArray = np.zeros([n, n + 1], dtype=object)
    for i in range(n):
        for j in range(n + 1):
            p[i, j] = mp.mpf(0)

    # compute the legendre polynomials at the collocation points t, and update t using the
    # newton-raphson method until convergence.
    for _ in range(10):
        # legendre polynomials evaluated at t. p_{n}(t) is in column n+1.
        p[:, 0] = 1
        p[:, 1] = t

        for j in range(2, n + 1):
            # the array on the left of each product: see the module docstring
            fj = mp.mpf(j)
            p[:, j] = (t * p[:, j - 1] * (2 * fj - 1) - p[:, j - 2] * (fj - 1)) / fj

        # do newton raphson step to drive t to zero of g_{n}(t)
        d_pn_d_t = n * (p[:, n - 1] - t * p[:, n]) / (1 - t**2)
        t = t - p[:, n] / d_pn_d_t

    # quadrature weights
    w: MPArray = np.zeros([n + 1], dtype=object)
    w[1:] = 2 * (1 - t**2) / (n * p[:, n - 1]) ** 2

    # derivative matrix
    d: MPArray = np.zeros([n + 1, n + 1], dtype=object)

    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                if i == 0:
                    d[i, i] = -mp.mpf(n * (n + 1)) / 2
                else:
                    d[i, i] = 1 / (1 - t[i - 1] ** 2)
            elif i == 0:
                d[i, j] = -((-1) ** n) * (1 - t[j - 1]) / (n * (1 + t[j - 1]) * p[j - 1, n - 1])
            elif j == 0:
                d[i, j] = (-1) ** n * n * p[i - 1, n - 1] / (1 - t[i - 1] ** 2)
            else:
                d[i, j] = (
                    (1 - t[j - 1])
                    * p[i - 1, n - 1]
                    / ((1 - t[i - 1]) * p[j - 1, n - 1] * (t[i - 1] - t[j - 1]))
                )

    # endpoint extrapolation
    b1: MPArray = np.zeros([n + 1], dtype=object)
    b1[0] = mp.mpf((-1) ** n)
    b1[1:] = 2 / (p[:, n - 1] * n)

    t_: MPArray = np.zeros([n + 1], dtype=object)
    t_[0] = mp.mpf(-1)
    t_[1:] = t

    return t_, w, d, b1
