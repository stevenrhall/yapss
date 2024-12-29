"""

Provides functions and classes for LGL and LGR integration and differentiation.

* The functions :meth:`lgl` and :meth:`lgr` compute the Legendre-Gauss_Lobatto (LGL)
  and Legendre-Gauss-Radau (LGR) collocation points, quadrature weights, and
  derivative matrix. The functions are memoized to avoid repeated calculations.

* :class:`Mesh` instances represent the mesh structure of the NLP

"""

# future imports
from __future__ import annotations

__all__ = ["lgl", "lgr"]

# standard imports
from collections.abc import Callable
from functools import wraps

# third party imports
import mpmath
import numpy as np
from mpmath import mp
from numpy.typing import NDArray

# typing
Array = NDArray[np.float64]
MPArray = NDArray[np.object_]

LG_func = Callable[[int], tuple[Array, ...]]

mp.dps = 30


def memoize(f: LG_func) -> LG_func:
    """Memoize the `lgl` and `lgr` functions.

    We could use `functools.cache`, except that the returned tuple has mutable arrays, and
    so copies of the arrays are returned instead.

    Parameters
    ----------
    f : Callable[[int], tuple[NDArray, ...]]
        decorated function (`lgl` or `lgr`)

    Returns
    -------
    Callable[[int], tuple[NDArray, ...]]
        memoized function
    """
    memo = {}

    @wraps(f)
    def function(n: int) -> tuple[Array, ...]:
        if n not in memo:
            memo[n] = f(n)
        return tuple(item.copy() for item in memo[n])

    return function


@memoize
def lgl(
    n: int,
) -> tuple[Array, Array, Array, Array]:
    r"""Compute the LGL collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Should have :math:`n \ge 3`

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
    min_collocation_points = 3
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 3."
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
            fj = mp.mpf(j)
            p[:, j] = ((2 * fj - 1) * t * p[:, j - 1] - (fj - 1) * p[:, j - 2]) / fj

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

    d0 = n * (n - 1) / 4 * p[:, n - 1]
    return (
        np.array(t, dtype=float),
        np.array(w, dtype=float),
        np.array(d, dtype=float),
        np.array(d0, dtype=float),
    )


@memoize
def lgr(
    n: int,
) -> tuple[Array, Array, Array, Array]:
    r"""Compute the LGR collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Should have :math:`n \ge 3`

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
    min_collocation_points = 3
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 3."
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

    return (
        np.array(t, dtype=float),
        np.array(w, dtype=float),
        np.array(d, dtype=float),
        np.array(b, dtype=float),
    )


@memoize
def lg(
    n: int,
) -> tuple[Array, Array, Array, Array]:
    r"""Compute the LGL collocation points, quadrature weights, and derivative matrix.

    Parameters
    ----------
    n : int
        The number of collocation points. Should have :math:`n \ge 3`

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
    min_collocation_points = 3
    if n < min_collocation_points:
        msg = "The number of collocation points (n) must be at least 3."
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
            fj = mp.mpf(j)
            p[:, j] = ((2 * fj - 1) * t * p[:, j - 1] - (fj - 1) * p[:, j - 2]) / fj

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
                    d[i, i] = -n * (n + 1) / 2
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
    b1: Array = np.zeros([n + 1], dtype=float)
    b1[0] = (-1) ** n
    b1[1:] = 2 / (p[:, n - 1] * n)

    t_: Array = np.zeros([n + 1], dtype=float)
    t_[0] = -1
    t_[1:] = t

    w = np.array(w, dtype=float)
    d = np.array(d, dtype=float)

    return t_, np.array(w, dtype=float), np.array(d, dtype=float), b1
