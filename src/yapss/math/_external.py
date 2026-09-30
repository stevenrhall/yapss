"""

A function that ``"auto"`` cannot trace, used in a callback: `external`.

Under ``"auto"`` a callback is called once with symbols, and CasADi differentiates the graph
that results. A table lookup, a library, or compiled code has no graph. `external` wraps such a
function so that the callback can call it anyway, and everything around the call is still
traced exactly.

In the trace, the wrapped function ``f(a)`` is replaced by a quadratic in its arguments,

    y0 + g . (a - a0) + (a - a0)' H (a - a0) / 2

whose coefficients ``a0``, ``y0``, ``g`` and ``H`` are extra symbols, one set for each use. The
quadratic agrees with ``f`` to second order at ``a0``, so the first and second derivatives
CasADi takes of the traced graph are those of the real model wherever ``a0`` is the value of
the arguments; and a composition of functions that each agree to second order agrees to
second order, so one wrapped function may feed another. Whenever the traced functions are
evaluated, the coefficients are computed first, for every point at once: the arguments by
evaluating the graph up to the call, and the value, gradient and Hessian by calling the wrapped
function, and its derivatives or differences of it, on whole arrays. `coefficients` does that,
for the ``"auto"`` derivative functions.

"""

from __future__ import annotations

import contextlib
from itertools import combinations
from typing import TYPE_CHECKING, Any, overload

import casadi as ca
import numpy as np

from yapss._backend.difference_steps import difference_steps

from .wrapper import SXW, SXArray, _as_object_array, is_symbolic

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from numpy.typing import NDArray

    Array = NDArray[np.float64]

__all__ = ["External", "StandIn", "coefficients", "external", "tracing"]

# the uses recorded by each trace in progress, innermost last
_TRACES: list[list[StandIn]] = []


@contextlib.contextmanager
def tracing() -> Iterator[list[StandIn]]:
    """Collect the uses of wrapped functions made while a callback is traced.

    Yields
    ------
    list of StandIn
        The uses, in the order the callback made them, which is an order in which each one's
        arguments depend only on the ones before it.
    """
    uses: list[StandIn] = []
    _TRACES.append(uses)
    try:
        yield uses
    finally:
        _TRACES.pop()


def _pairs(m: int) -> list[tuple[int, int]]:
    """Return the entries of a symmetric ``m`` by ``m`` matrix that a stand-in carries."""
    return [(i, i) for i in range(m)] + list(combinations(range(m), 2))


class StandIn:
    """One use of a wrapped function in a trace: its arguments, and the quadratic in its place.

    Parameters
    ----------
    function : External
        The wrapped function.
    arguments : list of casadi.SX
        The expressions the function was called with.
    index : int
        The position of the use in its trace, which names its symbols.
    """

    def __init__(self, function: External, arguments: list[Any], index: int) -> None:
        m = len(arguments)
        self.function = function
        self.arguments = ca.vertcat(*arguments)
        a0 = ca.SX.sym(f"a0_{index}", m)
        y0 = ca.SX.sym(f"y0_{index}")
        g = ca.SX.sym(f"g_{index}", m)
        pairs = _pairs(m)
        h = ca.SX.sym(f"h_{index}", len(pairs))
        self.symbols = ca.vertcat(a0, y0, g, h)
        d = self.arguments - a0
        value = y0 + ca.dot(g, d)
        for n, (i, j) in enumerate(pairs):
            value += h[n] * d[i] * d[j] * (0.5 if i == j else 1.0)
        self.value = value

    def rows(self, a: Array, order: int) -> Array:
        """Return the coefficients at every point: the rows ``a0``, ``y0``, ``g`` and ``h``.

        Parameters
        ----------
        a : numpy.ndarray
            The arguments, one row for each and one column for each point.
        order : int
            The highest derivative of the traced graph that is wanted. Coefficients it does
            not reach are left zero, and the wrapped function is not asked for them.

        Returns
        -------
        numpy.ndarray
            One row for each coefficient symbol and one column for each point.
        """
        m, width = a.shape
        y0, g, h = self.function.derivatives(a, order)
        rows = np.zeros((self.symbols.shape[0], width))
        rows[:m] = a
        rows[m] = y0
        if g is not None:
            rows[m + 1 : 2 * m + 1] = g
        if h is not None:
            rows[2 * m + 1 :] = h
        return rows


def coefficients(
    uses: Sequence[StandIn], stages: Sequence[Any], points: Array, order: int
) -> Array:
    """Return the coefficients of every stand-in in a traced function, at every point.

    Parameters
    ----------
    uses : sequence of StandIn
        The stand-ins, in trace order.
    stages : sequence of casadi.Function
        For each stand-in, the function of the traced variables and the coefficients that
        returns its arguments.
    points : numpy.ndarray
        The traced variables, one column for each point.
    order : int
        The highest derivative of the traced function that is wanted.

    Returns
    -------
    numpy.ndarray
        The coefficients, stacked in trace order, one column for each point.
    """
    sizes = [use.symbols.shape[0] for use in uses]
    table = np.zeros((sum(sizes), points.shape[1]))
    row = 0
    for use, stage, size in zip(uses, stages, sizes, strict=True):
        # the stand-ins before this one are already filled in, which is all its arguments read
        a = stage(points, table).full()
        table[row : row + size] = use.rows(a, order)
        row += size
    return table


class External:
    """A wrapped function. Created by `external`, never directly.

    Called with numbers it is the function itself, broadcast over arrays. Called with symbols,
    while a callback is traced, it returns the quadratic that stands in for it.
    """

    def __init__(  # noqa: PLR0913  -- one parameter for each option external() takes
        self,
        function: Callable[..., Any],
        *,
        name: str,
        scale: Sequence[float] | float | None,
        vectorized: bool,
        jacobian: Callable[..., Any] | None,
        hessian: Callable[..., Any] | None,
        eps: float | None,
    ) -> None:
        self._function = function
        self._name = name
        self._scale = scale
        self._vectorized = vectorized
        self._jacobian = jacobian
        self._hessian = hessian
        self._eps = eps
        self._checked = not vectorized
        self.__doc__ = function.__doc__
        self.__name__ = name

    def __repr__(self) -> str:
        """Return the wrapped function's name."""
        return f"<external {self._name}>"

    def __call__(self, *args: Any) -> Any:
        """Return the function's value, or its stand-in where an argument is a symbol."""
        if not args:
            msg = f"{self._name}() takes at least one argument"
            raise TypeError(msg)
        if not any(is_symbolic(arg) for arg in args):
            return self._numeric(args)
        if not _TRACES:
            msg = (
                f"{self._name}() was called with a symbolic value outside a callback. A "
                f'function wrapped by yapss.math.external takes symbols only while "auto" '
                f"traces a callback."
            )
            raise RuntimeError(msg)
        uses = _TRACES[-1]
        arrays = [_as_object_array(arg) for arg in args]
        shape = np.broadcast_shapes(*[array.shape for array in arrays])
        arrays = [np.broadcast_to(array, shape) for array in arrays]
        out = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            use = StandIn(self, [SXW(array[index])._value for array in arrays], len(uses))
            uses.append(use)
            out[index] = SXW(use.value)
        return out[()] if shape == () else out.view(SXArray)

    def _numeric(self, args: Sequence[Any]) -> Any:
        """Return the function's value at numbers, broadcast as a ufunc would."""
        arrays = np.broadcast_arrays(*[np.asarray(arg, dtype=float) for arg in args])
        shape = arrays[0].shape
        values = self.values(np.array([array.reshape(-1) for array in arrays]))
        return values.reshape(shape)[()]

    def values(self, a: Array) -> Array:
        """Return the function's value at every column of `a`.

        Parameters
        ----------
        a : numpy.ndarray
            The arguments, one row for each and one column for each point.

        Returns
        -------
        numpy.ndarray
            The values, one for each point.
        """
        width = a.shape[1]
        if self._vectorized:
            raw = np.asarray(self._function(*a))
            if raw.shape != (width,):
                msg = (
                    f"{self._name}() was given arrays of {width} values and returned an array "
                    f"of shape {raw.shape}. A function wrapped with vectorized=True returns "
                    f"one value for each element of its arguments."
                )
                raise ValueError(msg)
        else:
            raw = np.array([self._function(*map(float, column)) for column in a.T])
            if raw.shape != (width,):
                msg = (
                    f"{self._name}() returned a value of shape {raw.shape[1:]}. A function "
                    f"wrapped by yapss.math.external returns one number."
                )
                raise ValueError(msg)
        if self._eps is None:
            # the precision of what the function returns, which is single if it says so
            dtype = raw.dtype if np.issubdtype(raw.dtype, np.floating) else np.float64
            self._eps = float(np.finfo(dtype).eps) / 2
        values = raw.astype(float)
        if not self._checked and width > 1:
            self._checked = True
            self._check_elementwise(a, values)
        return values

    def _check_elementwise(self, a: Array, values: Array) -> None:
        """Refuse a function declared vectorized that is not elementwise, once."""
        single = np.array(
            [np.asarray(self._function(*column[:, None])).reshape(-1)[0] for column in a.T],
            dtype=float,
        )
        assert self._eps is not None
        tolerance = 1e4 * self._eps * (1 + np.abs(single))
        disagree = ~(np.abs(values - single) <= tolerance) & ~(np.isnan(values) & np.isnan(single))
        if disagree.any():
            k = int(np.flatnonzero(disagree)[0])
            msg = (
                f"{self._name}() is wrapped with vectorized=True, but called with arrays it "
                f"returns {values[k]!r} at element {k}, where called with that element alone "
                f"it returns {single[k]!r}. A vectorized function must work element by "
                f"element. Drop vectorized=True to have it called one point at a time."
            )
            raise ValueError(msg)

    def _steps(self, m: int, which: int) -> Array:
        """Return the step in each argument for a first (0) or second (1) difference."""
        assert self._eps is not None
        scale = np.broadcast_to(np.asarray(1.0 if self._scale is None else self._scale), (m,))
        return difference_steps(self._eps)[which] * scale.astype(float)

    def _supplied(self, function: Callable[..., Any], a: Array, shape: tuple[int, ...]) -> Array:
        """Return a supplied derivative at every column of `a`, as `shape` plus the points."""
        width = a.shape[1]
        if self._vectorized:
            raw = np.asarray(function(*a), dtype=float)
            return raw.reshape((*shape, width))
        columns = [np.asarray(function(*map(float, column)), dtype=float) for column in a.T]
        return np.stack([column.reshape(shape) for column in columns], axis=-1)

    def derivatives(self, a: Array, order: int) -> tuple[Array, Array | None, Array | None]:
        """Return the value, gradient and Hessian entries at every column of `a`.

        Parameters
        ----------
        a : numpy.ndarray
            The arguments, one row for each and one column for each point.
        order : int
            0 for the value alone, 1 to add the gradient, 2 to add the Hessian.

        Returns
        -------
        tuple
            The value; the gradient, one row for each argument, or None; and the Hessian, one
            row for each entry of `_pairs`, or None.
        """
        m = a.shape[0]
        y0 = self.values(a)
        if order == 0:
            return y0, None, None

        def shifted(step: Array, *moves: tuple[int, int]) -> Array:
            b = a.copy()
            for i, sign in moves:
                b[i] += sign * step[i]
            return b

        if self._jacobian is not None:
            g = self._supplied(self._jacobian, a, (m,))
        else:
            d = self._steps(m, 0)
            g = np.array(
                [
                    (self.values(shifted(d, (i, +1))) - self.values(shifted(d, (i, -1))))
                    / (2 * d[i])
                    for i in range(m)
                ]
            )
        if order == 1:
            return y0, g, None

        pairs = _pairs(m)
        if self._hessian is not None:
            full = self._supplied(self._hessian, a, (m, m))
            h = np.array([full[i, j] for i, j in pairs])
        elif self._jacobian is not None:
            # a first difference of the supplied gradient, which is more accurate than a
            # second difference of the value
            d = self._steps(m, 0)
            columns = [
                (
                    self._supplied(self._jacobian, shifted(d, (j, +1)), (m,))
                    - self._supplied(self._jacobian, shifted(d, (j, -1)), (m,))
                )
                / (2 * d[j])
                for j in range(m)
            ]
            h = np.array([(columns[j][i] + columns[i][j]) / 2 for i, j in pairs])
        else:
            d = self._steps(m, 1)
            rows = []
            for i, j in pairs:
                if i == j:
                    plus, minus = self.values(shifted(d, (i, +1))), self.values(shifted(d, (i, -1)))
                    rows.append((plus - 2 * y0 + minus) / d[i] ** 2)
                else:
                    mixed = (
                        self.values(shifted(d, (i, +1), (j, +1)))
                        - self.values(shifted(d, (i, +1), (j, -1)))
                        - self.values(shifted(d, (i, -1), (j, +1)))
                        + self.values(shifted(d, (i, -1), (j, -1)))
                    )
                    rows.append(mixed / (4 * d[i] * d[j]))
            h = np.array(rows)
        return y0, g, h


@overload
def external(
    function: Callable[..., Any],
    /,
    *,
    scale: Sequence[float] | float | None = ...,
    vectorized: bool = ...,
    jacobian: Callable[..., Any] | None = ...,
    hessian: Callable[..., Any] | None = ...,
    eps: float | None = ...,
    name: str | None = ...,
) -> External: ...
@overload
def external(
    function: None = None,
    /,
    *,
    scale: Sequence[float] | float | None = ...,
    vectorized: bool = ...,
    jacobian: Callable[..., Any] | None = ...,
    hessian: Callable[..., Any] | None = ...,
    eps: float | None = ...,
    name: str | None = ...,
) -> Callable[[Callable[..., Any]], External]: ...
def external(  # noqa: PLR0913  -- every one is a named option, and all but one optional
    function: Callable[..., Any] | None = None,
    /,
    *,
    scale: Sequence[float] | float | None = None,
    vectorized: bool = False,
    jacobian: Callable[..., Any] | None = None,
    hessian: Callable[..., Any] | None = None,
    eps: float | None = None,
    name: str | None = None,
) -> Any:
    """Wrap a function of numbers that ``"auto"`` cannot trace, for use in a callback.

    The function takes one or more numbers and returns one: a table lookup, a library, or
    compiled code. Wrapped, it can be called in a callback with the callback's own values.
    Under ``"auto"`` everything around the call is traced and differentiated exactly, and the
    function's own derivatives are those supplied as `jacobian` and `hessian`, or differences
    of it in its own arguments alone. Under the central-difference methods the wrapper is
    the function.

    Parameters
    ----------
    function : callable, optional
        The function to wrap. Omit it to use `external` as a decorator with options.
    scale : float or sequence of float, optional
        The typical size of each argument, which the difference steps are proportional to.
        One number applies to every argument; the default is 1. It is given here because an
        argument is an expression of the problem's variables, not one of them, and so has no
        scale of its own.
    vectorized : bool, default False
        Whether the function takes arrays and works on them element by element, as NumPy's
        functions do. It is then called once for all the points of a phase, where otherwise it
        is called once for each. The claim is checked the first time the function is called
        with more than one point.
    jacobian : callable, optional
        The function's first derivative, taking the same arguments: one value for a function
        of one argument, and otherwise one for each argument. When `vectorized`, each is an
        array like the arguments.
    hessian : callable, optional
        The function's second derivative, taking the same arguments: one value for a function
        of one argument, and otherwise a square matrix of them.
    eps : float, optional
        The relative precision of the function's values, from which the difference steps are
        taken. The default is that of the numbers the function returns, double precision
        unless they are single. State it for a function whose values are less accurate than
        their type, as when they come from an iteration stopped at a tolerance.
    name : str, optional
        The name used in messages; the function's own by default.

    Returns
    -------
    callable
        The wrapped function, or a decorator that wraps one.

    Raises
    ------
    TypeError
        If `function` is not callable.
    ValueError
        If `eps` or a `scale` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from yapss.math import external
    >>> @external(scale=8500.0, vectorized=True)
    ... def density(h):
    ...     return 1.225 * np.exp(-h / 8500.0)
    >>> round(float(density(1000.0)), 6)
    1.089037
    """
    if function is None:

        def decorator(inner: Callable[..., Any]) -> External:
            return external(
                inner,
                scale=scale,
                vectorized=vectorized,
                jacobian=jacobian,
                hessian=hessian,
                eps=eps,
                name=name,
            )

        return decorator

    for label, value in (("function", function), ("jacobian", jacobian), ("hessian", hessian)):
        if value is not None and not callable(value):
            # a checker reads the annotations and sees this cannot happen; a caller can still
            # do it, and the message is better here than at the first call
            msg = (  # type: ignore[unreachable]
                f"external(): {label} must be callable; got {value!r}"
            )
            raise TypeError(msg)
    if eps is not None and not eps > 0:
        msg = f"external(): eps must be positive; got {eps!r}"
        raise ValueError(msg)
    if scale is not None and not np.all(np.asarray(scale, dtype=float) > 0):
        msg = f"external(): scale must be positive; got {scale!r}"
        raise ValueError(msg)
    return External(
        function,
        name=name or getattr(function, "__name__", None) or repr(function),
        scale=scale,
        vectorized=vectorized,
        jacobian=jacobian,
        hessian=hessian,
        eps=eps,
    )
