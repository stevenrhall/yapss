"""

A function that ``"auto"`` cannot trace, used in a callback: `external`.

Under ``"auto"`` a callback is called once with symbols, and CasADi differentiates the graph
that results. A table lookup, a library, or compiled code has no graph. `external` wraps such a
function so that the callback can call it anyway, and everything around the call is still
traced exactly. The mechanism is `yapss._standin`; this module is what makes a wrapped function
take and return the values a callback works with, scalars and arrays of `SXW`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from yapss._standin import External as _External

from .wrapper import SXW, SXArray, _as_object_array, is_symbolic

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["External", "external"]


class External(_External):
    """A wrapped function. Created by `external`, never directly.

    Called with numbers it is the function itself, broadcast over arrays. Called with symbols,
    while a callback is traced, it returns the quadratic that stands in for it, one for each
    element of the broadcast arguments.
    """

    def __call__(self, *args: Any) -> Any:
        """Return the function's value, or its stand-in where an argument is a symbol."""
        if not args:
            msg = f"{self._name}() takes at least one argument"
            raise TypeError(msg)
        if not any(is_symbolic(arg) for arg in args):
            return self.numeric(*args)
        arrays = [_as_object_array(arg) for arg in args]
        shape = np.broadcast_shapes(*[array.shape for array in arrays])
        arrays = [np.broadcast_to(array, shape) for array in arrays]
        out = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            out[index] = SXW(self.standin(*[SXW(array[index])._value for array in arrays]))
        return out[()] if shape == () else out.view(SXArray)

    def standin(self, *args: Any) -> Any:
        """Return the stand-in, refusing a call outside a callback with the callback's words."""
        try:
            return super().standin(*args)
        except RuntimeError as error:
            msg = (
                f"{self._name}() was called with a symbolic value outside a callback. A "
                f'function wrapped by yapss.math.external takes symbols only while "auto" '
                f"traces a callback."
            )
            raise RuntimeError(msg) from error


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
            return External(
                inner,
                scale=scale,
                vectorized=vectorized,
                jacobian=jacobian,
                hessian=hessian,
                eps=eps,
                name=name,
            )

        return decorator

    return External(
        function,
        scale=scale,
        vectorized=vectorized,
        jacobian=jacobian,
        hessian=hessian,
        eps=eps,
        name=name,
    )
