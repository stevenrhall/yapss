"""

Provides a set of functions that are compatible with both NumPy and CasADi.

"""

from typing import Any, Callable

import casadi as ca
import numpy as np

from .wrapper import SXW

__all__ = ["arctan2", "hypot", "maximum", "minimum", "power", "sign"]


def vectorized_two_arg_func(
    func: Callable[..., Any],
    casadi_func: Callable[..., Any],
) -> Callable[[Any, Any], Any]:
    """
    Wrap a two-argument function to handle both SXW and numeric inputs.

    Parameters
    ----------
    func : Callable
        A numeric function that takes two arguments (e.g., np.arctan2 or np.maximum).
    casadi_func : Callable
        A CasADi-compatible function for SXW inputs (e.g., ca.arctan2 or ca.fmax).

    Returns
    -------
    Callable
        A vectorized version of the function that applies element-wise and handles mixed types.
    """

    def _elementwise_func(x: Any, y: Any) -> Any:
        if isinstance(x, SXW) or isinstance(y, SXW):
            x_value = x._value if isinstance(x, SXW) else x
            y_value = y._value if isinstance(y, SXW) else y
            return SXW(casadi_func(x_value, y_value))

        return func(x, y)

    vectorized_func = np.vectorize(_elementwise_func)

    def wrapped_func(x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            np.array(x, dtype=np.float64)
            np.array(y, dtype=np.float64)
            return func(x, y, *args, **kwargs)
        except TypeError:
            pass

        result = vectorized_func(x, y, *args, **kwargs)
        return result.item() if result.ndim == 0 else result

    return wrapped_func


def vectorized_one_arg_func(
    func: Callable[..., Any],
    casadi_func: Callable[..., Any],
) -> Callable[[Any], Any]:
    """
    Wrap a one-argument function to handle both SXW and numeric inputs.

    Parameters
    ----------
    func : Callable
        A numeric function that takes one argument (e.g., np.sqrt or np.sin).
    casadi_func : Callable
        A CasADi-compatible function for SXW inputs (e.g., ca.sqrt or ca.sin).

    Returns
    -------
    Callable
        A vectorized version of the function that applies element-wise and handles mixed types.
    """

    def _elementwise_func(x: Any) -> Any:
        if isinstance(x, SXW):
            x_value = x._value
            return SXW(casadi_func(x_value))

        return func(x)

    vectorized_func = np.vectorize(_elementwise_func)

    def wrapped_func(x: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            np.array(x, dtype=np.float64)
            return func(x, *args, **kwargs)
        except TypeError:
            pass
        result = vectorized_func(x, *args, **kwargs)
        return result.item() if result.ndim == 0 else result

    return wrapped_func


arctan2 = vectorized_two_arg_func(np.arctan2, ca.atan2)
maximum = vectorized_two_arg_func(np.maximum, ca.fmax)
minimum = vectorized_two_arg_func(np.minimum, ca.fmin)
power = vectorized_two_arg_func(np.power, ca.power)
hypot = vectorized_two_arg_func(np.hypot, ca.hypot)

sign = vectorized_one_arg_func(np.sign, ca.sign)
