"""

One conversion for every real-valued number a user assigns.

Bounds, guess values, and scale factors are all "real numbers the user supplies", and each
used to convert them its own way, through a bare ``np.array(value, dtype=float)``. That
accepts far more than it should: NumPy converts ``"1"`` to 1.0, ``True`` to 1.0, and
``1 + 1j`` to 1.0 with a warning, so a typo, a stray comparison, or a complex intermediate
became a silently plausible number. `real_array` and `real_scalar` are the one place that
decides what a real number is, so the three cannot drift apart.

**Accepted:** Python ``int`` and ``float``, NumPy integers and floats, and sequences and
arrays of them. Infinity is a float, so ``np.inf``, ``math.inf``, and ``float("inf")`` all
work where the caller allows them.

**Refused, with `TypeError` naming the attribute:** strings, *including numeric ones* ---
``"inf"`` is a string, ``np.inf`` is the number; ``bool``, since `True` as a bound or a
scale factor is a mistake rather than 1.0; complex values; and anything else (``None``,
objects, ragged sequences).

One case slips through and cannot be caught here: a *mixed* sequence such as ``[1, True]``,
which NumPy has already reduced to an integer array before this code sees it. Writing a
bool into an element of a stored array is the same problem, and both are for the checked
array of E2 part 2.
"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["real_array", "real_scalar"]


def _describe(value: Any) -> str:
    """Name what the user supplied, as specifically as helps them find it."""
    if isinstance(value, np.ndarray):
        return f"an array of dtype {value.dtype}"
    if isinstance(value, (list, tuple)):
        # name the first offending element: "a list containing a str" locates the typo in a
        # long guess far better than "a list" does
        for item in _flatten(value):
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item,
                (int, float, np.integer, np.floating),
            ):
                return f"a {type(value).__name__} containing a {type(item).__name__}"
        return f"a {type(value).__name__}"
    return f"a {type(value).__name__}"


def _flatten(value: Any) -> Any:
    """Yield the leaves of nested lists and tuples."""
    for item in value:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _reject(label: str, value: Any) -> None:
    """Raise `TypeError` naming the attribute and what arrived."""
    msg = (
        f"{label} must be a real number or a sequence of real numbers, got {_describe(value)}. "
        f"Strings, bools, and complex values are not converted; for infinity use np.inf."
    )
    raise TypeError(msg)


def real_scalar(value: Any, label: str) -> float:
    """Return ``value`` as a float, accepting only a real number.

    Parameters
    ----------
    value : Any
        The value the user assigned.
    label : str
        The attribute as the user spells it, such as ``scale.objective``.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        _reject(label, value)
    return float(value)


def real_array(
    value: Any,
    label: str,
    *,
    shape: tuple[int, ...] | None = None,
    finite: bool = False,
) -> NDArray[np.float64]:
    """Return ``value`` as a new float64 array, accepting only real numbers.

    The result is always a copy, so a stored array never aliases the caller's.

    Parameters
    ----------
    value : Any
        The value the user assigned.
    label : str
        The attribute as the user spells it, such as ``bounds.phase[0].state.lower``.
    shape : tuple[int, ...], optional
        The shape required, checked with a message naming the attribute and the shape.
    finite : bool, default False
        Whether to refuse NaN and infinity. Bounds allow infinity; guess values do not.
    """
    if isinstance(value, (str, bytes)):  # a string is a sequence, so check it before asarray
        _reject(label, value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):  # ragged sequences, and objects with no array form
        _reject(label, value)
    if not (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    ):  # bool_, complex, str_, bytes_, object_, datetimes, ...
        _reject(label, value)

    if shape is not None and array.shape != shape:
        one_dimensional = len(shape) == 1
        kind = "length" if one_dimensional else "shape"
        expected = str(shape[0]) if one_dimensional else str(shape)
        if array.ndim == 0:
            got = "a scalar"  # "got ()" is unreadable, and a scalar is a common mistake
        elif array.ndim == 1 and one_dimensional:
            got = str(array.shape[0])
        else:
            got = f"shape {array.shape}"
        msg = f"{label} must have {kind} {expected}, got {got}."
        raise ValueError(msg)

    result: NDArray[np.float64] = array.astype(np.float64)
    if finite and not np.all(np.isfinite(result)):
        bad = np.flatnonzero(~np.isfinite(result.reshape(-1)))
        msg = (
            f"{label} must be finite, but element {bad[0]} of "
            f"{'the flattened array' if result.ndim > 1 else 'it'} is "
            f"{result.reshape(-1)[bad[0]]}."
        )
        raise ValueError(msg)
    return result
