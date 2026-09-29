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

A *mixed* sequence such as ``[1.0, True]`` would be reduced by NumPy to a float array with
no bool left in it, so the elements of a list or tuple are inspected before conversion.
Writing into an element of a stored array is checked by `checked_array.CheckedArray`,
which calls `real_array` on the value written.
"""

# future imports
from __future__ import annotations

# standard imports
import operator
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["integer_scalar", "integer_sequence", "real_array", "real_scalar"]


def _article(word: str) -> str:
    """Return "a" or "an" to suit the word, so messages read as English."""
    return "an" if word[:1].lower() in "aeiou" else "a"


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
                inner = type(item).__name__
                return (
                    f"{_article(type(value).__name__)} {type(value).__name__} "
                    f"containing {_article(inner)} {inner}"
                )
        return f"{_article(type(value).__name__)} {type(value).__name__}"
    return f"{_article(type(value).__name__)} {type(value).__name__}"


def _flatten(value: Any) -> Any:
    """Yield the leaves of nested lists and tuples."""
    for item in value:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _is_bool(item: Any) -> bool:
    """Return whether a sequence element is a bool, or an array of them."""
    return isinstance(item, (bool, np.bool_)) or (
        isinstance(item, np.ndarray) and item.dtype == np.bool_
    )


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
    if isinstance(value, (list, tuple)) and any(_is_bool(item) for item in _flatten(value)):
        _reject(label, value)  # NumPy would convert [1.0, True] to floats, losing the bool
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


def integer_sequence(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Return ``value`` as a tuple of ints, accepting any sequence of integers.

    Counts and collocation points are integers rather than measurements, so they take a
    different rule from `real_array`: anything `operator.index` accepts is an integer,
    which covers Python ``int``, NumPy integers, and any object that defines
    ``__index__``. A float is refused even when it is whole --- ``4.0`` collocation points
    is a mistake, not a rounding --- and so is ``bool``, which ``operator.index`` would
    otherwise accept as 0 or 1.

    Parameters
    ----------
    value : Any
        The value the user assigned.
    label : str
        The attribute as the user spells it, such as ``mesh.phase[0].collocation_points``.
    minimum : int, optional
        The smallest value each element may take, checked with a message naming the label.
    allow_empty : bool, default False
        Whether an empty sequence is allowed. A mesh needs at least one segment; a problem
        with no phases is written ``nx=[]``.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        msg = f"{label} must be a sequence of integers, got {_describe(value)}."
        raise TypeError(msg)
    items = list(value)
    if not items and not allow_empty:
        msg = f"{label} must have at least one element, got an empty sequence."
        raise ValueError(msg)
    result = []
    for i, item in enumerate(items):
        if isinstance(item, (bool, np.bool_)):
            msg = f"{label}[{i}] must be an integer, got the bool {item!r}."
            raise TypeError(msg)
        try:
            result.append(operator.index(item))
        except TypeError:
            kind = type(item).__name__
            msg = f"{label}[{i}] must be an integer, got {_article(kind)} {kind}, {item!r}."
            raise TypeError(msg) from None
    if minimum is not None:
        for i, item in enumerate(result):
            if item < minimum:
                msg = f"{label}[{i}] must be at least {minimum}, got {item}."
                raise ValueError(msg)
    return tuple(result)


def integer_scalar(value: Any, label: str) -> int:
    """Return ``value`` as an int, by the same rule as `integer_sequence`.

    Parameters
    ----------
    value : Any
        The value the user supplied.
    label : str
        The attribute or argument as the user spells it, such as ``ns``.
    """
    if isinstance(value, (bool, np.bool_)):
        msg = f"{label} must be an integer, got the bool {value!r}."
        raise TypeError(msg)
    try:
        return operator.index(value)
    except TypeError:
        kind = type(value).__name__
        msg = f"{label} must be an integer, got {_article(kind)} {kind}, {value!r}."
        raise TypeError(msg) from None
