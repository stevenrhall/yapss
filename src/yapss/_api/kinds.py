"""

What the elements of a `Vector` instance mean.

A `Vector` subclass such as ``Rocket`` declares field names and block sizes and nothing else.
The same declaration is reused for every aspect of the vector it names -- bounds, guess,
callback output rows, callback input rows -- and each of those reads and writes a *different*
kind of element. A `Kind` says which: what an element may be, how it is stored, and how it is
described when a user writes something else.

Each kind is a class used as a namespace; none is instantiated. `Vector` holds one on the
generated subclass it makes per (declaration, kind) pair, so the check is resolved by the class
rather than by a branch on every write.

"""

from __future__ import annotations

import math
from typing import Any, TypeGuard

import numpy as np

from .sampled import Interp

__all__ = ["Bounds", "Guess", "Kind", "ReadOnlyRows", "Rows", "ScalarGuess", "Scale"]

PAIR = 2
"""The length of a bound or two-point guess tuple."""


def is_bool(value: object) -> bool:
    """Report whether `value` is a Python or NumPy boolean.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True if `value` is a boolean.
    """
    return isinstance(value, bool | np.bool_)


def is_real(value: object) -> TypeGuard[float]:
    """Report whether `value` is a real number that YAPSS accepts as a magnitude.

    Booleans are excluded. Python treats `bool` as an `int`, so ``(2, False)`` would otherwise
    read as the interval ``(2, 0)``; a bool in a numeric position is nearly always a comparison
    that was meant to be a value.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True if `value` is a real number and not a boolean.
    """
    if is_bool(value):
        return False
    return isinstance(value, int | float | np.integer | np.floating)


def is_sequence(value: object) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    """Report whether `value` is a sequence of the kind an element or a row list is written as.

    A list or a tuple, and nothing else: a string is not a sequence of values here, and neither
    is an `Interp`, which is one element however many rows it carries.
    """
    return isinstance(value, list | tuple)


def is_pair(value: object) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    """Report whether `value` is a two-element list or tuple.

    The bracket type is deliberately not consulted. ``(0, 1)`` and ``[0, 1]`` are the same
    bound, and nothing in the language makes one of them mean "one value" and the other "one
    per row".
    """
    return is_sequence(value) and len(value) == PAIR


def _bool_message(label: str, name: str) -> str:
    return (
        f"{label} '{name}': a boolean is not a number. If this came from a comparison, "
        f"the comparison is probably the mistake."
    )


class Kind:
    """Base of the element kinds. Subclasses are namespaces, never instantiated.

    Attributes
    ----------
    by_row : bool
        True when each row of a block field is stored separately, so that a single row can be
        written by position. False when one value covers every row of the field.
    positional : bool
        True when integer and slice writes are accepted.
    slice_read : bool
        True when a slice read is accepted, returning an array of the rows covered.
    per_row : bool
        True when a *list* of values may be given for a block field, one per row, as against a
        single value covering every row.
    read_only : bool
        True when the user may not write at all.
    default : Any
        The value a field takes when it is never assigned, or `MISSING` when reading an
        unassigned field is an error.
    """

    by_row: bool = False
    per_row: bool = False
    positional: bool = False
    slice_read: bool = False
    read_only: bool = False
    default: Any = None

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is one element rather than a sequence of them.

        Parameters
        ----------
        value : object
            The value to test.

        Returns
        -------
        bool
            True if `value` is a single element of this kind.
        """
        raise NotImplementedError

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one element and return the value to store.

        Parameters
        ----------
        value : object
            The value written by the user.
        label : str
            What to call this vector in a message, such as ``"phase 'boost' state bounds"``.
        name : str
            The field being written.
        npoints : int or None
            The number of time points a row must cover, or None when it is not known.

        Returns
        -------
        Any
            The value to store.
        """
        raise NotImplementedError


MISSING: Any = object()
"""Sentinel: reading this field before it is assigned is an error."""


class Bounds(Kind):
    """A bound: a ``(lower, upper)`` pair, whose sides are each a number or None.

    A bare number is *not* a bound. An interval is a pair, and a field has rows, so a number
    standing for a bound would collide with a row count -- a two-row field given ``[0.0, 1.0]``
    would be either two fixed values or one interval, with nothing in the values to say which.
    Requiring the pair is what lets depth tell one element from a sequence of them, and it
    costs one ``(x, x)`` at each fixed endpoint.

    The brackets carry nothing: ``(0, 1)`` and ``[0, 1]`` are the same bound, and
    ``[(0, 1), (2, 3)]`` and ``((0, 1), (2, 3))`` are the same two.

    A value is stored normalized as a ``(lower, upper)`` pair of floats, with infinities for
    the free sides, which is what the solver is given.
    """

    by_row = False
    per_row = True
    default = (-math.inf, math.inf)

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single bound. See `Kind.is_element`."""
        return is_pair(value) and all(side is None or is_real(side) for side in value)

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one bound and return it as ``(lower, upper)``. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if is_real(value):
            msg = (
                f"{label} '{name}': a bound is a pair, and {value!r} is one number. To fix the "
                f"value, write ({value!r}, {value!r}); for an interval, write its two ends."
            )
            raise TypeError(msg)
        if value is None:
            msg = (
                f"{label} '{name}': a bound is a pair. For no bound at either end, write "
                f"(None, None)."
            )
            raise TypeError(msg)
        if not is_sequence(value):
            msg = f"{label} '{name}': a bound is a (lower, upper) pair; got {value!r}"
            raise TypeError(msg)
        pair = tuple(value)
        if len(pair) != PAIR:
            msg = (
                f"{label} '{name}': a bound is a (lower, upper) pair; got {len(pair)} "
                f"values, {value!r}"
            )
            raise ValueError(msg)
        lower, upper = (
            cls._side(side, index=i, label=label, name=name) for i, side in enumerate(pair)
        )
        # Refused here rather than by the solver, which reports them against its own vector
        # and so names no field: an infinite side on the wrong end leaves nothing feasible.
        if lower == math.inf:
            msg = (
                f"{label} '{name}': a lower bound of +inf leaves nothing feasible; for no lower "
                f"bound, write None."
            )
            raise ValueError(msg)
        if upper == -math.inf:
            msg = (
                f"{label} '{name}': an upper bound of -inf leaves nothing feasible; for no upper "
                f"bound, write None."
            )
            raise ValueError(msg)
        if lower > upper:
            msg = f"{label} '{name}': lower {lower} > upper {upper}"
            raise ValueError(msg)
        return (lower, upper)

    @classmethod
    def _side(cls, side: object, *, index: int, label: str, name: str) -> float:
        """Return one side of a bound as a float; None means that side is unbounded."""
        if is_bool(side):
            raise TypeError(_bool_message(label, name))
        if side is None:
            return -math.inf if index == 0 else math.inf
        if is_real(side):
            value = float(side)
            if math.isnan(value):
                msg = (
                    f"{label} '{name}': a side of a bound cannot be NaN; for no bound on that "
                    f"side, write None."
                )
                raise ValueError(msg)
            return value
        msg = f"{label} '{name}': each side of a bound is a number or None; got {side!r}"
        raise TypeError(msg)


class Guess(Kind):
    """A state or control guess: a ``(first, last)`` pair, or `yapss.interp`.

    The pair is linear in the phase's independent variable from one end to the other, so a
    constant is a pair whose ends agree. A bare number is refused for the same reason a bare
    number is not a bound: the element of this aspect is a pair, and a number standing for one
    would collide with a row count (see `Bounds`). ``guess.v = (500.0, 500.0)`` is a speed held
    at 500.

    `yapss.interp` gives samples on a grid of the field's own, and is one element however many
    rows of samples it carries.

    A field that is never assigned is guessed as zero.
    """

    by_row = False
    per_row = True
    default = ("constant", 0.0)

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single guess. See `Kind.is_element`."""
        if isinstance(value, Interp):
            return True
        return is_pair(value) and all(is_real(side) for side in value)

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one guess element. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if isinstance(value, Interp):
            return ("sampled", value)
        if is_real(value):
            msg = (
                f"{label} '{name}': a guess is a (first, last) pair, and {value!r} is one "
                f"number. To hold it there, write ({value!r}, {value!r})."
            )
            raise TypeError(msg)
        if not is_sequence(value):
            msg = (
                f"{label} '{name}': a guess is a (first, last) pair or yapss.interp(...); "
                f"got {value!r}"
            )
            raise TypeError(msg)
        pair = tuple(value)
        if len(pair) != PAIR:
            msg = f"{label} '{name}': a guess is (first, last); got {len(pair)} values, {value!r}"
            raise ValueError(msg)
        first, last = pair
        for side in (first, last):
            if not is_real(side):
                msg = f"{label} '{name}': a (first, last) guess takes two numbers; got {side!r}"
                raise TypeError(msg)
        return ("linear", float(first), float(last))


class Rows(Kind):
    """Rows of a callback output. An element is a scalar, or one value per time point.

    Only a true scalar broadcasts: any array is a sequence of rows, so that a per-point row is
    never mistaken for several rows when the point count happens to match the field's size.
    A field that is never assigned has no value, and reading it is an error.
    """

    by_row = True
    positional = True
    slice_read = True
    default = MISSING

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is one row rather than a sequence of rows.

        Only true scalars count: a Python or NumPy number, a 0-d array, or a symbolic value
        that has no shape.
        """
        if is_real(value):
            return True
        if isinstance(value, np.ndarray):
            return value.ndim == 0
        return not hasattr(value, "__len__")

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one row. See `Kind.check`.

        A row is nearly always a float array over the time points, or a plain number, and this
        runs once per row per callback call, so those two are recognized first. The checks are
        the same either way; only the order differs.
        """
        kind = type(value)
        if kind is np.ndarray:
            shape = value.shape  # type: ignore[attr-defined]
            if len(shape) == 1 and (npoints is None or shape[0] == npoints):
                return value
        elif kind is float or kind is np.float64:
            return value
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if cls.is_element(value):
            return value
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                msg = (
                    f"{label} '{name}': a row is a scalar or one value per time point; got an "
                    f"array of shape {value.shape}"
                )
                raise ValueError(msg)
            if npoints is not None and value.shape[0] != npoints:
                msg = (
                    f"{label} '{name}': a row needs one value per time point, {npoints}; "
                    f"got {value.shape[0]}"
                )
                raise ValueError(msg)
            return value
        msg = (
            f"{label} '{name}': a row is a scalar or one value per time point; "
            f"got {type(value).__name__}"
        )
        raise TypeError(msg)


class ReadOnlyRows(Rows):
    """Rows the user reads but never writes: callback inputs and solution values."""

    read_only = True


class ScalarGuess(Kind):
    """A guess that is one number rather than a trajectory: an integral or a parameter.

    Unlike a state or control guess, there is nothing for a pair of values to interpolate
    between, so only a number is accepted. A field that is never assigned is guessed as zero.
    """

    by_row = False
    per_row = True
    default = 0.0

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single guess. See `Kind.is_element`."""
        return is_real(value)

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one guess. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if is_real(value):
            return float(value)
        if isinstance(value, tuple):
            msg = (
                f"{label} '{name}': this is guessed as one number, not a trajectory; "
                f"got {value!r}"
            )
            raise TypeError(msg)
        msg = f"{label} '{name}': must be a number; got {value!r}"
        raise TypeError(msg)


class Scale(Kind):
    """A scale factor: one positive number, used to condition the problem for the solver.

    A scale says how large a quantity typically is; it never changes what the problem means.
    In particular it cannot flip a sign, so it must be positive -- to maximize rather than
    minimize, set `problem.objective.sense`.
    """

    by_row = False
    per_row = True
    default = 1.0

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single scale. See `Kind.is_element`."""
        return is_real(value)

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one scale. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if not is_real(value):
            msg = f"{label} '{name}': a scale is a positive number; got {value!r}"
            raise TypeError(msg)
        scaled = float(value)
        if not math.isfinite(scaled):
            msg = (
                f"{label} '{name}': a scale must be a finite number; got {scaled}. It says how "
                f"large the quantity typically is, and the solver divides by it."
            )
            raise ValueError(msg)
        if scaled <= 0:
            msg = (
                f"{label} '{name}': a scale must be positive; got {scaled}. A scale conditions "
                f"the problem and never changes what it means; to maximize, set "
                f"problem.objective.sense."
            )
            raise ValueError(msg)
        return scaled
