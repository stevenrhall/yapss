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
    """Bounds. An element is None (free), a number (fixed), or a ``(lower, upper)`` tuple.

    A value is stored normalized as a ``(lower, upper)`` pair of floats, with infinities for
    the free sides, which is what the solver is given. One bound covers every row of a block
    field; per-row bounds are not yet supported.
    """

    by_row = False
    per_row = True
    default = (-math.inf, math.inf)

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single bound. See `Kind.is_element`."""
        if value is None or is_real(value):
            return True
        return (
            isinstance(value, tuple)
            and len(value) == PAIR
            and all(side is None or is_real(side) for side in value)
        )

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one bound and return it as ``(lower, upper)``. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if value is None:
            return (-math.inf, math.inf)
        if is_real(value):
            return (float(value), float(value))
        if isinstance(value, tuple):
            if len(value) != PAIR:
                msg = (
                    f"{label} '{name}': a bound tuple is (lower, upper); got {len(value)} "
                    f"values, {value!r}"
                )
                raise ValueError(msg)
            lower, upper = (
                cls._side(side, index=i, label=label, name=name) for i, side in enumerate(value)
            )
            if lower > upper:
                msg = f"{label} '{name}': lower {lower} > upper {upper}"
                raise ValueError(msg)
            return (lower, upper)
        msg = (
            f"{label} '{name}': must be a number, None, or a (lower, upper) tuple; "
            f"got {value!r}"
        )
        raise TypeError(msg)

    @classmethod
    def _side(cls, side: object, *, index: int, label: str, name: str) -> float:
        """Return one side of a bound as a float; None means that side is unbounded."""
        if is_bool(side):
            raise TypeError(_bool_message(label, name))
        if side is None:
            return -math.inf if index == 0 else math.inf
        if is_real(side):
            return float(side)
        msg = f"{label} '{name}': each side of a bound is a number or None; got {side!r}"
        raise TypeError(msg)


class Guess(Kind):
    """An initial guess. An element is a number (constant) or a ``(first, last)`` tuple.

    A constant holds over the phase; a pair is linear in time from the start of the phase to
    its end; `yapss.interp` gives samples on a grid of the field's own. A field that is never
    assigned is guessed as zero.
    """

    by_row = False
    per_row = True
    default = ("constant", 0.0)

    @classmethod
    def is_element(cls, value: object) -> bool:
        """Report whether `value` is a single guess. See `Kind.is_element`."""
        if is_real(value) or isinstance(value, Interp):
            return True
        return isinstance(value, tuple) and len(value) == PAIR

    @classmethod
    def check(cls, value: object, *, label: str, name: str, npoints: int | None) -> Any:
        """Validate one guess element. See `Kind.check`."""
        del npoints
        if is_bool(value):
            raise TypeError(_bool_message(label, name))
        if isinstance(value, Interp):
            return ("sampled", value)
        if is_real(value):
            return ("constant", float(value))
        if isinstance(value, tuple):
            if len(value) != PAIR:
                msg = (
                    f"{label} '{name}': a guess tuple is (first, last); got {len(value)} "
                    f"values, {value!r}"
                )
                raise ValueError(msg)
            first, last = value
            for side in (first, last):
                if not is_real(side):
                    msg = (
                        f"{label} '{name}': a (first, last) guess takes two numbers; "
                        f"got {side!r}"
                    )
                    raise TypeError(msg)
            return ("linear", float(first), float(last))
        msg = f"{label} '{name}': must be a number or a (first, last) tuple; got {value!r}"
        raise TypeError(msg)


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
        if scaled <= 0:
            msg = (
                f"{label} '{name}': a scale must be positive; got {scaled}. A scale conditions "
                f"the problem and never changes what it means; to maximize, set "
                f"problem.objective.sense."
            )
            raise ValueError(msg)
        return scaled
