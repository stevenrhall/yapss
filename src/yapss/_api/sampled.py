"""

Guesses given as samples, on a grid of their own.

``yapss.interp(t, values)`` says that a field is guessed by the values it takes at the times
`t`, interpolated linearly between them. Each field carries its own sample times, so nothing
has to agree in length across statements -- the shared time grid that 0.3.0 required, and the
mistakes that came with it, are gone.

Where the samples do not reach the ends of the phase, the end values are held, as
`numpy.interp` does. That is bounded and never invents a trend. Samples that fall well short of
the phase are refused instead: see `coverage_complaint`.

"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Interp", "coverage_complaint", "interp"]

COVERAGE = 0.1
"""How far from each end of the phase the samples may stop, as a fraction of its duration."""


class Interp:
    """A field guessed by samples on its own grid. Made by `interp`, never directly.

    Attributes
    ----------
    time : numpy.ndarray
        The sample times, strictly increasing.
    values : numpy.ndarray
        The sampled values: one row for a plain field, or one row per member of a block field.
    """

    __slots__ = ("time", "values")

    def __init__(self, time: Any, values: Any) -> None:
        self.time = time
        self.values = values

    def __repr__(self) -> str:
        """Return a representation naming the number of samples."""
        return f"interp({len(self.time)} samples from {self.time[0]} to {self.time[-1]})"

    def rows(self, size: int) -> list[Any]:
        """Return `size` rows of samples, broadcasting a single row across a block field.

        Parameters
        ----------
        size : int
            The number of rows the field holds.

        Returns
        -------
        list of numpy.ndarray
            One array of sampled values per row.
        """
        values = self.values
        if values.ndim == 1:
            return [values] * size
        assert values.shape[0] == size, "the rows are checked where the guess is assigned"
        return [values[row] for row in range(size)]


def interp(time: Any, values: Any, /) -> Interp:
    """Guess a field by the values it takes at given times.

    Parameters
    ----------
    time : array_like
        The sample times, which must be strictly increasing.
    values : array_like
        The sampled values: one per time, or, for a block field, one row per member.

    Returns
    -------
    Interp
        The guess, to be assigned to a state's or a control's ``guess``.

    Raises
    ------
    TypeError
        If a time or a value is not a real number.
    ValueError
        If a time or a value is not finite, there are fewer than two times, the times do not
        increase, or the values are not one per time.
    """
    time_array = _samples(time, "times")
    value_array = _samples(values, "values")
    if time_array.ndim != 1 or time_array.size < 2:  # noqa: PLR2004
        shape = "none" if time_array.size == 0 else f"shape {time_array.shape}"
        msg = f"interp: the times are a sequence of two or more; got {shape}"
        raise ValueError(msg)
    if np.any(np.diff(time_array) <= 0):
        k = int(np.flatnonzero(np.diff(time_array) <= 0)[0])
        msg = (
            f"interp: the times must be strictly increasing; times[{k + 1}] is "
            f"{time_array[k + 1]}, after {time_array[k]}"
        )
        raise ValueError(msg)
    if value_array.ndim not in (1, 2):
        msg = (
            f"interp: the values are one row, or one row per member of a block field; got "
            f"shape {value_array.shape}"
        )
        raise ValueError(msg)
    if value_array.shape[-1] != time_array.size:
        msg = (
            f"interp: {time_array.size} times but {value_array.shape[-1]} values. Give one "
            f"value per time."
        )
        raise ValueError(msg)
    return Interp(time_array, value_array)


def _samples(given: Any, what: str) -> Any:
    """Return `given` as a new, read-only float array, refusing what is not real and finite.

    A copy, because an array the caller keeps could be changed after it was checked -- times
    that were increasing when given need not stay so. The type is checked before converting:
    ``np.asarray(..., dtype=float)`` would turn "2", True and None into 2.0, 1.0 and NaN, and
    drop the imaginary part of a complex number.
    """
    # the first bad sample is named by its index: the whole argument could be thousands of numbers
    array = np.array(given)
    if array.dtype.kind not in "iuf":
        msg = f"interp: the {what} must be real numbers; {_first_not_real(array, what)}"
        raise TypeError(msg)
    array = array.astype(float)
    bad = np.argwhere(~np.isfinite(array))
    if bad.size:
        first = tuple(int(i) for i in bad[0])
        where = ", ".join(str(i) for i in first)
        msg = f"interp: the {what} must be finite; {what}[{where}] is {array[first]}"
        raise ValueError(msg)
    array.flags.writeable = False
    return array


def _first_not_real(array: Any, what: str) -> str:
    """Return which sample is the first that is not a real number, and what it is."""
    if array.dtype.kind in "OUSb" or array.dtype.kind == "c":
        for index, value in np.ndenumerate(array):
            if isinstance(value, bool | np.bool_) or not isinstance(
                value, int | float | np.integer | np.floating
            ):
                where = ", ".join(str(i) for i in index)
                shown = value.item() if isinstance(value, np.generic) else value
                return f"{what}[{where}] is {shown!r}"
    return f"got an array of {array.dtype}"


def coverage_complaint(sampled: Interp, guess: tuple[float, float], path: str) -> str | None:
    """Return a complaint if the samples do not reach far enough into the phase.

    The samples must reach within `COVERAGE` of the phase's duration at each end, so that they
    span at least the interior of it. That catches wrong units, samples taken from another
    phase, and a time guess moved away from its samples, while allowing rounding and small
    moves of either end.

    Parameters
    ----------
    sampled : Interp
        The sampled guess.
    guess : tuple of float
        The phase's ``(t0, tf)`` time guess.
    path : str
        The guess's path, such as ``"phases.boost.state.h.guess"``.

    Returns
    -------
    str or None
        The complaint, or None if the samples reach far enough.
    """
    t0, tf = guess
    margin = COVERAGE * (tf - t0)
    first, last = float(sampled.time[0]), float(sampled.time[-1])
    if first > t0 + margin:
        return (
            f"{path}: the samples start at {first}, and the time guess at {t0}; they must "
            f"reach {t0 + margin}"
        )
    if last < tf - margin:
        return (
            f"{path}: the samples end at {last}, and the time guess at {tf}; they must "
            f"reach {tf - margin}"
        )
    return None
