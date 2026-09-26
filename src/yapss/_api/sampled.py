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

    def rows(self, size: int, label: str, name: str) -> list[Any]:
        """Return `size` rows of samples, broadcasting a single row across a block field.

        Parameters
        ----------
        size : int
            The number of rows the field holds.
        label : str
            What to call the field's owner in a message.
        name : str
            The field's name.

        Returns
        -------
        list of numpy.ndarray
            One array of sampled values per row.
        """
        values = self.values
        if values.ndim == 1:
            return [values] * size
        if values.shape[0] != size:
            msg = (
                f"{label} '{name}': interp values have {values.shape[0]} rows, but the field "
                f"holds {size}"
            )
            raise ValueError(msg)
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
        The guess, to be assigned to a guess aspect.
    """
    time_array = _samples(time, "time")
    value_array = _samples(values, "values")
    if time_array.ndim != 1 or time_array.size < 2:  # noqa: PLR2004
        msg = f"interp(time=) must be at least two increasing times; got {time!r}"
        raise ValueError(msg)
    if np.any(np.diff(time_array) <= 0):
        msg = "interp(time=) must be strictly increasing"
        raise ValueError(msg)
    if value_array.ndim not in (1, 2):
        msg = (
            f"interp(values=) must be one row, or one row per member; got shape {value_array.shape}"
        )
        raise ValueError(msg)
    if value_array.shape[-1] != time_array.size:
        msg = (
            f"interp: {time_array.size} times but {value_array.shape[-1]} values; there must be "
            f"one value per time"
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
    array = np.array(given)
    if array.dtype.kind not in "iuf":
        msg = f"interp({what}=) takes real numbers; got {given!r}"
        raise TypeError(msg)
    array = array.astype(float)
    bad = np.argwhere(~np.isfinite(array))
    if bad.size:
        # the first bad sample, by index: the whole argument could be thousands of numbers
        first = tuple(int(i) for i in bad[0])
        where = ", ".join(str(i) for i in first)
        msg = f"interp({what}=) must be finite; {what}[{where}] is {array[first]}"
        raise ValueError(msg)
    array.flags.writeable = False
    return array


def coverage_complaint(
    sampled: Interp, guess: tuple[float, float], label: str, name: str
) -> str | None:
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
    label : str
        What to call the field's owner in a message.
    name : str
        The field's name.

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
            f"{label} '{name}': samples start at {first}; the time guess starts at {t0} "
            f"(samples must reach {t0 + margin})"
        )
    if last < tf - margin:
        return (
            f"{label} '{name}': samples end at {last}; the time guess ends at {tf} "
            f"(samples must reach {tf - margin})"
        )
    return None
