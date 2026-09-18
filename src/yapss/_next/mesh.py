"""

The mesh of one phase.

A mesh divides the phase's time interval into segments and says how many collocation points
each segment carries.

A mesh is one immutable value, assigned whole. That replaces the two parallel lists of 0.3.0,
where a fraction list and a points list had to be kept the same length by hand.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Mesh"]

MIN_POINTS = 2
"""The fewest collocation points a segment may have."""

FRACTION_TOLERANCE = 1e-9
"""How far the segment fractions may stray from summing to 1."""


@dataclass(frozen=True, slots=True)
class Mesh:
    """The segments of one phase, as ``(fraction, collocation points)`` pairs.

    Parameters
    ----------
    segments : sequence of (float, int)
        One pair per segment. The fractions are positive and sum to 1.
    """

    segments: tuple[tuple[float, int], ...]

    def __init__(self, segments: Any) -> None:
        checked = tuple(self._check(segments))
        object.__setattr__(self, "segments", checked)

    @staticmethod
    def _check(segments: Any) -> list[tuple[float, int]]:
        try:
            pairs = list(segments)
        except TypeError:
            msg = f"Mesh segments are (fraction, points) pairs; got a {type(segments).__name__}"
            raise TypeError(msg) from None
        if not pairs:
            msg = "a mesh needs at least one segment"
            raise ValueError(msg)
        checked = []
        for index, pair in enumerate(pairs):
            try:
                fraction, points = pair
            except (TypeError, ValueError):
                msg = f"Mesh segment {index} is not a (fraction, points) pair; got {pair!r}"
                raise TypeError(msg) from None
            if not isinstance(points, int) or isinstance(points, bool):
                msg = f"Mesh segment {index}: collocation points must be an integer; got {points!r}"
                raise TypeError(msg)
            if points < MIN_POINTS:
                msg = (
                    f"Mesh segment {index}: a segment needs at least {MIN_POINTS} collocation "
                    f"points; got {points}"
                )
                raise ValueError(msg)
            fraction = float(fraction)
            if fraction <= 0:
                msg = f"Mesh segment {index}: the fraction must be positive; got {fraction}"
                raise ValueError(msg)
            checked.append((fraction, points))
        total = sum(fraction for fraction, _ in checked)
        if abs(total - 1.0) > FRACTION_TOLERANCE:
            msg = f"the mesh fractions must sum to 1; they sum to {total}"
            raise ValueError(msg)
        return checked

    @classmethod
    def uniform(cls, segments: int = 10, points: int = 10) -> Mesh:
        """Return a mesh of `segments` equal segments of `points` collocation points each.

        Parameters
        ----------
        segments : int, default 10
            The number of segments.
        points : int, default 10
            The collocation points in each segment.

        Returns
        -------
        Mesh
            The mesh.
        """
        if not isinstance(segments, int) or isinstance(segments, bool) or segments < 1:
            msg = f"Mesh.uniform(segments=) must be a positive integer; got {segments!r}"
            raise ValueError(msg)
        return cls([(1.0 / segments, points)] * segments)

    @property
    def fractions(self) -> tuple[float, ...]:
        """Return the fraction of the phase each segment spans."""
        return tuple(fraction for fraction, _ in self.segments)

    @property
    def collocation_points(self) -> tuple[int, ...]:
        """Return the collocation points in each segment."""
        return tuple(points for _, points in self.segments)
