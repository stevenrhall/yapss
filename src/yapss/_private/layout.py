"""

The layout of one phase of the transcribed NLP under each spectral method.

A spectral method fixes, for each phase, how many points the continuous functions are
evaluated at, how many state values are stored and in what order, where the boundary
states sit among them, which evaluation point each state defect reads, and whether the
method adds boundary defects (LG) or zero modes (LGL). Everything that indexes into the
NLP vectors needs these facts. :func:`phase_layout` is the one place they are derived;
consumers read the resulting :class:`PhaseLayout` instead of branching on the method.

For a phase with ``K`` segments and ``N`` collocation points in total, per state:

========================  ==========  ==========  =====================================
                          LGR         LGL         LG
========================  ==========  ==========  =====================================
evaluation points         N           N - K + 1   N
defect rows               N           N           N, plus K boundary defects
state values at times     N + 1       N - K + 1   N + K + 1
extra stored values       --          K zero      --
                                      modes
storage order             time order  time order  collocation points, then the K
                                                  segment starts, then the final point
x0, xf position           0, N        0, N - K    N, N + K
========================  ==========  ==========  =====================================

In every method the evaluation points are the first stored state values.

"""

# future imports
from __future__ import annotations

# standard imports
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, assert_never, get_args

# third party imports
import numpy as np

# package imports
from .types_ import SpectralMethod

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

__all__ = ["SPECTRAL_METHODS", "PhaseLayout", "phase_layout", "problem_layout"]

SPECTRAL_METHODS: tuple[SpectralMethod, ...] = get_args(SpectralMethod)
"""Valid spectral methods."""


@dataclass(frozen=True)
class PhaseLayout:
    """The layout of one phase: point counts, positions, and index maps.

    Attributes
    ----------
    method : str
        The spectral method.
    n_segments : int
        Number of mesh segments, ``K``.
    n_collocation : int
        Total collocation points, ``N``: one state defect row per state each.
    n_eval : int
        Points at which the continuous functions are evaluated, and so the number of
        control values, path rows, and integrand samples.
    n_time : int
        State values stored at time points, per state.
    n_zero_mode : int
        Zero-mode values stored after the time points, per state (LGL only).
    n_boundary_defect : int
        Boundary defect rows after the state defects, per state (LG only).
    x0_position, xf_position : int
        Positions of the initial and final state among the stored time points.
    defect_index : NDArray[np.intp]
        For each state defect row, the evaluation point whose dynamics it reads.
    time_order : NDArray[np.intp]
        The storage position of each state time point, in time order.
    trim_t0, trim_tf : int
        How many evaluation points, at the end and at the start respectively, lie on
        ``tau = +1`` and ``tau = -1``, where the sensitivity of time to ``t0`` and to
        ``tf`` vanishes.
    """

    method: SpectralMethod
    n_segments: int
    n_collocation: int
    n_eval: int
    n_time: int
    n_zero_mode: int
    n_boundary_defect: int
    x0_position: int
    xf_position: int
    defect_index: NDArray[np.intp]
    time_order: NDArray[np.intp]
    trim_t0: int
    trim_tf: int

    @property
    def n_state_storage(self) -> int:
        """All stored values per state: the time points, then the zero modes."""
        return self.n_time + self.n_zero_mode


@functools.cache
def _cached_layout(method: SpectralMethod, collocation_points: tuple[int, ...]) -> PhaseLayout:
    n_segments = len(collocation_points)
    n_collocation = sum(collocation_points)
    defect_index = np.arange(n_collocation)

    match method:
        case "lgr":
            n_eval = n_collocation
            n_time = n_collocation + 1
            n_zero_mode = n_boundary_defect = 0
            x0_position, xf_position = 0, n_collocation
            time_order = np.arange(n_time)
            trim_t0, trim_tf = 0, 1
        case "lgl":
            n_eval = n_collocation - n_segments + 1
            n_time = n_eval
            n_zero_mode, n_boundary_defect = n_segments, 0
            x0_position, xf_position = 0, n_time - 1
            # each segment's collocation points start at the previous segment's last point
            starts = np.cumsum([0, *(m - 1 for m in collocation_points[:-1])])
            defect_index = np.concatenate(
                [start + np.arange(m) for start, m in zip(starts, collocation_points, strict=True)],
            )
            time_order = np.arange(n_time)
            trim_t0, trim_tf = 1, 1
        case "lg":
            n_eval = n_collocation
            n_time = n_collocation + n_segments + 1
            n_zero_mode, n_boundary_defect = 0, n_segments
            x0_position, xf_position = n_collocation, n_collocation + n_segments
            # in time order: segment k's start value, then its collocation values; last, the
            # final value. Storage holds the collocation values first, then the segment
            # starts, then the final value.
            order: list[int] = []
            first = 0
            for k, m in enumerate(collocation_points):
                order.append(n_collocation + k)
                order.extend(range(first, first + m))
                first += m
            order.append(n_collocation + n_segments)
            time_order = np.array(order)
            trim_t0, trim_tf = 0, 0
        case _:
            assert_never(method)

    for array in (defect_index, time_order):
        array.flags.writeable = False  # shared by every caller of the cache
    return PhaseLayout(
        method=method,
        n_segments=n_segments,
        n_collocation=n_collocation,
        n_eval=n_eval,
        n_time=n_time,
        n_zero_mode=n_zero_mode,
        n_boundary_defect=n_boundary_defect,
        x0_position=x0_position,
        xf_position=xf_position,
        defect_index=defect_index,
        time_order=time_order,
        trim_t0=trim_t0,
        trim_tf=trim_tf,
    )


def phase_layout(method: SpectralMethod, collocation_points: Sequence[int]) -> PhaseLayout:
    """Return the layout of a phase with these collocation points under `method`.

    Pure and cached: the same arguments always return the same record.
    """
    return _cached_layout(method, tuple(int(m) for m in collocation_points))


def problem_layout(problem: yapss.Problem) -> tuple[PhaseLayout, ...]:
    """Return the layout of every phase of `problem`, for its current mesh and method."""
    return tuple(
        phase_layout(problem.spectral_method, mesh_phase.collocation_points)
        for mesh_phase in problem.mesh.phase
    )
