"""
Module mesh.
"""

# future imports
from __future__ import annotations

# standard imports
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, assert_never

# third party imports
import numpy as np
from mpmath import mp
from scipy.sparse import csr_matrix, lil_matrix

# package imports
from .layout import phase_layout
from .quadrature import DIGITS, lg, lgl, lgr

if TYPE_CHECKING:
    # third party imports
    # package imports
    from collections.abc import Sequence

    from numpy.typing import NDArray

    class MeshLike(Protocol):
        """What `Mesh` needs of one phase: how it is divided, and how finely."""

        @property
        def fraction(self) -> Sequence[float]:
            """Return the fraction of the phase each segment spans."""

        @property
        def collocation_points(self) -> Sequence[int]:
            """Return the collocation points in each segment."""

    from .types_ import SpectralMethod

    # typing
    Array = NDArray[np.float64]
    MPArray = NDArray[np.object_]


class Mesh:
    """Mesh instances represent the mesh structure of the NLP.

    Sizes and orderings come from each phase's `layout.PhaseLayout`; this class computes
    the numerical quantities, which genuinely differ by spectral method.

    Attributes
    ----------
    d : Array
        Differentiation matrix acting on every stored state value, for each phase
    b_lg : Array
        Boundary-defect matrix acting on every stored state value, for each phase; it has a
        row per segment under LG and no rows under the other methods
    w : Array
        Quadrature weights at the evaluation points, for each phase
    tau_x : Array
        Mesh time of each state time point, in time order, for each phase
    tau_u : Array
        Mesh time of each evaluation point, for each phase
    phase : Sequence[MeshLike]
        The mesh phases of the problem
    """

    def __init__(self, mesh_phase: Sequence[MeshLike]) -> None:
        self.phase = mesh_phase
        self.d: list[NDArray[np.float64]] = []
        self.w: list[NDArray[np.float64]] = []
        self.tau_x: list[NDArray[np.float64]] = []
        self.tau_u: list[NDArray[np.float64]] = []
        self.b_lg: list[csr_matrix] = []

    def set_matrices(self, spectral_method: SpectralMethod) -> None:
        """Compute and save the matrices and vectors needed for integration and differentiation.

        Parameters
        ----------
        spectral_method : {"lg", "lgr", "lgl"}
        """
        match spectral_method:
            case "lgl":
                self.set_matrices_lgl()
            case "lgr":
                self.set_matrices_lgr()
            case "lg":
                self.set_matrices_lg()
            case _:
                assert_never(spectral_method)

    def set_matrices_lgr(self) -> None:
        """Compute and save the arrays for LGR integration and differentiation."""
        for phase in self.phase:
            col_points = phase.collocation_points
            layout = phase_layout("lgr", col_points)
            d = lil_matrix((layout.n_collocation, layout.n_state_storage), dtype=np.float64)
            with mp.workdps(DIGITS):
                fractions, starts = _segments(phase.fraction)
                w = np.full(layout.n_eval, mp.mpf(0), dtype=object)
                tau_x = np.full(layout.n_time, Fraction(0), dtype=object)
                i0 = 0
                for k, nc in enumerate(col_points):
                    tk, wk, dk, _ = _scaled("lgr", nc, fractions[k])
                    tau_x[i0 : i0 + nc] = tk + starts[k]
                    d[i0 : i0 + nc, i0 : i0 + nc + 1] = dk
                    w[i0 : i0 + nc] = wk
                    i0 += nc
                tau_x[-1] = Fraction(1)
                self._save(d, w, tau_x, tau_x[:-1])
            self.b_lg.append(csr_matrix((0, layout.n_state_storage), dtype=np.float64))

    def set_matrices_lgl(self) -> None:
        """Compute and save the arrays needed for LGL integration and differentiation."""
        for phase in self.phase:
            col_points = phase.collocation_points
            layout = phase_layout("lgl", col_points)
            nsegs = layout.n_segments
            d = lil_matrix((layout.n_collocation, layout.n_state_storage), dtype=np.float64)
            with mp.workdps(DIGITS):
                fractions, starts = _segments(phase.fraction)
                w = np.full(layout.n_eval, mp.mpf(0), dtype=object)
                tau_x = np.full(layout.n_time, Fraction(0), dtype=object)
                i0 = 0  # row (collocation) index
                j0 = 0  # column (state point) index
                for k, nc in enumerate(col_points):
                    tk, wk, dk, d0k = _scaled("lgl", nc, fractions[k])
                    # a point shared by two segments is the end of one and the start of the
                    # next: the same number either way, and it collects both weights
                    tau_x[j0 : j0 + nc] = tk + starts[k]
                    d[i0 : i0 + nc, j0 : j0 + nc] = dk
                    d[i0 : i0 + nc, layout.n_state_storage - nsegs + k] = d0k
                    w[j0 : j0 + nc] += wk
                    i0 += nc
                    j0 += nc - 1
                self._save(d, w, tau_x, tau_x)
            self.b_lg.append(csr_matrix((0, layout.n_state_storage), dtype=np.float64))

    def set_matrices_lg(self) -> None:
        """Compute and save the arrays needed for LG integration and differentiation."""
        for phase in self.phase:
            col_points = phase.collocation_points
            layout = phase_layout("lg", col_points)
            nsegs = layout.n_segments
            nxpoints = layout.n_time
            d = lil_matrix((layout.n_collocation, layout.n_state_storage), dtype=np.float64)
            bmat: Array = np.zeros([layout.n_boundary_defect, nxpoints], dtype=np.float64)
            with mp.workdps(DIGITS):
                fractions, starts = _segments(phase.fraction)
                w = np.full(layout.n_eval, mp.mpf(0), dtype=object)
                tau_x = np.full(nxpoints, Fraction(0), dtype=object)
                tau_u = np.full(layout.n_eval, Fraction(0), dtype=object)
                i0 = 0
                for k, nck in enumerate(col_points):
                    tk, wk, dk, bk = _scaled("lg", nck, fractions[k])
                    tau_x[i0 + k : i0 + k + nck + 1] = tk + starts[k]
                    tau_u[i0 : i0 + nck] = tk[1:] + starts[k]
                    d[i0 : i0 + nck, i0 : i0 + nck] = dk[1:, 1:]
                    d[i0 : i0 + nck, layout.n_state_storage - nsegs - 1 + k] = dk[1:, 0]
                    bmat[k, i0 : i0 + nck] = bk[1:]
                    bmat[k, -nsegs - 1 + k] = bk[0]
                    bmat[k, -nsegs + k] = -1.0
                    w[i0 : i0 + nck] = wk[1:]
                    i0 += nck
                tau_x[-1] = Fraction(1)
                self._save(d, w, tau_x, tau_u)
            self.b_lg.append(csr_matrix(bmat))

    def _save(self, d: lil_matrix, w: MPArray, tau_x: MPArray, tau_u: MPArray) -> None:
        """Round a phase's arrays to float, once, and save them.

        The grids are exact rationals, composed on [0, 1] and mapped to [-1, 1] here, so that
        the ends of the phase are exactly -1 and 1, a segment boundary is exactly where the
        fractions put it, and a symmetric mesh gives a symmetric grid.
        """
        csr = d.tocsr()
        csr.eliminate_zeros()
        self.d.append(csr)
        self.w.append(_rounded(w))
        grid_x = _rounded(2 * tau_x - 1)
        grid_u = _rounded(2 * tau_u - 1)
        assert grid_x[0] == -1.0 and grid_x[-1] == 1.0  # noqa: PT018 -- exact by construction
        self.tau_x.append(grid_x)
        self.tau_u.append(grid_u)


def _rounded(array: MPArray) -> Array:
    """Round an array of mpmath numbers or fractions to float, each correctly rounded."""
    return np.array([float(x) for x in array], dtype=np.float64)


def _segments(fraction: Sequence[float]) -> tuple[list[Fraction], list[Fraction]]:
    """Return each segment's fraction of the phase, and where it starts, as exact rationals.

    The fractions are the exact values of the floats given, normalized so that they sum to
    exactly 1, and the starts are their running sums. Exact, because a grid point is a
    difference of numbers near 1 (``2 * tau - 1``): high precision would still leave a point
    that should be 0, such as the middle of a uniform mesh, as a tiny number, not zero.
    """
    given = [Fraction(float(f)) for f in fraction]
    total = sum(given, Fraction(0))
    fractions = [f / total for f in given]
    starts = [Fraction(0)]
    for f in fractions[:-1]:
        starts.append(starts[-1] + f)
    return fractions, starts


def _exact(x: Any) -> Fraction:
    """Return an mpmath number as the exact rational it is."""
    sign, man, exp, _ = mp.mpf(x)._mpf_
    man = -man if sign else man
    return Fraction(man << exp) if exp >= 0 else Fraction(man, 1 << -exp)


@lru_cache(maxsize=1024)
def _scaled(method: str, n: int, fraction: Any) -> tuple[Any, ...]:
    """Return one segment's rule, scaled to a segment spanning `fraction` of the phase.

    The points, on [0, 1] relative to the segment's start, are exact rationals, from the
    rule's nodes (exact binary numbers at the working precision); the weights are in high
    precision, since LGL's shared weights are summed before the one rounding. The derivative
    blocks, which are placed as they are, are divided by the fraction in high precision and
    rounded here. Cached, so that the segments of a uniform mesh are computed once.
    """
    rule = {"lgl": lgl, "lgr": lgr, "lg": lg}[method]
    with mp.workdps(DIGITS):
        t, w, d, extra = rule.mp(n)  # type: ignore[attr-defined]
        points = np.array([(_exact(x) + 1) / 2 * fraction for x in t], dtype=object)
        f = mp.mpf(fraction.numerator) / fraction.denominator
        weights = w * f
        scaled = np.array([float(x / f) for x in d.flat], dtype=np.float64).reshape(d.shape)
        if method == "lgl":
            extra = np.array([float(x / f) for x in extra], dtype=np.float64)
        elif method == "lg":
            extra = _rounded(extra)
    return points, weights, scaled, extra
