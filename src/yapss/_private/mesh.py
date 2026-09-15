"""
Module mesh.
"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# package imports
from .layout import phase_layout
from .quadrature import lg, lgl, lgr

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    # package imports
    from .problem import MeshPhase

    # typing
    Array = NDArray[np.float64]


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
    phase : tuple[MeshPhase, ...]
        The mesh phases of the problem
    """

    def __init__(self, mesh_phase: tuple[MeshPhase, ...]) -> None:
        self.phase = mesh_phase
        self.d: list[NDArray[np.float64]] = []
        self.w: list[NDArray[np.float64]] = []
        self.tau_x: list[NDArray[np.float64]] = []
        self.tau_u: list[NDArray[np.float64]] = []
        self.b_lg: list[csr_matrix] = []

    def set_matrices(self, spectral_method: str) -> None:
        """Compute and save the matrices and vectors needed for integration and differentiation.

        Parameters
        ----------
        spectral_method : {"lg", "lgr", "lgl"}
        """
        if spectral_method == "lgl":
            self.set_matrices_lgl()
        elif spectral_method == "lgr":
            self.set_matrices_lgr()
        elif spectral_method == "lg":
            self.set_matrices_lg()
        else:
            raise RuntimeError

    def set_matrices_lgr(self) -> None:
        """Compute and save the arrays for LGR integration and differentiation."""
        for phase in self.phase:
            # number of collocation points
            col_points = phase.collocation_points
            fraction: Array = np.array(phase.fraction, dtype=float)
            s = fraction.sum()
            fraction /= s

            layout = phase_layout("lgr", col_points)
            nc = layout.n_collocation
            nxpoints = layout.n_time
            nupoints = layout.n_eval
            d = lil_matrix((nc, layout.n_state_storage), dtype=np.float64)

            # quadrature weighting for phase
            w: Array = np.zeros([nupoints], dtype=np.float64)

            tau_x: Array = np.zeros([nxpoints], dtype=np.float64)
            tau_u: Array = np.zeros([nupoints], dtype=np.float64)
            tau_last = 0.0
            i0 = 0  # row (wc) index
            j0 = 0  # column (w) index
            k0 = 0

            for k, nc in enumerate(col_points):
                alpha = 1 / fraction[k]
                tk, wk, dk, _ = lgr(nc)

                tk[:] = (tk + 1) / 2
                tau_x[j0 : j0 + len(tk)] = tau_last + tk * fraction[k]
                tau_u[k0 : k0 + len(tk)] = tau_last + tk * fraction[k]
                k0 += len(tk)
                tau_last += fraction[k]
                d[i0 : i0 + nc, j0 : j0 + nc + 1] = dk * alpha

                # assemble w
                w[j0 : j0 + len(wk)] += wk * fraction[k]

                # update indices
                i0 += nc
                j0 += nc

            d = d.tocsr()
            d.eliminate_zeros()
            self.d.append(d)
            self.w.append(w)
            self.b_lg.append(csr_matrix((0, layout.n_state_storage), dtype=np.float64))

            tau_x[:] = 2 * tau_x - 1.0
            tau_x[0], tau_x[-1] = -1.0, 1.0
            tau_u[:] = 2 * tau_u - 1.0
            self.tau_x.append(tau_x)
            self.tau_u.append(tau_u)

    def set_matrices_lgl(self) -> None:
        """Compute and save the arrays needed for LGL integration and differentiation."""
        for phase in self.phase:
            # number of collocation points
            col_points = phase.collocation_points
            fraction: Array = np.array(phase.fraction, dtype=float)
            s = fraction.sum()
            fraction /= s

            layout = phase_layout("lgl", col_points)
            nc = layout.n_collocation
            nsegs = layout.n_segments
            nxpoints = layout.n_time
            nupoints = layout.n_eval
            d = lil_matrix((nc, layout.n_state_storage), dtype=np.float64)
            d0 = lil_matrix((nc, nsegs), dtype=np.float64)

            # quadrature weighting for phase
            w: Array = np.zeros([nupoints], dtype=np.float64)

            tau_x: Array = np.zeros([nxpoints], dtype=np.float64)
            tau_u: Array = np.zeros([nupoints], dtype=np.float64)

            tau_last = 0.0
            i0 = 0  # row (wc) index
            j0 = 0  # column (w) index
            k0 = 0

            for k, nc in enumerate(col_points):
                alpha = 1 / fraction[k]

                # assemble D matrix
                tk, wk, dk, d0k = lgl(nc)

                tk[:] = (tk + 1) / 2
                tau_x[j0 : j0 + len(tk)] = tau_last + tk * fraction[k]
                tau_u[k0 : k0 + len(tk)] = tau_last + tk * fraction[k]
                k0 += len(tk) - 1
                tau_last += fraction[k]
                d[i0 : i0 + nc, j0 : j0 + nc] = dk * alpha
                assert d0k is not None
                d0[i0 : i0 + nc, k] = d0k * alpha

                # assemble w
                w[j0 : j0 + len(wk)] += wk * fraction[k]

                # update indices
                i0 += nc
                j0 += nc - 1

            d[:, -nsegs:] = d0
            d = d.tocsr()
            d.eliminate_zeros()
            self.d.append(d)
            self.w.append(w)
            self.b_lg.append(csr_matrix((0, layout.n_state_storage), dtype=np.float64))

            tau_x[:] = 2 * tau_x - 1.0
            tau_x[0], tau_x[-1] = -1.0, 1.0
            tau_u[:] = 2 * tau_u - 1.0
            self.tau_x.append(tau_x)
            self.tau_u.append(tau_u)

    def set_matrices_lg(self) -> None:
        """Compute and save the arrays needed for LG integration and differentiation."""
        for phase in self.phase:
            # number of collocation points
            col_points = phase.collocation_points
            fraction: Array = np.array(phase.fraction, dtype=float)
            s = fraction.sum()
            fraction /= s

            layout = phase_layout("lg", col_points)
            nc = layout.n_collocation
            nsegs = layout.n_segments
            nxpoints = layout.n_time
            nupoints = layout.n_eval
            d = lil_matrix((nc, layout.n_state_storage), dtype=np.float64)
            d0 = lil_matrix((nc, nsegs), dtype=np.float64)

            # quadrature weighting for phase
            w: Array = np.zeros([nupoints], dtype=np.float64)

            tau_x: Array = np.zeros([nxpoints], dtype=np.float64)
            tau_u: Array = np.zeros([nupoints], dtype=np.float64)
            bmat: Array = np.zeros([layout.n_boundary_defect, nxpoints], dtype=np.float64)

            tau_last = 0.0
            i0 = 0  # row (wc) index

            for k, nck in enumerate(col_points):
                alpha = 1 / fraction[k]
                tk, wk, dk, bk = lg(nck)
                wk = wk[1:]

                tk[:] = (tk + 1) / 2
                tau_x[i0 + k : i0 + k + len(tk)] = tau_last + tk * fraction[k]
                tau_u[i0 : i0 + len(tk) - 1] = tau_last + tk[1:] * fraction[k]

                tau_last += fraction[k]
                d[i0 : i0 + nck, i0 : i0 + nck] = dk[1:, 1:] * alpha
                d0[i0 : i0 + nck, k] = dk[1:, 0] * alpha
                bmat[k, i0 : i0 + nck] = bk[1:]
                bmat[k, -nsegs - 1 + k] = bk[0]
                bmat[k, -nsegs + k] = -1.0

                # assemble w
                w[i0 : i0 + len(wk)] += wk * fraction[k]

                # update indices
                i0 += nck

            d[:, -nsegs - 1 : -1] = d0

            # save the results
            d = d.tocsr()
            d.eliminate_zeros()
            bmat_csr = csr_matrix(bmat)
            self.d.append(d)
            self.w.append(w)
            self.b_lg.append(bmat_csr)

            tau_x[:] = 2 * tau_x - 1.0
            tau_x[0], tau_x[-1] = -1.0, 1.0
            tau_u[:] = 2 * tau_u - 1.0
            self.tau_x.append(tau_x)
            self.tau_u.append(tau_u)


# TODO: Make bmat a csr matrix
# TODO: make tau have range [0,1]?
# TODO: do calculations in this module in high precision?
