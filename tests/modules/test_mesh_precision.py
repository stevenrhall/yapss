"""The mesh is composed in high precision and rounded to float once.

The quadrature rules are computed in mpmath, and the mesh is built from them before any
rounding: the grid in exact rational arithmetic (the segment fractions are the exact values of
the floats given, normalized to sum to exactly 1), the weights and the derivative matrix at the
working precision. So every entry is correctly rounded, the ends of the phase are exactly -1 and
1, a segment boundary is exactly where the fractions put it, and a symmetric mesh gives a
symmetric grid. Composed from floats, entries were off by up to three units in the last place,
the last collocation point could miss 1, and a uniform mesh was not symmetric.

The reference here composes the same mesh independently, at 50 digits.
"""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from mpmath import mp

from yapss._backend import quadrature
from yapss._backend.layout import phase_layout
from yapss._backend.mesh import Mesh

METHODS = ("lgl", "lgr", "lg")
CASES = {
    "uniform": ((10,) * 10, (0.1,) * 10),
    "uneven": ((4, 3, 5), (0.2, 0.3, 0.5)),
    "sevenths": ((6,) * 7, (1 / 7,) * 7),
    "many": ((15,) * 20, (0.05,) * 20),
    "one segment": ((40,), (1.0,)),
    "irregular": ((3, 8, 2, 5), (0.13, 0.4, 0.07, 0.4)),
}


def built(method, points, fractions):
    mesh = Mesh([SimpleNamespace(collocation_points=points, fraction=fractions)])
    mesh.set_matrices(method)
    return mesh.tau_x[0], mesh.tau_u[0], mesh.w[0], mesh.d[0].toarray(), mesh.b_lg[0].toarray()


def exact(x):
    sign, man, exp, _ = mp.mpf(x)._mpf_
    return Fraction(-man if sign else man) * Fraction(2) ** exp


def reference(method, points, fractions):
    """The same mesh, composed independently: grids exact, the rest at 50 digits."""
    with mp.workdps(50):
        rule = getattr(quadrature, method).__wrapped__
        given = [Fraction(f) for f in fractions]
        frac = [f / sum(given) for f in given]
        layout = phase_layout(method, points)
        nx, nu, ns = layout.n_time, layout.n_eval, layout.n_state_storage
        nseg = len(points)
        tx = [Fraction(0)] * nx
        tu = [Fraction(0)] * nu
        w = [mp.mpf(0)] * nu
        d = np.zeros((layout.n_collocation, ns))
        b = np.zeros((layout.n_boundary_defect, nx))
        start, i0, j0 = Fraction(0), 0, 0
        for k, n in enumerate(points):
            f, fm = frac[k], mp.mpf(frac[k].numerator) / frac[k].denominator
            t, wk, dk, extra = rule(n)
            grid = [start + (exact(x) + 1) / 2 * f for x in t]
            dk = np.array([[float(x / fm) for x in row] for row in dk])
            if method == "lgr":
                tx[j0 : j0 + n] = grid
                tu[j0 : j0 + n] = grid
                d[i0 : i0 + n, j0 : j0 + n + 1] = dk
                for i in range(n):
                    w[j0 + i] += wk[i] * fm
                i0 += n
                j0 += n
            elif method == "lgl":
                tx[j0 : j0 + n] = grid
                tu[j0 : j0 + n] = grid
                d[i0 : i0 + n, j0 : j0 + n] = dk
                d[i0 : i0 + n, ns - nseg + k] = [float(x / fm) for x in extra]
                for i in range(n):
                    w[j0 + i] += wk[i] * fm
                i0 += n
                j0 += n - 1
            else:
                tx[i0 + k : i0 + k + n + 1] = grid
                tu[i0 : i0 + n] = grid[1:]
                d[i0 : i0 + n, i0 : i0 + n] = dk[1:, 1:]
                d[i0 : i0 + n, ns - nseg - 1 + k] = dk[1:, 0]
                b[k, i0 : i0 + n] = [float(x) for x in extra[1:]]
                b[k, nx - nseg - 1 + k] = float(extra[0])
                b[k, nx - nseg + k] = -1.0
                for i in range(n):
                    w[i0 + i] += wk[i + 1] * fm
                i0 += n
            start += f
        tx[-1] = Fraction(1)
        grid_x = np.array([float(2 * x - 1) for x in tx])
        grid_u = np.array([float(2 * x - 1) for x in tu])
        return grid_x, grid_u, np.array([float(x) for x in w]), d, b


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("case", CASES)
def test_every_entry_is_correctly_rounded(method, case):
    points, fractions = CASES[case]
    for name, mine, theirs in zip(
        ("tau_x", "tau_u", "w", "d", "b"),
        built(method, points, fractions),
        reference(method, points, fractions),
        strict=True,
    ):
        if mine.size or theirs.size:
            np.testing.assert_array_equal(mine, theirs, err_msg=name)


@pytest.mark.parametrize("segments", [3, 7, 10, 50])
def test_a_collocated_end_is_exactly_the_end_of_the_phase(segments):
    """Both ends under LGL, the first under LGR; under LG neither is a collocation point."""
    for method, first, last in (("lgl", True, True), ("lgr", True, False), ("lg", False, False)):
        tau_x, tau_u, *_ = built(method, segments * (6,), segments * (1 / segments,))
        assert (tau_x[0], tau_x[-1]) == (-1.0, 1.0)
        assert (tau_u[0] == -1.0) == first
        assert (tau_u[-1] == 1.0) == last
        assert -1.0 <= tau_u[0] < tau_u[-1] <= 1.0


@pytest.mark.parametrize("method", ["lgl", "lg"])
@pytest.mark.parametrize("segments", [4, 7, 20])
def test_a_uniform_mesh_has_a_symmetric_grid(method, segments):
    tau_x = built(method, segments * (5,), segments * (1 / segments,))[0]
    np.testing.assert_array_equal(tau_x, -tau_x[::-1])


@pytest.mark.parametrize("method", METHODS)
def test_a_segment_boundary_is_where_the_fractions_put_it(method):
    """Seven floats of 1/7 sum to 0.9999999999999998; normalized exactly, they are sevenths."""
    tau_x = built(method, (6,) * 7, (1 / 7,) * 7)[0]
    step = {"lgl": 5, "lgr": 6, "lg": 7}[method]
    for k in range(8):
        assert tau_x[min(k * step, len(tau_x) - 1)] == float(Fraction(2 * k, 7) - 1)
