"""

Test the phase layout record and the NLP structures built from it.

The structures are checked against hand-worked indices for a two-segment mesh under each
spectral method, which document the layout independently of the code. The endpoint trims
are checked against the mesh times the quadrature actually produces, for every method and
a range of meshes (one segment, several, unequal, the two-point minimum).

"""

from __future__ import annotations

import numpy as np
import pytest

from yapss._legacy import Problem
from yapss._private.layout import SPECTRAL_METHODS, PhaseLayout, phase_layout, problem_layout
from yapss._private.mesh import Mesh
from yapss._private.structure import get_nlp_cf_structure, get_nlp_dv_structure

MESHES = {
    "one segment": [(7,)],
    "two segments": [(3, 4)],
    "unequal segments": [(2, 5, 3)],
    "two-point minimum": [(2, 2, 2)],
    "two phases": [(3, 4), (5,)],
}


def build(method: str, meshes: list[tuple[int, ...]]) -> Problem:
    n = len(meshes)
    problem = Problem(name="layout", nx=[2] * n, nu=[1] * n, nq=[1] * n, nh=[1] * n, ns=1, nd=1)
    for p, points in enumerate(meshes):
        problem.mesh.phase[p].collocation_points = points
        problem.mesh.phase[p].fraction = [1.0 / len(points)] * len(points)
    problem.spectral_method = method
    return problem


CASES = [(method, name) for method in SPECTRAL_METHODS for name in MESHES]


def r(start: int, stop: int) -> list[int]:
    return list(range(start, stop))


# Hand-worked NLP indices for nx=[2], nu=[1], nq=[1], nh=[1], ns=1, nd=1 on a (3, 4) mesh:
# K = 2 segments and N = 7 collocation points.
HAND_CHECKED = {
    # LGR: 7 evaluation points, 8 state time points (the final point is not collocated)
    "lgr": {
        "nz": 27, "xa": [r(0, 8), r(8, 16)], "x": [r(0, 8), r(8, 16)], "xc": [r(0, 7), r(8, 15)],
        "xs": [[], []], "x0": [0, 8], "xf": [7, 15], "u": [r(16, 23)], "q": [23], "t0": [24],
        "tf": [25], "s": [26],
        "nc": 24, "defect": [r(0, 7), r(7, 14)], "lg_defect": [[], []], "path": [r(14, 21)],
        "integral": [21], "duration": [22], "discrete": [23], "defect_index": r(0, 7),
        "time_order": r(0, 8),
    },
    # LGL: segment boundaries are shared, so 6 evaluation and time points, plus 2 zero modes
    # per state; each defect reads the point its collocation node sits on
    "lgl": {
        "nz": 26, "xa": [r(0, 8), r(8, 16)], "x": [r(0, 6), r(8, 14)], "xc": [r(0, 6), r(8, 14)],
        "xs": [[6, 7], [14, 15]], "x0": [0, 8], "xf": [5, 13], "u": [r(16, 22)], "q": [22],
        "t0": [23], "tf": [24], "s": [25],
        "nc": 23, "defect": [r(0, 7), r(7, 14)], "lg_defect": [[], []], "path": [r(14, 20)],
        "integral": [20], "duration": [21], "discrete": [22],
        "defect_index": [0, 1, 2, 2, 3, 4, 5], "time_order": r(0, 6),
    },
    # LG: 7 collocation values stored first, then the 2 segment starts, then the final value;
    # 2 boundary defects per state follow the defects
    "lg": {
        "nz": 31, "xa": [r(0, 10), r(10, 20)], "x": [r(0, 10), r(10, 20)],
        "xc": [r(0, 7), r(10, 17)], "xs": [[], []], "x0": [7, 17], "xf": [9, 19],
        "u": [r(20, 27)], "q": [27], "t0": [28], "tf": [29], "s": [30],
        "nc": 28, "defect": [r(0, 7), r(7, 14)], "lg_defect": [[14, 15], [16, 17]],
        "path": [r(18, 25)], "integral": [25], "duration": [26], "discrete": [27],
        "defect_index": r(0, 7), "time_order": [7, 0, 1, 2, 8, 3, 4, 5, 6, 9],
    },
}  # fmt: skip


@pytest.mark.parametrize("method", SPECTRAL_METHODS)
def test_structures_match_hand_worked_indices(method):
    expected = HAND_CHECKED[method]
    problem = build(method, [(3, 4)])
    dv = get_nlp_dv_structure(problem._to_spec(), int)
    dv.z[:] = np.arange(len(dv.z))
    cf = get_nlp_cf_structure(problem._to_spec(), int)
    cf.c[:] = np.arange(len(cf.c))
    (layout,) = problem_layout(problem._to_spec())
    dv_phase, cf_phase = dv.phase[0], cf.phase[0]

    def lists(views):
        return [view.tolist() for view in views]

    actual = {
        "nz": len(dv.z), "xa": lists(dv_phase.xa), "x": lists(dv_phase.x),
        "xc": lists(dv_phase.xc), "xs": lists(dv_phase.xs), "x0": dv_phase.x0.tolist(),
        "xf": dv_phase.xf.tolist(), "u": lists(dv_phase.u), "q": dv_phase.q.tolist(),
        "t0": dv_phase.t0.tolist(), "tf": dv_phase.tf.tolist(), "s": dv.s.tolist(),
        "nc": len(cf.c), "defect": lists(cf_phase.defect), "lg_defect": lists(cf_phase.lg_defect),
        "path": lists(cf_phase.path), "integral": cf_phase.integral.tolist(),
        "duration": cf_phase.duration.tolist(), "discrete": cf.discrete.tolist(),
        "defect_index": np.asarray(cf_phase.defect_index).tolist(),
        "time_order": layout.time_order.tolist(),
    }  # fmt: skip
    assert actual == expected


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_trims_match_the_mesh_times(method, mesh_name):
    """The trims count the evaluation points on tau = +1 (for t0) and tau = -1 (for tf)."""
    problem = build(method, MESHES[mesh_name])
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(method)
    for p, layout in enumerate(problem_layout(problem._to_spec())):
        tau_u = mesh.tau_u[p]
        assert layout.trim_t0 == int(np.isclose(tau_u[-1], 1.0))
        assert layout.trim_tf == int(np.isclose(tau_u[0], -1.0))


def test_layout_is_cached_and_its_arrays_are_read_only():
    first = phase_layout("lg", [3, 4])
    assert phase_layout("lg", (3, 4)) is first
    assert isinstance(first, PhaseLayout)
    with pytest.raises(ValueError, match="read-only"):
        first.time_order[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        first.defect_index[0] = 0
