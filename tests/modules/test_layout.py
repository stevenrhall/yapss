"""

Test the phase layout record against the code that derives the layout today.

Until the layout consumers read `layout.phase_layout`, this is an equivalence test: every
field of the record must agree with what `structure.py` builds, what `mesh.py` sizes and
orders, and what `assembly.phase_geometry` derives, for every spectral method and a range
of meshes (one segment, several, unequal, the two-point minimum). Once the consumers read
the record, the comparisons with their re-derivations become circular and are replaced by
hand-checked cases.

"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from yapss import Problem
from yapss._private.assembly import index_twins, phase_geometry
from yapss._private.layout import SPECTRAL_METHODS, PhaseLayout, phase_layout, problem_layout
from yapss._private.mesh import Mesh
from yapss._private.structure import (
    calculate_ic,
    calculate_nz,
    get_nlp_cf_structure,
    get_nlp_dv_structure,
)

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


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_layout_matches_the_decision_variable_structure(method, mesh_name):
    problem = build(method, MESHES[mesh_name])
    dv = get_nlp_dv_structure(problem, int)
    dv.z[:] = np.arange(len(dv.z))
    for p, layout in enumerate(problem_layout(problem)):
        phase = dv.phase[p]
        storage = phase.xa[0]
        start = storage[0]
        assert len(storage) == layout.n_state_storage
        assert len(phase.xc[0]) == layout.n_eval
        np.testing.assert_array_equal(phase.xc[0], storage[: layout.n_eval])
        assert phase.x0[0] - start == layout.x0_position
        assert phase.xf[0] - start == layout.xf_position
        assert len(phase.u[0]) == layout.n_eval
        if layout.n_zero_mode:
            np.testing.assert_array_equal(phase.xs[0], storage[layout.n_time :])


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_layout_matches_the_constraint_structure(method, mesh_name):
    problem = build(method, MESHES[mesh_name])
    cf = get_nlp_cf_structure(problem, int)
    for p, layout in enumerate(problem_layout(problem)):
        phase = cf.phase[p]
        assert len(phase.defect[0]) == layout.n_collocation
        assert len(phase.path[0]) == layout.n_eval
        if method == "lg":
            assert len(phase.lg_defect[0]) == layout.n_boundary_defect
        else:
            assert layout.n_boundary_defect == 0
        if method == "lgl":
            np.testing.assert_array_equal(phase.defect_index, layout.defect_index)
        else:
            np.testing.assert_array_equal(layout.defect_index, np.arange(layout.n_collocation))


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_layout_matches_the_vector_lengths(method, mesh_name):
    problem = build(method, MESHES[mesh_name])
    layouts = problem_layout(problem)
    nz = problem.ns + sum(
        problem.nx[p] * layout.n_state_storage + problem.nu[p] * layout.n_eval + problem.nq[p] + 2
        for p, layout in enumerate(layouts)
    )
    nc = problem.nd + sum(
        problem.nx[p] * (layout.n_collocation + layout.n_boundary_defect)
        + problem.nh[p] * layout.n_eval
        + problem.nq[p]
        + 1
        for p, layout in enumerate(layouts)
    )
    assert calculate_nz(problem, method) == nz
    assert calculate_ic(problem, method) == nc


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_layout_matches_the_mesh(method, mesh_name):
    problem = build(method, MESHES[mesh_name])
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(method)
    for p, layout in enumerate(problem_layout(problem)):
        assert len(mesh.tau_x[p]) == layout.n_time
        assert len(mesh.tau_u[p]) == layout.n_eval
        assert len(mesh.w[p]) == layout.n_eval
        if method == "lg":
            np.testing.assert_array_equal(mesh.lg_index[p], layout.time_order)
        else:
            np.testing.assert_array_equal(layout.time_order, np.arange(layout.n_time))
        # the trims count the evaluation points on tau = +1 (for t0) and tau = -1 (for tf)
        tau_u = mesh.tau_u[p]
        assert layout.trim_t0 == int(np.isclose(tau_u[-1], 1.0))
        assert layout.trim_tf == int(np.isclose(tau_u[0], -1.0))


@pytest.mark.parametrize(("method", "mesh_name"), CASES)
def test_layout_matches_the_assembly_geometry(method, mesh_name):
    problem = build(method, MESHES[mesh_name])
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(method)
    nlp = SimpleNamespace(problem=problem, mesh=mesh)
    dv = get_nlp_dv_structure(problem, float)
    twins = index_twins(problem)
    for p, layout in enumerate(problem_layout(problem)):
        geometry = phase_geometry(nlp, dv, twins, p)
        assert (geometry.nc, geometry.nw) == (layout.n_collocation, layout.n_eval)
        assert (geometry.trim_t0, geometry.trim_tf) == (layout.trim_t0, layout.trim_tf)
        np.testing.assert_array_equal(geometry.defect_index, layout.defect_index)


def test_layout_is_cached_and_its_arrays_are_read_only():
    first = phase_layout("lg", [3, 4])
    assert phase_layout("lg", (3, 4)) is first
    assert isinstance(first, PhaseLayout)
    with pytest.raises(ValueError, match="read-only"):
        first.time_order[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        first.defect_index[0] = 0


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="spectral method must be one of"):
        phase_layout("gauss", (3,))
