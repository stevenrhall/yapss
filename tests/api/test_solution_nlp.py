"""The solver's record: flat vectors in Ipopt's order, and the positions of everything in them.

``solution.nlp`` holds what Ipopt saw and returned; ``ps.nlp.index`` and ``solution.nlp.index``
say, under the names the problem declared, where each variable and constraint row is. The tests
that matter most are the partitions: every position is named exactly once, under every method,
so nothing in the record is unreachable by name and nothing is named twice by accident.
"""

import pickle

import numpy as np
import pytest

import yapss
from yapss.examples import brachistochrone_minimal, goddard_problem_3_phase

from .test_solution_pickling import declared_in_a_function

SPECTRAL_METHODS = ["lgl", "lgr", "lg"]


@pytest.fixture(scope="module", params=SPECTRAL_METHODS)
def solved(request):
    """The function-local problem -- a block field, a renamed independent variable, a path
    constraint, an integral, a parameter and a discrete constraint -- on three segments, so that
    LG's continuity rows, LGL's zero modes and LGL's doubled segment boundaries all appear."""
    problem = declared_in_a_function()
    problem.method = request.param
    problem.phases.run.mesh = yapss.Mesh.uniform(segments=3, points=4)
    solution = problem.solve()
    return request.param, problem, solution, solution[problem.phases.run]


def _concatenated(*arrays):
    return np.sort(np.concatenate([np.ravel(a) for a in arrays]).astype(np.intp))


def _variable_positions(method, solution, ps):
    var = ps.nlp.index.variable
    arrays = [
        var.state.r,
        var.state.y,
        var.control.u,
        [var.integral.effort],
        [var.initial.s],
        [var.final.s],
        [solution.nlp.index.variable.parameter.k],
    ]
    if method == "lgl":
        arrays += [var.zero_mode.r, var.zero_mode.y]
    return _concatenated(*arrays)


def _constraint_positions(method, solution, ps):
    con = ps.nlp.index.constraint
    arrays = [
        con.dynamics.r,
        con.dynamics.y,
        con.path.size,
        [con.integral.effort],
        [con.duration],
        [solution.nlp.index.constraint.discrete.end],
    ]
    if method == "lg":
        arrays += [con.continuity.r, con.continuity.y]
    return _concatenated(*arrays)


# -- every position is named exactly once -------------------------------------------------------


def test_the_variable_index_names_every_position_in_x_once(solved):
    method, _, solution, ps = solved
    positions = _variable_positions(method, solution, ps)
    np.testing.assert_array_equal(positions, np.arange(solution.nlp.x.size))


def test_the_constraint_index_names_every_position_in_g_once(solved):
    method, _, solution, ps = solved
    positions = _constraint_positions(method, solution, ps)
    np.testing.assert_array_equal(positions, np.arange(solution.nlp.g.size))


def test_every_vector_has_the_length_of_its_index(solved):
    _, _, solution, _ = solved
    nlp = solution.nlp
    n, m = nlp.x.size, nlp.g.size
    for name in ("x_L", "x_U", "z0", "mult_x_L", "mult_x_U", "grad_f"):
        assert getattr(nlp, name).shape == (n,), name
    for name in ("g_L", "g_U", "mult_g"):
        assert getattr(nlp, name).shape == (m,), name
    assert nlp.scale.x.shape == (n,)
    assert nlp.scale.g.shape == (m,)
    jac = nlp.jac_g
    assert jac.row.shape == jac.col.shape == jac.value.shape
    assert 0 <= jac.row.min() and jac.row.max() < m
    assert 0 <= jac.col.min() and jac.col.max() < n


# -- the index reads the processed layers' numbers back ---------------------------------------


def test_indices_are_shaped_like_what_they_index(solved):
    _, _, _, ps = solved
    var = ps.nlp.index.variable
    assert var.state.r.shape == ps.state.r.shape
    assert var.state.y.shape == ps.state.y.shape
    assert var.control.u.shape == ps.control.u[..., ps.collocated].shape
    assert var.final.r.shape == ps.final.r.shape


def test_x_read_through_the_index_is_the_primal_layer(solved):
    _, _, solution, ps = solved
    nlp, var = solution.nlp, ps.nlp.index.variable
    np.testing.assert_array_equal(nlp.x[var.state.r], ps.state.r)
    np.testing.assert_array_equal(nlp.x[var.state.y], ps.state.y)
    np.testing.assert_array_equal(nlp.x[var.control.u], ps.control.u[ps.collocated])
    assert nlp.x[var.integral.effort] == ps.integral.effort
    assert nlp.x[var.initial.s] == ps.initial.s
    assert nlp.x[var.final.s] == ps.final.s
    np.testing.assert_array_equal(nlp.x[var.final.r], ps.final.r)
    assert nlp.x[solution.nlp.index.variable.parameter.k] == solution.parameter.k


def test_endpoint_entries_repeat_the_ends_of_the_state(solved):
    _, _, _, ps = solved
    var = ps.nlp.index.variable
    np.testing.assert_array_equal(var.initial.r, var.state.r[:, 0])
    np.testing.assert_array_equal(var.final.r, var.state.r[:, -1])
    assert var.initial.y == var.state.y[0]
    assert var.final.y == var.state.y[-1]


def test_g_read_through_the_index_is_the_constraint_values(solved):
    _, _, solution, ps = solved
    nlp, con = solution.nlp, ps.nlp.index.constraint
    np.testing.assert_allclose(nlp.g[con.path.size], ps.path.size[ps.collocated])
    assert nlp.g[solution.nlp.index.constraint.discrete.end] == solution.discrete.end
    assert nlp.g[con.duration] == pytest.approx(ps.duration)
    assert np.abs(nlp.g[con.dynamics.r]).max() < 1e-6  # the defect residuals of a solve


def test_point_multipliers_agree_with_the_multiplier_tree(solved):
    """Where a constraint is one row or one bound, the raw multiplier is the reported one."""
    _, _, solution, ps = solved
    nlp, var, con = solution.nlp, ps.nlp.index.variable, ps.nlp.index.constraint

    def bound(position):
        return nlp.mult_x_U[position] - nlp.mult_x_L[position]

    assert nlp.mult_g[con.integral.effort] == ps.multiplier.integral.effort
    assert nlp.mult_g[con.duration] == ps.multiplier.duration
    assert nlp.mult_g[solution.nlp.index.constraint.discrete.end] == (
        solution.multiplier.discrete.end
    )
    assert bound(var.initial.s) == ps.multiplier.initial.s
    assert bound(var.final.s) == ps.multiplier.final.s
    assert bound(solution.nlp.index.variable.parameter.k) == solution.multiplier.parameter.k


# -- each method's own machinery, under that method only -----------------------------------------


def test_zero_modes_exist_under_lgl_only(solved):
    method, _, _, ps = solved
    var = ps.nlp.index.variable
    if method == "lgl":
        assert var.zero_mode.r.shape == (2, 3)
        assert var.zero_mode.y.shape == (3,)
    else:
        with pytest.raises(AttributeError, match="'zero_mode' exists only under LGL"):
            var.zero_mode  # noqa: B018


def test_continuity_exists_under_lg_only(solved):
    method, _, _, ps = solved
    con = ps.nlp.index.constraint
    if method == "lg":
        assert con.continuity.r.shape == (2, 3)
        assert con.continuity.y.shape == (3,)
    else:
        with pytest.raises(AttributeError, match="'continuity' exists only under LG"):
            con.continuity  # noqa: B018


def test_dynamics_rows_under_lgl_count_a_segment_boundary_once_per_segment(solved):
    method, _, _, ps = solved
    rows = ps.nlp.index.constraint.dynamics.y.size
    n_collocated = int(ps.collocated.sum())
    if method == "lgl":
        assert rows == sum(ps.mesh.collocation_points) == n_collocated + 2
    else:
        assert rows == n_collocated


# -- the point each defect and continuity row sits at ------------------------------------------

# Hand-worked for three segments of four points. LGL: 10 points, a boundary row from each side.
# LGR: 13 points, the last not collocated. LG: 16 points, the start and each segment's end not
# collocated, and the continuity rows at the three segment ends.
ROW_POINTS = {
    "lgl": ([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9], None),
    "lgr": (list(range(12)), None),
    "lg": ([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], [5, 10, 15]),
}


def test_row_points_are_the_hand_worked_ones(solved):
    method, _, _, ps = solved
    dynamics, continuity = ROW_POINTS[method]
    np.testing.assert_array_equal(ps.nlp.point.dynamics, dynamics)
    if continuity is None:
        with pytest.raises(AttributeError, match="'continuity' exists only under LG"):
            ps.nlp.point.continuity  # noqa: B018
    else:
        np.testing.assert_array_equal(ps.nlp.point.continuity, continuity)


def test_defect_rows_sit_at_the_collocated_points(solved):
    """Every collocated point has a defect row, and no other point does."""
    _, _, _, ps = solved
    points = ps.nlp.point.dynamics
    assert points.size == ps.nlp.index.constraint.dynamics.y.size
    np.testing.assert_array_equal(np.unique(points), np.flatnonzero(ps.collocated))


def test_continuity_rows_sit_at_the_segment_ends(solved):
    method, _, _, ps = solved
    if method != "lg":
        pytest.skip("continuity rows exist only under LG")
    fractions = np.array(ps.mesh.fractions)
    ends = ps.s[0] + np.cumsum(fractions / fractions.sum()) * (ps.s[-1] - ps.s[0])
    np.testing.assert_allclose(ps.s[ps.nlp.point.continuity], ends)


# -- the record ----------------------------------------------------------------------------------


def test_the_record_reports_the_solve(solved):
    _, _, solution, _ = solved
    nlp = solution.nlp
    assert nlp.version == yapss.__version__
    assert nlp.status == solution.status == 0
    assert nlp.objective == solution.objective
    assert nlp.scale.objective == 1.0
    assert nlp.convergence.iterations > 0
    assert nlp.convergence.inf_pr < 1e-6
    assert nlp.convergence.inf_du < 1e-6
    assert nlp.convergence.complementarity < 1e-6


def test_the_starting_point_is_within_the_bounds(solved):
    _, _, solution, _ = solved
    nlp = solution.nlp
    assert np.all(nlp.x_L <= nlp.z0)
    assert np.all(nlp.z0 <= nlp.x_U)


def test_no_array_of_the_record_shares_memory_with_another(solved):
    _, _, solution, ps = solved
    nlp = solution.nlp
    arrays = [
        *(getattr(nlp, n) for n in ("x_L", "x_U", "z0", "x", "mult_x_L", "mult_x_U", "grad_f")),
        *(getattr(nlp, n) for n in ("g_L", "g_U", "g", "mult_g")),
        nlp.jac_g.row,
        nlp.jac_g.col,
        nlp.jac_g.value,
        nlp.scale.x,
        nlp.scale.g,
        ps.state.r,
    ]
    for i, a in enumerate(arrays):
        for b in arrays[i + 1 :]:
            assert not np.shares_memory(a, b)


def test_the_record_is_read_only(solved):
    _, _, solution, ps = solved
    with pytest.raises(AttributeError, match="read-only"):
        solution.nlp.x = None
    with pytest.raises(AttributeError, match="read-only"):
        ps.nlp.index.variable.state = None
    with pytest.raises(AttributeError, match="Did you mean 'mult_g'"):
        solution.nlp.mult_gg  # noqa: B018


def test_the_record_and_its_index_survive_pickling(solved):
    method, problem, solution, ps = solved
    copy = pickle.loads(pickle.dumps(solution))
    ps_copy = copy[problem.phases.run]
    np.testing.assert_array_equal(copy.nlp.x, solution.nlp.x)
    np.testing.assert_array_equal(copy.nlp.jac_g.value, solution.nlp.jac_g.value)
    assert copy.nlp.convergence.iterations == solution.nlp.convergence.iterations
    var, var_copy = ps.nlp.index.variable, ps_copy.nlp.index.variable
    np.testing.assert_array_equal(var_copy.state.r, var.state.r)
    assert var_copy.final.s == var.final.s
    if method != "lgl":
        with pytest.raises(AttributeError, match="exists only under LGL"):
            var_copy.zero_mode  # noqa: B018


# -- stationarity, the condition only the stored derivatives can check ---------------------------


@pytest.mark.parametrize(
    "example",
    [brachistochrone_minimal, goddard_problem_3_phase],
    ids=["minimize", "maximize"],
)
def test_stationarity_holds_on_the_stored_derivatives(example):
    """grad_f + J' mult_g + mult_x_U - mult_x_L = 0, with f the objective as the problem states
    it, under either sense: the one sign convention the documentation states."""
    problem = example.setup()
    problem.ipopt_options.print_level = 0
    nlp = problem.solve().nlp
    assert (nlp.scale.objective < 0) == (problem.objective.sense == "maximize")
    jacobian = np.zeros((nlp.g.size, nlp.x.size))
    np.add.at(jacobian, (nlp.jac_g.row, nlp.jac_g.col), nlp.jac_g.value)
    residual = nlp.grad_f + jacobian.T @ nlp.mult_g + nlp.mult_x_U - nlp.mult_x_L
    assert np.abs(residual).max() < 1e-8 * max(1.0, np.abs(nlp.grad_f).max())
