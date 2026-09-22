"""One grid per phase: every per-point quantity is reported on the state points, under every method.

Where the method produced no value -- LGR's final point, both ends of every LG segment -- the
value is filled from the method's own polynomial: barycentric Lagrange on the segment's
collocation points, and at an LG join the average of the two segments' values. `ps.collocated`
marks the points that are the solver's own.
"""

import pickle

import numpy as np
import pytest

from yapss._api.solution import _Grid
from yapss.examples import goddard_problem_3_phase, orbit_raising

# ------------------------------------------------------------------ the fill, on known data


def lgr_like(points, t0=0.0, tf=1.0):
    """Return state and collocation times of an LGR-shaped grid: every segment's start, no end."""
    edges = np.linspace(t0, tf, len(points) + 1)
    time_c = np.concatenate(
        [np.linspace(a, b, n, endpoint=False) for a, b, n in zip(edges[:-1], edges[1:], points)]
    )
    return np.append(time_c, tf), time_c, edges


def lg_like(points, t0=0.0, tf=1.0):
    """Return state and collocation times of an LG-shaped grid: interior points, then the ends."""
    edges = np.linspace(t0, tf, len(points) + 1)
    time, time_c = [], []
    for a, b, n in zip(edges[:-1], edges[1:], points):
        inner = np.linspace(a, b, n + 2)[1:-1]
        time += [a, *inner]
        time_c += list(inner)
    return np.array([*time, tf]), np.array(time_c), edges


def test_lgr_extrapolates_the_last_segment_to_the_final_point():
    """A polynomial of degree below the segment's point count is reproduced exactly."""
    time, time_c, _ = lgr_like([4, 4, 4])
    grid = _Grid("lgr", (4, 4, 4), time, time_c)
    filled = grid.fill(time_c**3 - 2 * time_c)
    assert list(grid.collocated) == [True] * 12 + [False]
    assert filled[-1] == pytest.approx(1.0**3 - 2 * 1.0)


def test_lg_fills_the_phase_ends_from_the_first_and_last_segments():
    time, time_c, _ = lg_like([3, 3])
    grid = _Grid("lg", (3, 3), time, time_c)
    filled = grid.fill(5 * time_c**2 + 1)
    assert filled[0] == pytest.approx(1.0)  # t = 0
    assert filled[-1] == pytest.approx(6.0)  # t = 1


def test_lg_averages_the_two_segments_at_a_join():
    """Different polynomials either side of the join meet at the mean of their two values."""
    time, time_c, edges = lg_like([3, 3])
    grid = _Grid("lg", (3, 3), time, time_c)
    values = np.where(time_c < edges[1], time_c, 10 + time_c)
    filled = grid.fill(values)
    join = int(np.flatnonzero(time == edges[1])[0])
    assert not grid.collocated[join]
    assert filled[join] == pytest.approx(((0.5) + (10.5)) / 2)


def test_collocated_values_are_the_solver_s_untouched():
    time, time_c, _ = lg_like([3, 4])
    grid = _Grid("lg", (3, 4), time, time_c)
    values = np.arange(len(time_c), dtype=float)
    np.testing.assert_array_equal(grid.fill(values)[grid.collocated], values)


def test_rows_are_filled_row_by_row():
    time, time_c, _ = lgr_like([3, 3])
    grid = _Grid("lgr", (3, 3), time, time_c)
    filled = grid.fill(np.vstack([time_c, 2 * time_c]))
    assert filled.shape == (2, len(time))
    assert filled[:, -1] == pytest.approx([1.0, 2.0])


def test_lgl_needs_no_fill():
    time = np.linspace(0.0, 1.0, 7)
    grid = _Grid("lgl", (4, 4), time, time)
    assert grid.collocated.all()
    np.testing.assert_array_equal(grid.fill(time), time)


def test_a_zero_duration_phase_is_filled_with_nan():
    """Its points coincide, so there is no polynomial to evaluate."""
    time, time_c = np.zeros(5), np.zeros(4)
    grid = _Grid("lgr", (4,), time, time_c)
    filled = grid.fill(np.ones(4))
    assert np.isnan(filled[-1])
    assert (filled[:-1] == 1.0).all()


# ------------------------------------------------------------------- the fill, on a solve


@pytest.fixture(scope="module", params=["lgl", "lgr", "lg"])
def goddard(request):
    problem = goddard_problem_3_phase.setup()
    problem.ipopt_options.print_level = 0
    problem.method = request.param
    return problem, problem.solve()


def test_every_per_point_quantity_is_on_the_state_points(goddard):
    """What the corpus could not do under LGR and LG: plot a control against time."""
    _, solution = goddard
    for name in ("boost", "singular", "coast"):
        ps = solution[name]
        n = len(ps.time)
        assert ps.collocated.shape == (n,)
        for quantity in (
            ps.control.thrust,
            ps.dynamics.h,
            ps.costate.v,
            ps.hamiltonian,
            ps.multiplier.control.thrust,
        ):
            assert np.shape(quantity) == (n,)
        assert np.isfinite(ps.control.thrust).all()
    assert solution["singular"].path.switching.shape == (len(solution["singular"].time),)


def test_the_filled_points_are_where_the_method_has_none(goddard):
    """LGL: none. LGR: the final point. LG: every segment's start, and the final point."""
    problem, solution = goddard
    ps = solution["boost"]
    missing = ps.time[~ps.collocated]
    if problem.method == "lgl":
        assert missing.size == 0
    elif problem.method == "lgr":
        assert missing == pytest.approx([ps.time[-1]])
    else:
        fractions = np.array(ps.mesh.fractions)
        t0, tf = ps.time[0], ps.time[-1]
        edges = t0 + np.concatenate(([0.0], np.cumsum(fractions) / fractions.sum())) * (tf - t0)
        assert missing == pytest.approx(edges)


def test_the_fill_is_close_to_its_neighbours_on_a_smooth_arc(goddard):
    """The coast arc's costate is smooth, so an extrapolated end sits beside its neighbour."""
    _, solution = goddard
    ps = solution["coast"]
    lam = ps.costate.v
    scale = np.abs(lam).max()
    for i in np.flatnonzero(~ps.collocated):
        neighbour = i - 1 if i > 0 else i + 1
        assert abs(lam[i] - lam[neighbour]) < 0.05 * scale


def test_the_mask_and_the_fill_pickle():
    problem = orbit_raising.setup()
    problem.ipopt_options.print_level = 0
    problem.method = "lgr"
    solution = problem.solve()
    copy = pickle.loads(pickle.dumps(solution))
    ps, ps_copy = solution[problem.phases.raise_], copy[problem.phases.raise_]
    np.testing.assert_array_equal(ps_copy.collocated, ps.collocated)
    np.testing.assert_array_equal(ps_copy.control[:], ps.control[:])
