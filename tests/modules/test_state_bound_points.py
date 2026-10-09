"""The state's general bound is applied at the evaluation points and the two ends only.

Under LG the state at an interior segment boundary is not an evaluation point: it is fixed by
the collocation values, through the boundary defect of the segment before it, so a bound on it
would be redundant, and its multiplier would have no quadrature weight. LGL and LGR collocate
their segment boundaries, so nothing changes for them. The ends are bounded by the stricter of
the general bound and their own.
"""

import numpy as np
import pytest

import yapss
from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot
from yapss._backend.bounds import get_nlp_decision_variable_bounds
from yapss._backend.layout import problem_layout
from yapss._backend.structure import get_nlp_dv_structure


class State(yapss.State):
    a = yapss.scalar()
    b = yapss.scalar()


class Control(yapss.Control):
    u = yapss.scalar()


class Phase(yapss.Phase):
    state: State
    control: Control


class Phases(yapss.Phases):
    phase: Phase


class Bounded(yapss.Problem):
    phases: Phases


@pytest.mark.parametrize("method", ["lgl", "lgr", "lg"])
def test_the_general_state_bound_is_applied_at_the_evaluation_points_and_the_ends(method):
    problem = Bounded("bounds")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.a = arg.control.u
        out.dynamics.b = arg.control.u

    problem.register.objective(lambda arg: 0.0)
    problem.spectral_method = method
    ph.mesh = yapss.Mesh([(0.2, 4), (0.3, 3), (0.5, 5)])
    ph.state.a.bounds = (-1.0, 3.0)
    ph.state.b.bounds = (-2.0, 4.0)
    ph.initial_state.a.bounds = (0.5, 0.5)
    ph.final_state.b.bounds = (None, 10.0)
    ph.time.guess = (0.0, 1.0)
    spec = to_transcription_spec(snapshot(problem))

    ub, lb = get_nlp_decision_variable_bounds(spec)
    layout = problem_layout(spec)[0]
    index = get_nlp_dv_structure(spec, np.intp)
    index.z[:] = np.arange(len(index.z))
    ends = (layout.x0_position, layout.xf_position)
    interior = [k for k in range(layout.n_eval, layout.n_time) if k not in ends]
    at_points = [k for k in range(layout.n_eval) if k not in ends]
    assert len(interior) == (2 if method == "lg" else 0)
    for i, (lower, upper) in enumerate([(-1.0, 3.0), (-2.0, 4.0)]):
        rows = np.asarray(index.phase[0].x[i])
        assert np.all(lb[rows[at_points]] == lower)
        assert np.all(ub[rows[at_points]] == upper)
        assert np.all(lb[rows[interior]] == -np.inf)
        assert np.all(ub[rows[interior]] == np.inf)
    x0, xf = np.asarray(index.phase[0].x0), np.asarray(index.phase[0].xf)
    assert (lb[x0[0]], ub[x0[0]]) == (0.5, 0.5)  # the end's own bound is the stricter
    assert (lb[x0[1]], ub[x0[1]]) == (-2.0, 4.0)  # no end bound: the general one
    assert (lb[xf[1]], ub[xf[1]]) == (-2.0, 4.0)  # the general bound is the stricter
