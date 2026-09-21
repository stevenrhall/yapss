"""Per-row guesses of a block field.

A block field may be guessed row by row, each row with its own samples. Both places that read
a field's guess have to look at every row: the phase's guess grid is the merge of every row's
sample times, and the coverage check has to see every row's samples.
"""

import numpy as np
import pytest

import yapss
from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot


def build(row_guess):
    """Return a two-state problem whose block field's second row is guessed by `row_guess`."""

    class Pair(yapss.State):
        r = yapss.vector(2)

    class Rate(yapss.Control):
        u = yapss.scalar()

    class Phases(yapss.Phases):
        only = yapss.phase(state=Pair, control=Rate)

    problem = yapss.Problem("block guesses", phases=Phases)
    ph = problem.phases.only

    @ph.register.continuous
    def dynamics(arg, out):
        out.dynamics.r = [arg.control.u, arg.control.u]
        return out

    @problem.register.objective
    def objective(arg):
        return arg[ph].final.time

    ph.time.initial = (0.0, 0.0)
    ph.state.r.initial[:] = [0.0, 0.0]
    ph.state.r.bounds[:] = [(-10, 10), (-10, 10)]
    ph.control.u.bounds = (-1, 1)
    ph.time.guess = (0.0, 1.0)
    ph.state.r.guess[:] = [(0, 1), row_guess]
    problem.ipopt_options.print_level = 0
    return problem, ph


PEAK_TIME = 0.25
PEAK_VALUE = 5.0


def test_every_row_of_a_block_contributes_its_sample_times():
    # The second row's samples are the only source of PEAK_TIME. Reading row 0 alone leaves it
    # out of the grid, and the row is then interpolated straight past its own peak.
    problem, ph = build(yapss.interp([0.0, PEAK_TIME, 1.0], [0.0, PEAK_VALUE, 0.0]))
    spec = to_transcription_spec(snapshot(problem))
    grid = spec.phases[0].guess_time
    assert PEAK_TIME in grid.tolist()
    peak = grid.tolist().index(PEAK_TIME)
    assert spec.phases[0].guess_state[1][peak] == pytest.approx(PEAK_VALUE)


def test_every_row_of_a_block_is_checked_for_coverage():
    # Samples that start halfway into the phase are refused, whichever row carries them.
    problem, ph = build(yapss.interp([0.5, 1.0], [0.0, 1.0]))
    with pytest.raises(ValueError, match="incomplete"):
        problem.validate()


def test_a_covering_sampled_row_is_accepted():
    problem, ph = build(yapss.interp([0.0, 1.0], [0.0, 1.0]))
    problem.validate()
