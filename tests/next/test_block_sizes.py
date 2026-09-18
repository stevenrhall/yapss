"""End-to-end solves of a declaration whose block fields have one row and none.

A vector class built by a function takes its field sizes from an algorithm, so the sizes one
and zero are reachable. They are allowed for that reason, and what makes them useful is that a
block field keeps its leading axis at every size: the callback that works for three rows works
unchanged for one and for none.
"""

import math

import numpy as np
import pytest

from yapss import _next as yapss
from yapss.math import cos, sin

G0 = 32.174
ANALYTIC = math.sqrt(math.pi / G0)
"""The brachistochrone's minimum time, which this problem is a re-spelling of."""


def declarations():
    """Return the brachistochrone's classes, with a one-row block and an empty one."""

    class Slide(yapss.Vector):
        x = yapss.field(units="ft")
        y = yapss.field(size=1, units="ft")
        v = yapss.field(units="ft/s")
        spare = yapss.field(size=0, units="ft")

    class Angle(yapss.Vector):
        u = yapss.field(size=1, units="rad")

    return Slide, Angle


@pytest.fixture
def problem():
    slide_class, angle_class = declarations()

    class Phases(yapss.Phases):
        slide = yapss.phase(state=slide_class, control=angle_class)

    problem = yapss.Problem("block sizes", phases=Phases)
    ph = problem.phases.slide
    shapes = {}

    @ph.continuous
    def slide(arg, out):
        shapes["x"] = np.shape(arg.state.x)
        shapes["y"] = np.shape(arg.state.y)
        shapes["spare"] = np.shape(arg.state.spare)
        shapes["u"] = np.shape(arg.control.u)
        v, u = arg.state.v, arg.control.u[0]
        out.dynamics.x = v * cos(u)
        out.dynamics.y = v * sin(u)
        out.dynamics.v = G0 * sin(u)
        out.dynamics.spare = []
        return out

    @problem.objective_function
    def minimum_time(arg):
        return arg[ph].final_time

    ph.time.initial = 0.0
    ph.state.initial.x = 0.0
    ph.state.initial.y = 0.0
    ph.state.initial.v = 0.0
    ph.state.final.x = 1.0
    ph.state.bounds.x = (0, 10)
    ph.state.bounds.y = (0, 10)
    ph.state.bounds.v = (0, 10)
    ph.control.bounds.u = (-math.pi / 2, math.pi / 2)
    ph.time.guess = (0.0, 1.0)
    ph.state.guess.x = (0, 1)
    ph.state.guess.y = (0, 1)
    ph.state.guess.v = (0, 5)
    problem.ipopt_options.print_level = 0
    return problem, shapes


@pytest.mark.parametrize("method", ["auto", "central-difference", "central-difference-full"])
def test_one_row_and_empty_blocks_solve(problem, method):
    problem, _ = problem
    problem.derivatives.method = method
    assert problem.solve().objective == pytest.approx(ANALYTIC, rel=1e-6)


def test_a_block_keeps_its_axis_in_the_callback(problem):
    problem, shapes = problem
    problem.derivatives.method = "central-difference"
    problem.solve()
    npoints = shapes["x"][0]
    assert shapes["x"] == (npoints,)
    assert shapes["y"] == (1, npoints)
    assert shapes["u"] == (1, npoints)
    assert shapes["spare"] == (0, npoints)


def test_a_block_keeps_its_axis_in_the_solution(problem):
    problem, _ = problem
    solution = problem.solve()
    ps = solution[problem.phases.slide]
    npoints = ps.state.x.shape[0]
    assert ps.state.y.shape == (1, npoints)
    assert ps.state.spare.shape == (0, npoints)
    assert ps.control.u.shape == (1, npoints)
