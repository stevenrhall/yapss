"""A function singular at a fixed value solves through the 0.4 API.

A start from rest with drag through ``sqrt(vx**2 + vy**2)`` has, under ``"auto"``, a NaN
derivative with respect to the initial state, which is fixed. Ipopt removes a fixed variable and
never uses that derivative, so the problem solves, and the initial state's bound multiplier,
which Ipopt computes from it, is NaN.
"""

import numpy as np

import yapss
from yapss.math import sqrt


class State(yapss.State):
    vx = yapss.scalar()
    vy = yapss.scalar()


class Control(yapss.Control):
    u = yapss.scalar()


class Phase(yapss.Phase):
    state: State
    control: Control


class Phases(yapss.Phases):
    phase: Phase


class FromRest(yapss.Problem):
    phases: Phases


def test_a_start_from_rest_with_drag_solves():
    problem = FromRest("from rest")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg, out):
        vx, vy, u = arg.state.vx, arg.state.vy, arg.control.u
        v = sqrt(vx**2 + vy**2)
        out.dynamics.vx = u - 0.1 * v * vx
        out.dynamics.vy = 1.0 - 0.1 * v * vy

    @problem.register.objective
    def objective(arg):
        return arg[ph].final_state.vx

    ph.initial_time.bounds = (0.0, 0.0)
    ph.final_time.bounds = (1.0, 1.0)
    for field in (ph.initial_state.vx, ph.initial_state.vy):
        field.bounds = (0.0, 0.0)
    for field in (ph.state.vx, ph.state.vy, ph.control.u):
        field.bounds = (-10.0, 10.0)
    ph.time.guess = (0.0, 1.0)
    ph.state.vx.guess = (0.1, 1.0)
    ph.state.vy.guess = (0.1, 1.0)
    ph.control.u.guess = (0.5, 0.5)
    problem.ipopt_options.print_level = 0
    solution = problem.solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    multiplier = solution.phases.phase.multiplier.initial_state
    assert np.isnan(multiplier.vx)
    assert np.isnan(multiplier.vy)
