"""Fixed variables, and problems with fewer free variables than equality constraints.

YAPSS sets and reserves ``fixed_variable_treatment = "make_parameter"``: Ipopt removes a variable
whose bounds are equal. When too few variables would be left free, Ipopt keeps them instead, as
``relax_bounds``, and a consistent but redundant problem still solves.
`bare_np.relaxes_fixed_variables` reproduces Ipopt's count, so that YAPSS knows which case holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.support import legacy as yapss
from tests.support.legacy import Problem
from yapss._backend.bounds import (
    get_nlp_constraint_function_bounds,
    get_nlp_decision_variable_bounds,
)
from yapss._backend.mseipopt import bare_np


def test_fixed_variable_treatment_is_reserved() -> None:
    problem = Problem(name="t", nx=[], ns=1)
    with pytest.raises(ValueError, match="managed by YAPSS"):
        problem.ipopt_options.fixed_variable_treatment = "make_constraint"


def relaxes(problem: Problem) -> bool:
    spec = problem._to_spec()
    ub, lb = get_nlp_decision_variable_bounds(spec)
    gu, gl = get_nlp_constraint_function_bounds(spec)
    return bare_np.relaxes_fixed_variables(lb, ub, gl, gu)


def redundant_parameters() -> Problem:
    """x fixed by its bounds and y free, with y = 1 and x + y = 1: one free variable, two equalities."""
    problem = Problem(name="redundant", nx=[], ns=2, nd=2)

    def objective(arg: yapss.ObjectiveArg) -> None:
        arg.objective = arg.parameter[1] ** 2

    def discrete(arg: yapss.DiscreteArg) -> None:
        x, y = arg.parameter
        arg.discrete[:] = y, x + y

    problem.functions.objective = objective
    problem.functions.discrete = discrete
    problem.bounds.parameter.lower[:] = 0.0, -np.inf  # x fixed at 0, y free
    problem.bounds.parameter.upper[:] = 0.0, np.inf
    problem.bounds.discrete.lower[:] = problem.bounds.discrete.upper[:] = 1.0
    problem.guess.parameter = [0.0, 1.0]
    problem.ipopt_options.print_level = 0
    return problem


def pinned_final_state(spectral_method: str) -> Problem:
    """With no control, the initial state and the dynamics already determine the final state."""
    problem = Problem(name="pinned", nx=[1], nu=[0], nd=1)

    def continuous(arg: yapss.ContinuousArg) -> None:
        (x,) = arg.phase[0].state
        arg.phase[0].dynamics[0] = -x

    def objective(arg: yapss.ObjectiveArg) -> None:
        arg.objective = arg.phase[0].final_state[0]

    def discrete(arg: yapss.DiscreteArg) -> None:
        arg.discrete[0] = arg.phase[0].final_state[0]

    problem.functions.continuous = continuous
    problem.functions.objective = objective
    problem.functions.discrete = discrete
    problem.spectral_method = spectral_method
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower[0] = bounds.initial_state.upper[0] = 1.0
    problem.bounds.discrete.lower[:] = problem.bounds.discrete.upper[:] = np.exp(-1.0)
    guess = problem.guess.phase[0]
    guess.time, guess.state = [0.0, 1.0], [[1.0, np.exp(-1.0)]]
    problem.ipopt_options.print_level = 0
    return problem


def test_redundant_parameters_solve() -> None:
    problem = redundant_parameters()
    assert relaxes(problem)
    solution = problem.solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    np.testing.assert_allclose(solution.parameter, [0.0, 1.0], atol=1e-8)


@pytest.mark.parametrize("spectral_method", ["lg", "lgr", "lgl"])
def test_pinned_final_state_solves(spectral_method: str) -> None:
    problem = pinned_final_state(spectral_method)
    assert relaxes(problem)
    solution = problem.solve()
    assert solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED
    assert solution.objective == pytest.approx(np.exp(-1.0), rel=1e-8)


def test_an_ordinary_problem_keeps_its_fixed_variables_removed() -> None:
    from tests.support.legacy.examples import brachistochrone_minimal

    assert not relaxes(brachistochrone_minimal.setup())
