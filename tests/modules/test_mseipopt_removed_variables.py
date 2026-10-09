"""The binding leaves derivatives with respect to a removed variable out of its finite check.

Under ``make_parameter`` Ipopt removes a variable whose bounds are equal and never uses a
derivative with respect to it, so a NaN or an infinity there is not a failed evaluation. Every
other derivative entry is still checked, since a non-finite one reaches Ipopt's linear solver
(an infinite Jacobian entry at the start crashes MUMPS). Where Ipopt keeps the fixed variables
because too few are left free, every entry is checked.
"""

from __future__ import annotations

import numpy as np

from tests.modules.test_mseipopt_bare_np import (  # noqa: F401
    callbacks,
    invoke_native_callback,
    make_problem,
    native,
)
from yapss._backend.mseipopt import bare


def writing(values):
    """A callback that writes `values` to its output and succeeds."""

    def callback(x, new_x, out):
        out[:] = values
        return True

    return callback


def problem_with(**overrides):
    """Two variables, the first fixed at 0; one equality constraint; jacobian entries (0, 1), (0, 0)."""
    arguments = {"x_l": [0.0, -2.0], "x_u": [0.0, 2.0], "g_l": [0.0], "g_u": [0.0]}
    arguments.update(overrides)
    return make_problem(**arguments)


def test_a_removed_variable_column_is_not_checked(native):
    problem = problem_with(eval_jac_g=writing([0.0, np.inf]))  # entry 1 is column 0, fixed
    problem.remove_fixed_variables()
    assert invoke_native_callback(problem, "eval_jac_g") is True


def test_a_free_variable_column_is_still_checked(native):
    problem = problem_with(eval_jac_g=writing([np.inf, 0.0]))  # entry 0 is column 1, free
    problem.remove_fixed_variables()
    assert invoke_native_callback(problem, "eval_jac_g") is False


def test_the_gradient_follows_the_same_rule(native):
    problem = problem_with(eval_grad_f=writing([np.nan, 1.0]))
    problem.remove_fixed_variables()
    assert invoke_native_callback(problem, "eval_grad_f") is True
    problem = problem_with(eval_grad_f=writing([1.0, np.nan]))
    problem.remove_fixed_variables()
    assert invoke_native_callback(problem, "eval_grad_f") is False


def test_without_removal_every_entry_is_checked(native):
    problem = problem_with(eval_jac_g=writing([0.0, np.inf]))
    assert invoke_native_callback(problem, "eval_jac_g") is False


def test_when_ipopt_keeps_the_fixed_variables_every_entry_is_checked(native):
    """One fixed variable, one free, two equalities: Ipopt switches to relax_bounds itself."""
    problem = problem_with(
        g_l=[0.0, 0.0],
        g_u=[0.0, 0.0],
        jacobian_structure=([0, 1], [1, 0]),
        eval_g=writing([0.0, 0.0]),
        eval_jac_g=writing([0.0, np.inf]),  # entry 1 is column 0, fixed
    )
    problem.remove_fixed_variables()
    assert invoke_native_callback(problem, "eval_jac_g") is False


def test_a_removed_variable_with_a_non_finite_derivative_has_nan_multipliers(native, monkeypatch):
    problem = problem_with(eval_grad_f=writing([np.nan, 1.0]))
    problem.remove_fixed_variables()

    def solve(*args):
        assert invoke_native_callback(problem, "eval_grad_f") is True
        return 0

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    mult_x_l, mult_x_u = np.array([3.0, 4.0]), np.array([5.0, 6.0])
    result = problem.solve(np.zeros(2), mult_x_L=mult_x_l, mult_x_U=mult_x_u)
    assert np.isnan(result.mult_x_L[0]) and np.isnan(result.mult_x_U[0])
    assert result.mult_x_L[1] == 4.0 and result.mult_x_U[1] == 6.0
