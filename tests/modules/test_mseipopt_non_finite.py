"""The binding's own check that a values callback returned finite numbers.

Ipopt's ``check_derivatives_for_naninf`` is left off, because with it on Ipopt crashes whenever
a constraint evaluation reports failure (coin-or/Ipopt#865). The binding scans every values
callback's result instead and reports a non-finite one to Ipopt as a failed evaluation, which
is what the option did, and remembers it for the message when Ipopt then gives up.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.modules.test_mseipopt_bare_np import make_problem, native  # noqa: F401
from yapss._backend.mseipopt import bare_np


def test_a_non_finite_result_is_a_failed_evaluation_and_is_remembered(native):  # noqa: F811
    problem = make_problem()
    output = np.array([1.0, np.nan, np.inf])
    assert problem._invoke_callback("eval_g", "values", lambda: True, output=output) is False
    assert problem._non_finite == bare_np.NonFinite("eval_g", 2, 3, 1)
    assert str(problem._non_finite) == (
        "eval_g returned a NaN or Inf: 2 of 3 entries, the first at index 1"
    )
    # a finite result passes, and the last non-finite one stays remembered
    assert problem._invoke_callback("eval_g", "values", lambda: True, output=np.ones(3)) is True
    assert problem._non_finite == bare_np.NonFinite("eval_g", 2, 3, 1)


def test_a_result_the_callback_reported_as_failed_is_not_scanned(native):  # noqa: F811
    problem = make_problem()
    output = np.array([np.nan])
    assert problem._invoke_callback("eval_f", "values", lambda: False, output=output) is False
    assert problem._non_finite is None


def test_one_value_is_described_as_one(native):  # noqa: F811
    problem = make_problem()
    assert not problem._invoke_callback("eval_f", "values", lambda: True, output=np.array(np.nan))
    assert str(problem._non_finite) == "eval_f returned a NaN or Inf: value, the first at index 0"


def test_the_record_is_cleared_between_solves(native):  # noqa: F811
    problem = make_problem()
    problem._invoke_callback("eval_g", "values", lambda: True, output=np.array([np.nan]))
    problem._clear_termination()
    assert problem._non_finite is None


@pytest.mark.parametrize("name", ["eval_f", "eval_g", "eval_grad_f", "eval_jac_g", "eval_h"])
def test_a_persistent_nan_in_any_callback_stops_ipopt_cleanly(name):
    """Through real Ipopt: a callback that always returns NaN ends with a failure status.

    This is every wrapper handing its result to the check: were one not, Ipopt would run on
    the NaN and the record would be empty.
    """
    from yapss._backend.mseipopt import library

    library.initialize_ipopt()

    def value(callback, finite):
        return np.nan if callback == name else finite

    def eval_f(x, new_x, out):
        out[()] = value("eval_f", (x[0] - 1.0) ** 2)
        return True

    def eval_g(x, new_x, out):
        out[0] = value("eval_g", x[0])
        return True

    def eval_grad_f(x, new_x, out):
        out[0] = value("eval_grad_f", 2 * (x[0] - 1.0))
        return True

    def eval_jac_g(x, new_x, out):
        out[0] = value("eval_jac_g", 1.0)
        return True

    def eval_h(x, new_x, obj_factor, multipliers, new_multipliers, out):
        out[0] = value("eval_h", 2 * obj_factor)
        return True

    with bare_np.Problem(
        [-10.0],
        [10.0],
        [0.0],
        [0.0],
        eval_f=eval_f,
        eval_g=eval_g,
        eval_grad_f=eval_grad_f,
        jacobian_structure=([0], [0]),
        eval_jac_g=eval_jac_g,
        hessian_structure=([0], [0]),
        eval_h=eval_h,
    ) as problem:
        problem.add_int_option("print_level", 0)
        problem.add_str_option("sb", "yes")
        result = problem.solve(np.array([0.5]))
    assert result.status not in (0, 1)
    assert result.non_finite is not None
    assert result.non_finite.callback == name
