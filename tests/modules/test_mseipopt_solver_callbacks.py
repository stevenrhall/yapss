"""Test YAPSS's adapters from NLP derivatives to mseipopt callbacks."""

import numpy as np

from yapss._private import solver


def test_derivative_callbacks_copy_returned_values():
    """Nonempty native outputs receive the arrays returned by the NLP functions."""
    x = np.array([1.0, 2.0])
    multipliers = np.array([3.0])
    jacobian_output = np.empty(2)
    hessian_output = np.empty(3)
    hessian_arguments = []

    def jacobian(argument):
        assert argument is x
        return np.array([4.0, 5.0])

    def hessian(argument, lambda_, objective_factor):
        hessian_arguments.append((argument, lambda_, objective_factor))
        return np.array([6.0, 7.0, 8.0])

    assert solver._jacobian_callback(jacobian)(x, False, jacobian_output)
    assert solver._hessian_callback(hessian)(
        x,
        False,
        2.5,
        multipliers,
        False,
        hessian_output,
    )

    np.testing.assert_array_equal(jacobian_output, [4.0, 5.0])
    np.testing.assert_array_equal(hessian_output, [6.0, 7.0, 8.0])
    assert len(hessian_arguments) == 1
    argument, lambda_, objective_factor = hessian_arguments[0]
    assert argument is x
    assert lambda_ is multipliers
    assert objective_factor == 2.5
    assert isinstance(objective_factor, np.float64)


def test_derivative_callbacks_skip_zero_length_outputs():
    """A derivative with no structural nonzeros does not evaluate user code."""
    calls = {"jacobian": 0, "hessian": 0}

    def jacobian(argument):
        calls["jacobian"] += 1
        return np.empty(0)

    def hessian(argument, multipliers, objective_factor):
        calls["hessian"] += 1
        return np.empty(0)

    empty = np.empty(0)
    assert solver._jacobian_callback(jacobian)(np.ones(1), False, empty)
    assert solver._hessian_callback(hessian)(
        np.ones(1),
        False,
        1.0,
        empty,
        False,
        empty,
    )
    assert calls == {"jacobian": 0, "hessian": 0}
