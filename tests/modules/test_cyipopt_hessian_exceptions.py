"""Regression tests for YAPSS's cyipopt Hessian exception workaround."""

from typing import Any, cast

import numpy as np
import pytest

from yapss._private import solver


class FakeNLP:
    """Minimal NLP surface used by the cyipopt callback adapter."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = iter(errors)
        self.intermediate_calls = 0
        self.intermediate = self._intermediate

    def hessianstructure(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (0, 1, 1), (0, 0, 1)

    def hessian(
        self,
        z: np.ndarray[Any, Any],
        lambda_: np.ndarray[Any, Any],
        objective_factor: np.float64,
    ) -> np.ndarray[Any, Any]:
        raise next(self.errors)

    def objective(self, z: np.ndarray[Any, Any]) -> float:
        return float(z[0])

    def _intermediate(self, *args: Any) -> bool:
        self.intermediate_calls += 1
        return True


@pytest.mark.parametrize("exception", [RuntimeError("hessian failed"), SystemExit(7)])
def test_adapter_latches_hessian_base_exceptions(exception: BaseException):
    """Hessian failures become initialized output until native code unwinds."""
    nlp = FakeNLP([exception])
    adapter = solver._CyipoptProblemAdapter(cast(Any, nlp))

    output = adapter.hessian(np.ones(2), np.ones(1), np.float64(1.0))

    np.testing.assert_array_equal(output, np.zeros(3))
    assert output.dtype == np.float64
    assert adapter.intermediate() is False
    assert nlp.intermediate_calls == 0
    with pytest.raises(type(exception)) as caught:
        adapter.raise_hessian_exception()
    assert caught.value is exception
    traceback_names = []
    traceback = caught.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "hessian" in traceback_names


def test_adapter_retains_first_hessian_exception():
    """A fallback Hessian evaluation cannot overwrite the original failure."""
    first = RuntimeError("first")
    second = ValueError("second")
    adapter = solver._CyipoptProblemAdapter(cast(Any, FakeNLP([first, second])))

    adapter.hessian(np.ones(2), np.ones(1), np.float64(1.0))
    adapter.hessian(np.ones(2), np.ones(1), np.float64(1.0))

    with pytest.raises(RuntimeError, match="first") as caught:
        adapter.raise_hessian_exception()
    assert caught.value is first
    adapter.raise_hessian_exception()


def test_adapter_delegates_unaffected_callbacks():
    """The adapter changes only Hessian and post-failure intermediate behavior."""
    nlp = FakeNLP([])
    adapter = solver._CyipoptProblemAdapter(cast(Any, nlp))

    assert adapter.objective(np.array([2.0])) == 2.0
    assert adapter.intermediate() is True
    assert nlp.intermediate_calls == 1


@pytest.mark.parametrize("solve_raises", [False, True])
def test_solve_reraises_latched_exception_after_cyipopt_unwinds(solve_raises: bool):
    """The user error wins over cyipopt's return status or secondary error."""
    original = RuntimeError("user objective hessian failed")
    adapter = solver._CyipoptProblemAdapter(cast(Any, FakeNLP([original])))

    class FakeCyipoptProblem:
        def solve(self, z0: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
            adapter.hessian(z0, np.ones(1), np.float64(1.0))
            if solve_raises:
                msg = "secondary cyipopt error"
                raise ValueError(msg)
            return z0, {"status": -1}

    with pytest.raises(RuntimeError, match="user objective hessian failed") as caught:
        solver._solve_ipopt_problem(
            FakeCyipoptProblem(),
            np.ones(2),
            "cyipopt",
            adapter,
        )
    assert caught.value is original
