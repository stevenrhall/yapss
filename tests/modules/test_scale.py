"""
Scale factors are characteristic magnitudes: finite and positive, checked when set.

Through 0.2.2 only ``scale.objective`` was checked, and a NaN passed even that, since
``nan <= 0`` is false. The others were divided by in the NLP scaling and used to size
finite-difference steps; a NaN reached Ipopt's scaling arrays and crashed the process.
"""

import numpy as np
import pytest

from yapss import Problem


def _problem() -> Problem:
    return Problem(name="scale", nx=[2], nu=[1], nq=[1], nh=[1], nd=1, ns=1)


BAD_VALUES = [0.0, -1.0, np.nan, np.inf]
ARRAYS = ["state", "control", "integral", "dynamics", "path"]


@pytest.mark.parametrize("bad", BAD_VALUES)
@pytest.mark.parametrize("name", ARRAYS)
def test_phase_scale_arrays_must_be_finite_and_positive(name: str, bad: float) -> None:
    problem = _problem()
    good = np.ones_like(getattr(problem.scale.phase[0], name))
    good[0] = bad
    with pytest.raises(ValueError, match=f"Scale '{name}' in phase 0 must be finite and positive"):
        setattr(problem.scale.phase[0], name, good)
    # the failed assignment left the default in place
    assert np.all(getattr(problem.scale.phase[0], name) == 1.0)


@pytest.mark.parametrize("bad", BAD_VALUES)
@pytest.mark.parametrize("name", ["parameter", "discrete"])
def test_problem_scale_arrays_must_be_finite_and_positive(name: str, bad: float) -> None:
    problem = _problem()
    with pytest.raises(ValueError, match=f"Scale '{name}' must be finite and positive"):
        setattr(problem.scale, name, [bad])


@pytest.mark.parametrize("bad", [*BAD_VALUES, "abc"])
def test_phase_time_scale_is_validated(bad: object) -> None:
    problem = _problem()
    with pytest.raises((ValueError, TypeError)):
        problem.scale.phase[0].time = bad  # type: ignore[assignment]
    assert problem.scale.phase[0].time == 1.0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_objective_scale_rejects_non_finite(bad: float) -> None:
    problem = _problem()
    with pytest.raises(ValueError, match="must be positive"):
        problem.scale.objective = bad


def test_valid_scales_are_stored() -> None:
    problem = _problem()
    problem.scale.phase[0].state = [10.0, 0.5]
    problem.scale.phase[0].time = 100
    problem.scale.parameter = np.array([2.0])
    assert np.array_equal(problem.scale.phase[0].state, [10.0, 0.5])
    assert problem.scale.phase[0].time == 100.0
    assert problem.scale.parameter[0] == 2.0
