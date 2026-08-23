"""Tests for synchronizing numeric continuous-function arguments."""

import numpy as np
import pytest

from yapss import Problem
from yapss._private.finite_difference import get_continuous_jacobian_structure_nan
from yapss._private.input_args import ContinuousArg
from yapss._private.mesh import Mesh
from yapss._private.structure import get_nlp_dv_structure


@pytest.mark.parametrize("spectral_method", ("lg", "lgr", "lgl"))
def test_sync_updates_decision_variables_and_time(spectral_method: str) -> None:
    """Synchronizing z must update every numeric continuous input coherently."""
    problem = Problem(name="sync-test", nx=[1], nu=[1])
    problem.spectral_method = spectral_method
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(spectral_method)

    source = get_nlp_dv_structure(problem, np.float64)
    source.phase[0].t0[:] = 2.0
    source.phase[0].tf[:] = 5.0
    source.phase[0].xc[0][:] = 3.0
    source.phase[0].u[0][:] = 4.0

    target = get_nlp_dv_structure(problem, np.float64)
    arg = ContinuousArg(problem, target, dtype=np.float64, tau_u=mesh.tau_u)
    arg._sync(source.z)

    expected_time = mesh.tau_u[0] * 1.5 + 3.5
    np.testing.assert_array_equal(arg._dv.z, source.z)
    np.testing.assert_allclose(arg.phase[0].time, expected_time)
    np.testing.assert_array_equal(arg.phase[0].state[0], source.phase[0].xc[0])
    np.testing.assert_array_equal(arg.phase[0].control[0], source.phase[0].u[0])


def test_nan_structure_discovery_uses_initial_guess_time() -> None:
    """NaN sparsity discovery must evaluate the physical initial-guess time vector."""
    problem = Problem(name="time-dependent-structure", nx=[1], nu=[0], nq=[0], nh=[0])
    problem.spectral_method = "lgr"
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)

    def continuous(arg: ContinuousArg[np.float64]) -> None:
        for p in arg.phase_list:
            state = arg.phase[p].state[0]
            time = arg.phase[p].time
            arg.phase[p].dynamics[:] = np.where(time > 0.0, state, 0.0)

    problem.functions.continuous = continuous
    dv = get_nlp_dv_structure(problem, np.float64)
    dv.phase[0].t0[:] = 1.0
    dv.phase[0].tf[:] = 2.0
    dv.phase[0].xc[0][:] = 3.0

    structure = get_continuous_jacobian_structure_nan(problem, dv.z.copy(), mesh.tau_u)

    assert structure == (((("f", 0), ("x", 0)),),)


def test_time_is_not_assignable() -> None:
    """``phase.time`` is set by the framework; user code cannot overwrite it.

    It used to be listed among the assignable attributes as a workaround for the
    symbolic time array being patched in after construction; that left a hole in the
    typo protection for exactly the attribute a callback is most likely to touch.
    """
    problem = Problem(name="time", nx=[1], nu=[1])
    problem.mesh.phase[0].collocation_points = [3]
    problem.mesh.phase[0].fraction = [1.0]
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)
    dv = get_nlp_dv_structure(problem, np.float64)
    arg = ContinuousArg(problem, dv, dtype=np.float64, tau_u=mesh.tau_u)
    with pytest.raises(AttributeError, match="time"):
        arg.phase[0].time = np.zeros(3)


def test_symbolic_time_is_built_in_place() -> None:
    """The symbolic time array is an SXArray from construction, so time gates work."""
    from yapss._private.auto import make_args
    from yapss.math.wrapper import SXArray

    problem = Problem(name="time", nx=[1], nu=[1])
    problem.mesh.phase[0].collocation_points = [3]
    problem.mesh.phase[0].fraction = [1.0]
    _, _, continuous_arg = make_args(problem)
    time = continuous_arg.phase[0].time
    assert isinstance(time, SXArray)
    assert isinstance(time <= 0.5, SXArray), "comparison on symbolic time must stay symbolic"
