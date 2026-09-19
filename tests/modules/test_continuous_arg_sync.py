"""Tests for synchronizing numeric continuous-function arguments."""

import numpy as np
import pytest

from yapss._legacy import Problem
from yapss._private.finite_difference import get_continuous_jacobian_structure_nan
from yapss._private.input_args import ContinuousArg, ContinuousStore
from yapss._private.mesh import Mesh
from yapss._private.structure import get_nlp_dv_structure


@pytest.mark.parametrize("spectral_method", ("lg", "lgr", "lgl"))
def test_sync_updates_decision_variables_and_time(spectral_method: str) -> None:
    """Synchronizing z must update every numeric continuous input coherently."""
    problem = Problem(name="sync-test", nx=[1], nu=[1])
    problem.spectral_method = spectral_method
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(spectral_method)

    source = get_nlp_dv_structure(problem._to_spec(), np.float64)
    source.phase[0].t0[:] = 2.0
    source.phase[0].tf[:] = 5.0
    source.phase[0].xc[0][:] = 3.0
    source.phase[0].u[0][:] = 4.0

    target = get_nlp_dv_structure(problem._to_spec(), np.float64)
    store = ContinuousStore(problem._to_spec(), target, dtype=np.float64, tau_u=mesh.tau_u)
    store._sync(source.z)

    expected_time = mesh.tau_u[0] * 1.5 + 3.5
    np.testing.assert_array_equal(store._dv.z, source.z)
    np.testing.assert_allclose(store.phase[0].time, expected_time)
    np.testing.assert_array_equal(store.phase[0].state[0], source.phase[0].xc[0])
    np.testing.assert_array_equal(store.phase[0].control[0], source.phase[0].u[0])


def test_nan_structure_discovery_uses_initial_guess_time() -> None:
    """NaN sparsity discovery must evaluate the physical initial-guess time vector."""
    problem = Problem(name="time-dependent-structure", nx=[1], nu=[0], nq=[0], nh=[0])
    problem.spectral_method = "lgr"
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(problem.spectral_method)

    def continuous(arg: ContinuousArg[np.float64]) -> None:
        for p in arg.phase_list:
            state = arg.phase[p].state[0]
            time = arg.phase[p].time
            arg.phase[p].dynamics[:] = np.where(time > 0.0, state, 0.0)

    problem.functions.continuous = continuous
    dv = get_nlp_dv_structure(problem._to_spec(), np.float64)
    dv.phase[0].t0[:] = 1.0
    dv.phase[0].tf[:] = 2.0
    dv.phase[0].xc[0][:] = 3.0

    structure = get_continuous_jacobian_structure_nan(problem._to_spec(), dv.z.copy(), mesh.tau_u)

    assert structure == (((("f", 0), ("x", 0)),),)


def test_nan_probe_sees_dependency_through_where() -> None:
    """A dependency carried only by the branch ``where`` selects away is still found.

    ``where(u > 0, u, 0.0)`` at u <= 0 selects the constant branch, and a NaN probe on
    u compares False and selects it too. numpy's ``where`` would drop the NaN and the
    control column would vanish from the Jacobian; yapss.math.where propagates it.
    """
    import yapss.math as ym

    problem = Problem(name="where-structure", nx=[1], nu=[1], nq=[0], nh=[0])
    problem.spectral_method = "lgr"
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(problem.spectral_method)

    def continuous(arg: ContinuousArg[np.float64]) -> None:
        for p in arg.phase_list:
            u = arg.phase[p].control[0]
            arg.phase[p].dynamics[:] = ym.where(u > 0, u, 0.0)

    problem.functions.continuous = continuous
    dv = get_nlp_dv_structure(problem._to_spec(), np.float64)
    dv.phase[0].t0[:] = 0.0
    dv.phase[0].tf[:] = 1.0
    dv.phase[0].u[0][:] = -1.0  # the constant branch is selected everywhere

    structure = get_continuous_jacobian_structure_nan(problem._to_spec(), dv.z.copy(), mesh.tau_u)

    assert (("f", 0), ("u", 0)) in structure[0]


def test_time_is_not_assignable() -> None:
    """``phase.time`` is set by the framework; user code cannot overwrite it.

    It used to be listed among the assignable attributes as a workaround for the
    symbolic time array being patched in after construction; that left a hole in the
    typo protection for exactly the attribute a callback is most likely to touch.
    """
    problem = Problem(name="time", nx=[1], nu=[1])
    problem.mesh.phase[0].collocation_points = [3]
    problem.mesh.phase[0].fraction = [1.0]
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(problem.spectral_method)
    dv = get_nlp_dv_structure(problem._to_spec(), np.float64)
    store = ContinuousStore(problem._to_spec(), dv, dtype=np.float64, tau_u=mesh.tau_u)
    with pytest.raises(AttributeError, match="time"):
        store.value_arg.phase[0].time = np.zeros(3)


def test_symbolic_time_is_built_in_place() -> None:
    """The symbolic time array is an SXArray from construction, so time gates work."""
    from yapss._private.auto import make_args
    from yapss.math.wrapper import SXArray

    problem = Problem(name="time", nx=[1], nu=[1])
    problem.mesh.phase[0].collocation_points = [3]
    problem.mesh.phase[0].fraction = [1.0]
    _, _, continuous_arg = make_args(problem._to_spec())
    time = continuous_arg.phase[0].time
    assert isinstance(time, SXArray)
    assert isinstance(time <= 0.5, SXArray), "comparison on symbolic time must stay symbolic"


def _goddard_point(spectral_method: str):
    """Goddard's three-phase problem, its mesh, and a perturbed initial point."""
    from yapss._private.guess import make_initial_guess_nlp
    from yapss.examples import goddard_problem_3_phase

    problem = goddard_problem_3_phase.setup()
    problem.spectral_method = spectral_method
    problem.validate()
    mesh = Mesh(problem._to_spec().phases)
    mesh.set_matrices(spectral_method)
    z0 = make_initial_guess_nlp(problem._to_spec(), mesh)
    rng = np.random.default_rng(0)
    return problem, mesh, z0 * (1 + 1e-3 * rng.standard_normal(z0.size))


def _outputs(store: ContinuousStore) -> list[np.ndarray]:
    return [
        np.asarray(getattr(phase, name))
        for phase in store.phase
        for name in ("dynamics", "integrand", "path")
    ]


@pytest.mark.parametrize("spectral_method", ("lg", "lgr", "lgl"))
def test_a_node_subset_presents_the_selected_points_in_the_given_order(
    spectral_method: str,
) -> None:
    """Inputs and outputs of a subset argument are the full argument's at those nodes."""
    problem, mesh, z = _goddard_point(spectral_method)
    nodes = [np.arange(1, len(tau) - 1)[::-1] for tau in mesh.tau_u]
    full = ContinuousStore(
        problem._to_spec(),
        get_nlp_dv_structure(problem._to_spec(), np.float64),
        np.float64,
        tau_u=mesh.tau_u,
    )
    subset = ContinuousStore(
        problem._to_spec(),
        get_nlp_dv_structure(problem._to_spec(), np.float64),
        np.float64,
        tau_u=mesh.tau_u,
        nodes=nodes,
    )
    for store in (full, subset):
        store._sync(z)
        problem.functions.continuous(store.value_arg)

    for p, selected in enumerate(nodes):
        np.testing.assert_array_equal(subset.phase[p].time, full.phase[p].time[selected])
        for name in ("state", "control"):
            for part, whole in zip(
                getattr(subset.phase[p], name), getattr(full.phase[p], name), strict=True
            ):
                np.testing.assert_array_equal(part, whole[selected])
    per_phase = [nodes[p] for p in range(problem.np) for _ in range(3)]
    for part, whole, selected in zip(_outputs(subset), _outputs(full), per_phase, strict=True):
        assert part.shape == (whole.shape[0], len(selected))
        np.testing.assert_allclose(part, whole[:, selected], rtol=1e-12, atol=0.0)


def test_sync_refreshes_the_inputs_of_a_node_subset() -> None:
    """The subset's inputs are copies, so every sync must rewrite them."""
    problem, mesh, z = _goddard_point("lgr")
    nodes = [np.array([3, 1]) for _ in mesh.tau_u]
    store = ContinuousStore(
        problem._to_spec(),
        get_nlp_dv_structure(problem._to_spec(), np.float64),
        np.float64,
        tau_u=mesh.tau_u,
        nodes=nodes,
    )
    store._sync(z)
    store._sync(2 * z)
    doubled = get_nlp_dv_structure(problem._to_spec(), np.float64)
    doubled.z[:] = 2 * z
    np.testing.assert_array_equal(store.phase[1].state[2], doubled.phase[1].xc[2][[3, 1]])
    np.testing.assert_array_equal(store.phase[1].control[0], doubled.phase[1].u[0][[3, 1]])


def test_nodes_are_for_numeric_arguments_with_one_array_per_phase() -> None:
    problem, mesh, _ = _goddard_point("lgr")
    with pytest.raises(ValueError, match="one array per phase"):
        ContinuousStore(
            problem._to_spec(),
            get_nlp_dv_structure(problem._to_spec(), np.float64),
            np.float64,
            tau_u=mesh.tau_u,
            nodes=[np.array([1])],
        )
    with pytest.raises(ValueError, match="numeric ContinuousStore instances only"):
        ContinuousStore(
            problem._to_spec(),
            get_nlp_dv_structure(problem._to_spec(), np.object_),
            np.object_,
            nodes=[np.array([1])] * problem.np,
        )
