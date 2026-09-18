"""

What the transcription is given: a problem reduced to numbers.

`ProblemSpec` is the whole contract between a front end and everything downstream of it -- the
mesh, the derivative methods, the NLP assembly, and the Ipopt binding. It is plain data:
counts, read-only arrays, a few settings, and the user's callbacks. It carries no validation
and no user-facing messages, because those belong to whichever front end produced it, and
nothing in it can be edited once it is made, so a solve can never be altered by a later change
to the problem it came from.

Two front ends produce one: `yapss.Problem._to_spec`, and the redesigned API in `yapss._next`.
Neither is visible from here, which is the point -- the transcription has no idea which it is
serving.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = ["PhaseSpec", "ProblemSpec", "VariableScale", "frozen_array"]

Sense = Literal["minimize", "maximize"]
SpectralMethod = Literal["lgl", "lgr", "lg"]
DerivativeMethod = Literal["auto", "central-difference", "central-difference-full", "user"]
DerivativeOrder = Literal["first", "second"]


def frozen_array(values: Any, size: int | None = None) -> NDArray[np.float64]:
    """Return `values` as a read-only float array, so that a spec cannot be edited in place.

    Parameters
    ----------
    values : array_like
        The values to freeze. A scalar is broadcast when `size` is given.
    size : int, optional
        The length the result must have.

    Returns
    -------
    numpy.ndarray
        A read-only copy.
    """
    array = np.array(values, dtype=float, copy=True)
    if size is not None and array.shape == ():
        array = np.full(size, float(array))
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class PhaseSpec:
    """One phase, reduced to numbers.

    Every ``*_lower``/``*_upper`` pair is a bound, every ``*_scale`` is positive, and the guess
    arrays are samples on ``guess_time``, which the transcription interpolates onto the mesh.
    """

    index: int
    nx: int
    nu: int
    nq: int
    nh: int

    state_lower: NDArray[np.float64]
    state_upper: NDArray[np.float64]
    initial_state_lower: NDArray[np.float64]
    initial_state_upper: NDArray[np.float64]
    final_state_lower: NDArray[np.float64]
    final_state_upper: NDArray[np.float64]
    control_lower: NDArray[np.float64]
    control_upper: NDArray[np.float64]
    path_lower: NDArray[np.float64]
    path_upper: NDArray[np.float64]
    integral_lower: NDArray[np.float64]
    integral_upper: NDArray[np.float64]
    # Bounds on the LGL zero modes, which are internal to the transcription and are not part
    # of any front end's surface; they are always free.
    zero_mode_lower: NDArray[np.float64]
    zero_mode_upper: NDArray[np.float64]
    initial_time_lower: float
    initial_time_upper: float
    final_time_lower: float
    final_time_upper: float
    duration_lower: float
    duration_upper: float

    state_scale: NDArray[np.float64]
    control_scale: NDArray[np.float64]
    integral_scale: NDArray[np.float64]
    dynamics_scale: NDArray[np.float64]
    path_scale: NDArray[np.float64]
    time_scale: float

    guess_time: NDArray[np.float64]
    guess_state: NDArray[np.float64]
    guess_control: NDArray[np.float64]
    guess_integral: NDArray[np.float64]

    fraction: tuple[float, ...]
    collocation_points: tuple[int, ...]


class VariableScale:
    """The characteristic magnitude of one decision variable, by key.

    Used by the finite-difference methods to size their perturbation steps. It relies on the
    scaling model having one scale per state and one per phase time: ``x``, ``x0`` and ``xf``
    share the state scale, and ``t``, ``t0`` and ``tf`` share the time scale. If separate
    endpoint scales are ever introduced, this mapping must be split accordingly.
    """

    __slots__ = ("_spec",)

    def __init__(self, spec: ProblemSpec) -> None:
        self._spec = spec

    def __getitem__(self, item: tuple[int, str, int]) -> float:
        """Return the characteristic magnitude of the decision variable `item` names."""
        p, v, i = item
        if v == "s":
            return float(self._spec.parameter_scale[i])
        phase = self._spec.phases[p]
        if v in ("x", "x0", "xf"):
            return float(phase.state_scale[i])
        if v == "u":
            return float(phase.control_scale[i])
        if v in ("t", "t0", "tf"):
            return phase.time_scale
        if v == "q":
            return float(phase.integral_scale[i])
        msg = f"no scale for variable kind {v!r}"
        raise KeyError(msg)


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A whole problem, reduced to numbers and callbacks.

    Attributes
    ----------
    phases : tuple of PhaseSpec
        The phases, in order.
    functions : UserFunctions
        The callbacks, in the protocol the transcription calls them with.
    auxdata : Any
        Carried through to the callback arguments untouched. It belongs to the released API's
        callback surface; the redesigned API has no use for it and leaves it empty.
    """

    name: str
    phases: tuple[PhaseSpec, ...]
    nd: int
    ns: int

    discrete_lower: NDArray[np.float64]
    discrete_upper: NDArray[np.float64]
    parameter_lower: NDArray[np.float64]
    parameter_upper: NDArray[np.float64]

    discrete_scale: NDArray[np.float64]
    parameter_scale: NDArray[np.float64]
    objective_scale: float

    guess_parameter: NDArray[np.float64]

    functions: Any
    auxdata: Any

    sense: Sense
    spectral_method: SpectralMethod
    derivative_method: DerivativeMethod
    derivative_order: DerivativeOrder
    ipopt_options: dict[str, Any]
    catch_keyboard_interrupt: bool
    intermediate_callback: Any = None

    @property
    def variable_scale(self) -> VariableScale:
        """Return the characteristic magnitude of each decision variable, by key."""
        return VariableScale(self)

    @property
    def np(self) -> int:
        """Return the number of phases."""
        return len(self.phases)

    @property
    def nx(self) -> tuple[int, ...]:
        """Return the number of states in each phase."""
        return tuple(phase.nx for phase in self.phases)

    @property
    def nu(self) -> tuple[int, ...]:
        """Return the number of controls in each phase."""
        return tuple(phase.nu for phase in self.phases)

    @property
    def nq(self) -> tuple[int, ...]:
        """Return the number of integrals in each phase."""
        return tuple(phase.nq for phase in self.phases)

    @property
    def nh(self) -> tuple[int, ...]:
        """Return the number of path constraints in each phase."""
        return tuple(phase.nh for phase in self.phases)

    def mesh_phases(self) -> Sequence[Any]:
        """Return the per-phase mesh records the `Mesh` builder reads.

        Returns
        -------
        Sequence
            One record per phase, each with ``fraction`` and ``collocation_points``.
        """
        return self.phases
