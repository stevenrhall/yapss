"""

Module input_args.

This module defines arguments which are used to call the user-defined callback functions.
"""

# future imports
from __future__ import annotations

from collections.abc import Callable

# standard inputs
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, assert_never, cast

# third party imports
import numpy  # noqa: ICN001
from casadi import SX

from yapss.math.wrapper import SXW, sx_array

# package imports
from .layout import problem_layout
from .outputs import Output, OutputArray
from .types_ import Protected, set_private

if TYPE_CHECKING:
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .problem import Auxdata
    from .structure import DVPhase, DVStructure
    from .types_ import (
        CHFDS,
        CHS,
        CJFDS,
        CJS,
        DHFDS,
        DHS,
        DJFDS,
        DJS,
        OGS,
        OHS,
        CFIndex,
        CFKey,
        CFName,
        CVIndex,
        CVKey,
        CVName,
        DFIndex,
        DVKey,
        OHSTerm,
        PhaseIndex,
    )

from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

# Define a generic type for DVStructure elements
T = TypeVar("T", bound=np.generic)


def _check_keys(arg: Any) -> None:
    """Check that a derivative callback set the key set its structure was deduced from.

    Under the ``"user"`` method the key set *is* the sparsity structure, and the structure is
    deduced from one call at the initial guess. A key that appears later is outside the
    structure and would be ignored; a key that stops appearing leaves the assembly without a
    value. Neither can be intended: a derivative that is structurally zero is expressed by
    never declaring the key, and one that is zero at this point by assigning ``0.0``.

    This raises where an unassigned output row only warns. Whether that split is right is an
    open question for 0.3.0 -- see W9b in the work list; raising is the direction that can
    still be relaxed after release.
    """
    for (where, entries), expected in zip(arg._key_groups(), arg._expected_keys, strict=True):
        if entries.keys() == expected:
            continue
        actual = set(entries)
        missing = sorted(expected - actual, key=repr)
        extra = sorted(actual - expected, key=repr)
        if missing:
            msg = (
                f"the {arg._callback} callback did not set {missing[0]!r}{where} on this "
                f"call. The sparsity structure is taken from the first call, so every call "
                f"must set the same keys; assign 0.0 for an entry that is zero here."
            )
        else:
            msg = (
                f"the {arg._callback} callback set {extra[0]!r}{where}, which the first call "
                f"did not set and which is not in the sparsity structure. The structure is "
                f"taken from the first call, so every call must set the same keys."
            )
        raise ValueError(msg)


def require_keys(arg: Any, structure: Any, *, per_phase: bool = False) -> None:
    """Require ``arg``'s callback to set exactly the keys its structure was deduced from.

    Called only for the ``"user"`` method, and only once each, when the NLP is built: under
    the other methods the generated callbacks emit their structure by construction, and
    `BaseArg._expected_keys` stays None so nothing is checked.
    """
    if structure is None:
        return
    groups = (
        tuple(frozenset(phase) for phase in structure) if per_phase else (frozenset(structure),)
    )
    set_private(arg, "_expected_keys", groups)


def callback_location(function: Callable[..., Any]) -> str:
    """Name a callback and, when it has one, the file and line of its ``def``.

    A callable object is named by its class, at the ``def`` of its ``__call__``: its ``repr``
    carries a memory address, which names nothing a user can find.
    """
    code = getattr(function, "__code__", None)
    name = getattr(function, "__qualname__", None)
    if name is None:
        name = type(function).__qualname__
        code = getattr(type(function).__call__, "__code__", None)
    return name if code is None else f"{name} ({code.co_filename}, line {code.co_firstlineno})"


_SYMBOLIC_HINT = (
    'Under the "auto" derivative method the callback is called with symbolic inputs, which '
    "functions from math and other numeric libraries cannot take. Use the functions of "
    "yapss.math instead."
)


def call_callback(function: Any, arg: Any) -> None:
    """Call one user callback, under the rules every callback obeys.

    Every call of a user-supplied function goes through here. The callback's outputs are
    cleared first, so a row the callback does not assign on this call is zero rather than
    whatever the previous call left there. A value returned instead of assigned is refused,
    naming the idiom to use. Each argument clears only its own outputs (E7c), and a derivative
    callback's key set must be the same on every call (E7b), checked after the call.

    An exception raised by the callback propagates unchanged -- same object, message, and
    traceback -- with a note naming the callback and the line of its ``def``, since the
    traceback alone does not say which of the user's functions YAPSS was calling. When the
    call was the symbolic trace of the ``"auto"`` method and the error is the kind a
    float-only function raises on a symbol, a second note points to `yapss.math`.
    """
    arg._reset()
    try:
        result = function(arg)
    except Exception as exc:
        # YAPSS's own derivative functions pass through here too (central difference wraps
        # the user's callbacks); the note is for the user's function, which is innermost
        if getattr(function, "__module__", "").startswith("yapss._private"):
            raise
        exc.add_note(f"Raised in functions.{arg._callback} = {callback_location(function)}.")
        if (
            arg._dtype is np.object_
            and isinstance(exc, (TypeError, NotImplementedError))
            and "yapss.math" not in str(exc)
        ):
            exc.add_note(_SYMBOLIC_HINT)
        raise
    if arg._expected_keys is not None:
        _check_keys(arg)
    if result is not None:
        msg = (
            f"the {arg._callback} callback returned a {type(result).__name__}; a callback "
            f"assigns its results to the argument and returns nothing: {arg._results_in}"
        )
        raise TypeError(msg)


def read_only(array: NDArray[Any]) -> NDArray[Any]:
    """Return a view of ``array`` that a callback cannot write into.

    The view tracks the array it is taken from, so a callback always reads the values of the
    point it was called at; only writing is refused, at the user's own line, with NumPy's
    message for a read-only array.
    """
    view = array.view()
    view.flags.writeable = False
    return view


class BaseArg(Generic[T]):
    """Base class for all argument classes.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class containing problem data.
    dv : DVStructure
        A structure for decision variables.
    """

    _callback: str
    """The callback this argument is passed to, as `problem.functions` spells it."""

    _results_in: str
    """The assignment a user makes instead of returning a value."""

    _expected_keys: tuple[frozenset[Any], ...] | None = None
    """Key set per group the callback must set on every call, or None to not check.

    Set only for the ``"user"`` method, where the keys come from the user. The generated
    callbacks of the other methods emit exactly their structure by construction.
    """

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        """(phase description, entries) for each key set that must stay invariant."""
        return ()

    def _reset(self) -> None:
        """Clear whatever the callback assigns."""

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        self.auxdata = problem.auxdata
        self._dv: DVStructure[T] = dv
        self._parameter: NDArray[T] = read_only(dv.s)
        self._dtype: type[T] = dtype

    @property
    def parameter(self) -> NDArray[T]:
        """Return the parameter vector."""
        return self._parameter


class DiscreteArgBase(BaseArg[T], Generic[T]):
    """Base class for ObjectiveArg, DiscreteArg, etc.

    This class serves as the base for ObjectiveArg, ObjectiveGradientArg, ObjectiveHessianArg,
    DiscreteArg, DiscreteJacobianArg, and DiscreteHessianArg classes.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class.
    dv : DVStructure
        A structure for decision variables.
    """

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        super().__init__(problem, dv, dtype)
        # _phase is a tuple of DiscretePhase instances, assuming they are not parameterized by T
        self._phase: tuple[DiscretePhase[T], ...] = tuple(DiscretePhase(p, dtype) for p in dv.phase)
        # _parameter and _dv use the generic type T
        self._parameter: NDArray[T] = read_only(dv.s)
        self._dv: DVStructure[T] = dv

    @property
    def phase(self) -> tuple[DiscretePhase[T], ...]:
        """Return the tuple of DiscretePhase objects."""
        return self._phase


class DiscretePhase(Protected, Generic[T]):
    """Defines the discrete phase object.

    This class defines the `phase` attribute of the argument passed to user-defined
    objective and discrete functions. It is designed to be effectively immutable, so
    users cannot inadvertently modify the decision variables.

    Parameters
    ----------
    dv_phase : DVPhase
        A single element of the `phase` attribute from the decision variable structure.
    """

    def __init__(self, dv_phase: DVPhase[T], dtype: type[T]) -> None:
        # Initialize attributes as arrays with the generic type T
        self._dtype: type[T] = dtype
        self._initial_state: NDArray[T] = dv_phase.x0
        self._final_state: NDArray[T] = dv_phase.xf
        self._initial_time: NDArray[T] = dv_phase.t0
        self._final_time: NDArray[T] = dv_phase.tf
        self._integral: NDArray[T] = dv_phase.q

    @property
    def initial_time(self) -> T:
        """Initial time of the phase."""
        return cast(T, self._initial_time[0])

    @property
    def final_time(self) -> T:
        """Final time of the phase."""
        return cast(T, self._final_time[0])

    @property
    def initial_state(self) -> NDArray[T]:
        """Initial state of the phase, as a read-only copy."""
        return read_only(self._initial_state.copy())

    @property
    def final_state(self) -> NDArray[T]:
        """Final state of the phase, as a read-only copy."""
        return read_only(self._final_state.copy())

    @property
    def integral(self) -> NDArray[T]:
        """Array of integral values for the phase, as a read-only copy."""
        return read_only(self._integral.copy())


class ObjectiveArg(DiscreteArgBase[T], Protected, Generic[T]):
    """Argument for objective callback function.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class containing problem data.
    dv : DVStructure
        A structure for decision variables.
    """

    auxdata: Auxdata
    """SimpleNamespace container for user-defined data."""

    _callback = "objective"
    _results_in = "arg.objective = ..."

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        # Initialize the DiscreteArgBase with problem and dv
        DiscreteArgBase.__init__(self, problem, dv, dtype)
        # the objective starts at zero and unassigned
        if dtype == np.object_:
            self._objective = cast(T, SXW(0.0))
        elif dtype == np.float64:
            self._objective = cast(T, 0.0)
        else:
            msg = f"Unsupported type for objective: {dtype}"
            raise TypeError(msg)
        self._objective_written = False
        self._objective_zero = self._objective

    def _reset(self) -> None:
        """Return the objective to zero and unassigned."""
        set_private(self, "_objective", self._objective_zero)
        set_private(self, "_objective_written", value=False)

    @property
    def objective(self) -> T:
        """The objective value, set by the callback."""
        return self._objective

    @objective.setter
    def objective(self, value: T) -> None:
        # an array here fails much later, inside the derivative method, with a message that
        # names neither the objective nor the line that set it
        if np.ndim(value) != 0:
            msg = f"arg.objective must be a scalar, got a value of shape {np.shape(value)}."
            raise TypeError(msg)
        set_private(self, "_objective", value)
        set_private(self, "_objective_written", value=True)


class ObjectiveGradientArg(DiscreteArgBase[np.float64], Protected):
    """Argument for objective gradient callback function.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class containing problem data.
    dv : DVStructure
        A structure for decision variables.
    """

    auxdata: Auxdata
    """SimpleNamespace container for user-defined data."""

    gradient: dict[DVKey, float | np.floating[Any]]
    """Dictionary mapping decision variable keys to gradient values of type T."""

    _callback = "objective_gradient"
    _results_in = "arg.gradient[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return (("", self.gradient),)

    def _reset(self) -> None:
        """Empty the gradient entries."""
        self.gradient.clear()

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the base class with the provided problem and dv
        DiscreteArgBase.__init__(self, problem, dv, np.float64)
        # Initialize the gradient dictionary with the specific type T for values
        self.gradient: dict[DVKey, float | np.floating[Any]] = {}


class ObjectiveHessianArg(DiscreteArgBase[np.float64], Protected):
    """Argument for objective Hessian callback function.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class containing problem data.
    dv : DVStructure
        A structure for decision variables.
    """

    auxdata: Auxdata
    """SimpleNamespace container for user-defined data."""

    hessian: dict[OHSTerm, float]
    """Dictionary mapping terms to Hessian values of type T."""

    _callback = "objective_hessian"
    _results_in = "arg.hessian[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return (("", self.hessian),)

    def _reset(self) -> None:
        """Empty the Hessian entries."""
        self.hessian.clear()

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the base class with the provided problem and dv
        DiscreteArgBase.__init__(self, problem, dv, np.float64)
        # Initialize hessian as an empty dictionary with values of type T
        self.hessian: dict[OHSTerm, float] = {}


class DiscreteArg(DiscreteArgBase[T], Protected, Generic[T]):
    """Discrete argument to be passed to user-defined discrete constraint function.

    Parameters
    ----------
    problem : Problem
        The optimal control problem object.
    dv : DVStructure
        Decision variable structure.
    dtype : Type
        The data type for the discrete array elements, such as float or object.

    Attributes
    ----------
    discrete : NDArray[T]
        Array holding the discrete values to be passed to the constraint function.
    """

    _callback = "discrete"
    _results_in = "arg.discrete[i] = ..."

    def _reset(self) -> None:
        """Return every discrete constraint value to zero and unassigned."""
        self._discrete.reset()

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        super().__init__(problem, dv, dtype)
        # the discrete constraint values, assigned by whole rows (one value each)
        self._discrete: OutputArray[T] = OutputArray.zeros(
            (problem.nd,),
            dtype,
            label="arg.discrete",
            count=f"nd = {problem.nd}",
        )
        self._dv = dv

    # Use the Discrete descriptor with generic typing for consistency

    @property
    def discrete(self) -> OutputArray[T]:
        """Return the discrete constraint values."""
        return self._discrete

    @discrete.setter
    def discrete(self, value: OutputArray[T] | Sequence[Any] | NDArray[Any]) -> None:
        """Assign every discrete constraint value."""
        self._discrete[:] = value


class DiscreteJacobianArg(DiscreteArgBase[np.float64], Protected):
    """Discrete argument to be passed to user-defined discrete constraint function.

    Parameters
    ----------
    problem : Problem
        The optimal control problem object.
    dv : DVStructure
        Decision variable structure.
    dtype : Type
        The data type for the discrete array elements, such as float or object.

    Attributes
    ----------
    jacobian : dict[tuple[DFIndex, DVKey], T]
        Dictionary representing the Jacobian structure with values of type T.
    """

    _callback = "discrete_jacobian"
    _results_in = "arg.jacobian[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return (("", self.jacobian),)

    def _reset(self) -> None:
        """Empty the Jacobian entries."""
        self.jacobian.clear()

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the superclass with problem and dv
        super().__init__(problem, dv, np.float64)
        # Initialize jacobian as an empty dictionary with values of type T
        self.jacobian: dict[tuple[DFIndex, DVKey], float] = {}
        self._dv = dv


class DiscreteHessianArg(DiscreteArgBase[np.float64], Protected):
    """Discrete argument to be passed to user-defined discrete constraint function.

    Parameters
    ----------
    problem : Problem
        The optimal control problem object.
    dv : DVStructure
        Decision variable structure.
    dtype : Type
        The data type for the discrete array elements, such as float or object.

    Attributes
    ----------
    hessian : dict[tuple[DFIndex, DVKey, DVKey], T]
        Dictionary representing the Hessian structure with values of type T.
    """

    _callback = "discrete_hessian"
    _results_in = "arg.hessian[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return (("", self.hessian),)

    def _reset(self) -> None:
        """Empty the Hessian entries."""
        self.hessian.clear()

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the superclass with problem and dv
        super().__init__(problem, dv, np.float64)
        # Initialize hessian as an empty dictionary with values of type T
        self.hessian: dict[tuple[DFIndex, DVKey, DVKey], float] = {}
        self._dv = dv


class ContinuousStore(BaseArg[T], Generic[T]):
    """All continuous data of one problem: the inputs, the outputs, and the derivatives.

    Parameters
    ----------
    problem : Problem
        The optimal control problem object.
    dv : DVStructure
        Decision variable structure.
    dtype : Type
        The data type for the continuous array elements, such as float or object.
    tau_u : Sequence[NDArray], optional
        Mesh time of each evaluation point, per phase; required for a numeric argument.
    nodes : Sequence[NDArray[np.intp]], optional
        For a numeric argument, the evaluation points to present, per phase, in the order
        given: time, states, controls, and outputs all have ``len(nodes[p])`` points. The
        inputs are copies, refreshed by `_sync`. Default: every evaluation point, with the
        states and controls as views of ``dv``.
    """

    def __init__(
        self,
        problem: yapss.Problem,
        dv: DVStructure[T],
        dtype: type[T],
        *,
        tau_u: Sequence[NDArray[np.float64]] | None = None,
        nodes: Sequence[NDArray[np.intp]] | None = None,
    ) -> None:
        super().__init__(problem, dv, dtype)
        if dtype == np.float64 and tau_u is None:
            msg = "Numeric ContinuousStore instances require tau_u."
            raise ValueError(msg)
        if nodes is not None and (dtype != np.float64 or len(nodes) != problem.np):
            msg = "nodes are for numeric ContinuousStore instances only, one array per phase."
            raise ValueError(msg)
        self._nodes = nodes
        self._tau_u = tau_u
        # Initialize _phase with a tuple of ContinuousPhase instances
        self._phase: tuple[ContinuousPhaseData[T], ...] = tuple(
            ContinuousPhaseData(problem, dv, q, dtype, None if nodes is None else nodes[q])
            for q in range(problem.np)
        )
        # Initialize phase list based on problem.np
        self._phase_list: tuple[int, ...] = tuple(range(problem.np))
        # `_reset` runs before every call of the callback, so it works on the buffers
        # directly: an output with no rows has nothing to clear and is left out
        self._output_buffers: tuple[tuple[NDArray[T], NDArray[np.bool_]], ...] = tuple(
            (output._storage, output._written)
            for phase in self._phase
            for output in phase._outputs.values()
            if output.shape[0]
        )
        self._value_arg: ContinuousArg[T] | None = None
        self._jacobian_arg: ContinuousJacobianArg | None = None
        self._hessian_arg: ContinuousHessianArg | None = None

    def _sync(self, z: NDArray[np.float64]) -> None:
        """Synchronize numeric continuous inputs with an NLP decision vector."""
        if self._dtype != np.float64 or self._tau_u is None:
            msg = "ContinuousStore._sync() is available for numeric arguments only."
            raise TypeError(msg)

        self._dv.z[:] = z
        for p, tau in enumerate(self._tau_u):
            t0 = self._dv.phase[p].t0[0]
            tf = self._dv.phase[p].tf[0]
            phase = self.phase[p]
            if self._nodes is None:
                phase._time[:] = tau * (tf - t0) / 2 + (t0 + tf) / 2
                continue
            # a node subset: the inputs are copies of the selected points
            nodes = self._nodes[p]
            phase._time[:] = tau[nodes] * (tf - t0) / 2 + (t0 + tf) / 2
            for state, values in zip(phase._state, self._dv.phase[p].xc, strict=True):
                state[:] = values[nodes]
            for control, values in zip(phase._control, self._dv.phase[p].u, strict=True):
                control[:] = values[nodes]

    def __getitem__(
        self,
        item: tuple[PhaseIndex, CVName, CVIndex] | tuple[PhaseIndex, CFName, CFIndex],
    ) -> NDArray[T]:
        """Get items from the store using a CVKey or CFKey object.

        The __getitem__ method used here is not guaranteed to be stable and should not
        be used in callback functions. This method is used internally by YAPSS to
        reference user-supplied values. It reaches the writable arrays behind the
        read-only inputs a callback sees, because the central-difference stencils
        perturb variables through it.

        Parameters
        ----------
        item : Union[tuple[PhaseIndex, CVName, CVIndex], tuple[PhaseIndex, CFName, CFIndex]]
            Key for extracting data from the continuous argument.
        """
        p, letter, i = item
        phase = self.phase[p]
        # outputs are read as plain arrays: this accessor is on the central-difference hot path
        match letter:
            case "f":
                value = phase.dynamics.view(numpy.ndarray)[i]
            case "g":
                value = phase.integrand.view(numpy.ndarray)[i]
            case "h":
                value = phase.path.view(numpy.ndarray)[i]
            case "x":
                value = phase._state[i]
            case "u":
                value = phase._control[i]
            case "t":
                value = phase._time
            case "s":
                value = self._dv.s[i : i + 1]
            case _:
                assert_never(letter)

        return cast(NDArray[T], value)

    @property
    def phase_list(self) -> tuple[int, ...]:
        """Return phase list."""
        return self._phase_list

    @property
    def phase(self) -> tuple[ContinuousPhaseData[T], ...]:
        """Return the tuple of per-phase data objects."""
        return self._phase

    # ---------------------------------------------------- the three callback arguments
    # Built once, on demand: a symbolic store never needs the derivative arguments.

    @property
    def value_arg(self) -> ContinuousArg[T]:
        """The argument passed to the continuous callback."""
        if self._value_arg is None:
            phases = tuple(ContinuousPhase(data) for data in self._phase)
            self._value_arg = ContinuousArg(self, phases)
        return self._value_arg

    @property
    def jacobian_arg(self) -> ContinuousJacobianArg:
        """The argument passed to the continuous Jacobian callback."""
        if self._jacobian_arg is None:
            store = cast("ContinuousStore[np.float64]", self)
            phases = tuple(ContinuousJacobianPhase(data) for data in store._phase)
            self._jacobian_arg = ContinuousJacobianArg(store, phases)
        return self._jacobian_arg

    @property
    def hessian_arg(self) -> ContinuousHessianArg:
        """The argument passed to the continuous Hessian callback."""
        if self._hessian_arg is None:
            store = cast("ContinuousStore[np.float64]", self)
            phases = tuple(ContinuousHessianPhase(data) for data in store._phase)
            self._hessian_arg = ContinuousHessianArg(store, phases)
        return self._hessian_arg


class ContinuousPhaseData(Generic[T]):
    """One phase of a `ContinuousStore`: its inputs, outputs and derivative entries."""

    dynamics: Output[T] = Output()
    integrand: Output[T] = Output()
    path: Output[T] = Output()

    def __init__(
        self,
        problem: yapss.Problem,
        dv: SimpleNamespace,
        q: int,
        dtype: type[T],
        nodes: NDArray[np.intp] | None = None,
    ) -> None:
        self._outputs: dict[str, OutputArray[T]] = {}
        self._p: int = q
        self._nx: int = problem.nx[q]
        self._nq: int = problem.nq[q]
        self._nh: int = problem.nh[q]

        nx = self._nx
        nq = self._nq
        nh = self._nh

        # one column per evaluation point (or per selected node); a symbolic argument is
        # traced at a single node
        if dtype == np.object_:
            nt = 1
        elif nodes is not None:
            nt = len(nodes)
        else:
            nt = problem_layout(problem)[q].n_eval

        # symbolic time is a single free symbol: the continuous functions are traced
        # once at a generic node, and the tau -> t chain rule is applied by the NLP
        # assembly, which keeps the CasADi graph independent of the mesh size
        # inputs are read-only to the callback. Each is a read-only view of a writable
        # array kept beside it: `_time`, `_state`, and `_control` are what `_sync` and the
        # central-difference NaN probes write, and the views a callback holds follow them.
        self.time: NDArray[T]
        if dtype == np.object_:
            self._time: NDArray[T] = sx_array([SXW(SX.sym("t"))])
        else:
            self._time = numpy.zeros([nt], dtype=dtype)
        self.time = read_only(self._time)
        self._state: list[NDArray[Any]] = [
            dv.phase[q].xc[i] if nodes is None else numpy.zeros(nt) for i in range(problem.nx[q])
        ]
        self._control: list[NDArray[Any]] = [
            dv.phase[q].u[i] if nodes is None else numpy.zeros(nt) for i in range(problem.nu[q])
        ]
        self.state: NDArray[Any] = numpy.zeros([problem.nx[q]], dtype=object)
        for i, values in enumerate(self._state):
            self.state[i] = read_only(values)
        self.control: NDArray[Any] = numpy.zeros([problem.nu[q]], dtype=object)
        for i, values in enumerate(self._control):
            self.control[i] = read_only(values)
        # the containers too: `arg.phase[p].state[i] = ...` replaces an input
        self.state.flags.writeable = False
        self.control.flags.writeable = False

        # outputs, assigned by whole rows
        outputs = (("dynamics", "nx", nx), ("integrand", "nq", nq), ("path", "nh", nh))
        for name, count, rows in outputs:
            self._outputs[name] = OutputArray.zeros(
                (rows, nt),
                dtype,
                label=f"arg.phase[{q}].{name}",
                count=f"{count} = {rows} in phase {q}",
            )
        self._hessian: dict[tuple[CFKey, CVKey, CVKey], Any] = {}
        self._jacobian: dict[tuple[CFKey, CVKey], Any] = {}

    @property
    def jacobian(self) -> dict[tuple[CFKey, CVKey], Any]:
        """Return jacobian."""
        return self._jacobian

    @property
    def hessian(self) -> dict[tuple[CFKey, CVKey, CVKey], Any]:
        """Return hessian."""
        return self._hessian


# --------------------------------------------------------------------------------------
# The three continuous arguments.
#
# Ipopt asks for the constraint values, the Jacobian, and the Hessian at each iterate, and
# the Hessian's chain-rule terms need the values and the Jacobian too, so all three are
# evaluated once per point from one `ContinuousStore`. Each callback is handed its own
# argument over that store, exposing the inputs and that callback's own output and nothing
# else, so one continuous callback cannot read or write another's results (E7c). Before,
# one object went to all three and `nlp.py` merely *cast* it, so a dynamics row written
# from `continuous_jacobian` overwrote the cached constraint values and spoiled the solve
# with no error.
#
# The three are siblings, not a hierarchy. `_ContinuousPhaseInputs` is a base only for what
# is genuinely common -- reading the time, states and controls of the point the call was
# made at -- which every one of them honors, so it is substitutable in the Liskov sense.
# An argument that must refuse `dynamics` cannot stand in for one that accepts it, so
# neither derivative argument is a kind of `ContinuousArg`.


class _ContinuousPhaseInputs(Generic[T]):
    """One phase's inputs, as every continuous callback sees them."""

    def __init__(self, data: ContinuousPhaseData[T]) -> None:
        # direct references, not properties: the input views are made once and written
        # through the storage behind them, so a callback's reads cost what they always did
        self.time: NDArray[T] = data.time
        self.state: NDArray[Any] = data.state
        self.control: NDArray[Any] = data.control


class ContinuousPhase(_ContinuousPhaseInputs[T], Protected, Generic[T]):
    """One phase as the continuous callback sees it: its inputs and its three outputs."""

    dynamics: Output[T] = Output()
    integrand: Output[T] = Output()
    path: Output[T] = Output()

    def __init__(self, data: ContinuousPhaseData[T]) -> None:
        super().__init__(data)
        self._outputs: dict[str, OutputArray[T]] = data._outputs


class ContinuousJacobianPhase(_ContinuousPhaseInputs[np.float64], Protected):
    """One phase as the continuous Jacobian callback sees it: its inputs and `jacobian`."""

    def __init__(self, data: ContinuousPhaseData[np.float64]) -> None:
        super().__init__(data)
        self._jacobian = data._jacobian

    @property
    def jacobian(self) -> dict[tuple[CFKey, CVKey], Any]:
        """The Jacobian entries of this phase, keyed by (function, variable)."""
        return self._jacobian


class ContinuousHessianPhase(_ContinuousPhaseInputs[np.float64], Protected):
    """One phase as the continuous Hessian callback sees it: its inputs and `hessian`."""

    def __init__(self, data: ContinuousPhaseData[np.float64]) -> None:
        super().__init__(data)
        self._hessian = data._hessian

    @property
    def hessian(self) -> dict[tuple[CFKey, CVKey, CVKey], Any]:
        """The Hessian entries of this phase, keyed by (function, variable, variable)."""
        return self._hessian


class _ContinuousArgBase(BaseArg[T], Generic[T]):
    """What the three continuous arguments share: the inputs and the phase list."""

    def __init__(self, store: ContinuousStore[T], phase: tuple[Any, ...]) -> None:
        self.auxdata = store.auxdata
        self._store = store
        self._dv = store._dv
        self._parameter = store._parameter
        self._dtype = store._dtype
        self._phase = phase

    @property
    def phase(self) -> tuple[Any, ...]:
        """The phases, as this callback sees them."""
        return self._phase

    @property
    def phase_list(self) -> tuple[int, ...]:
        """The phases this call is for."""
        return self._store._phase_list


class ContinuousArg(_ContinuousArgBase[T], Protected, Generic[T]):
    """Argument of the continuous callback: the inputs and the continuous outputs."""

    _callback = "continuous"
    _results_in = "arg.phase[p].dynamics[i] = ..."

    def _reset(self) -> None:
        """Return every output row of every phase to zero and unassigned."""
        for storage, written in self._store._output_buffers:
            storage.fill(0)
            written[:] = False


class ContinuousJacobianArg(_ContinuousArgBase[np.float64], Protected):
    """Argument of the continuous Jacobian callback: the inputs and `phase[p].jacobian`."""

    _callback = "continuous_jacobian"
    _results_in = "arg.phase[p].jacobian[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return tuple(
            (f" for phase {p}", phase._jacobian) for p, phase in enumerate(self._store.phase)
        )

    def _reset(self) -> None:
        """Empty every phase's Jacobian entries."""
        for phase in self._store.phase:
            phase._jacobian.clear()


class ContinuousHessianArg(_ContinuousArgBase[np.float64], Protected):
    """Argument of the continuous Hessian callback: the inputs and `phase[p].hessian`."""

    _callback = "continuous_hessian"
    _results_in = "arg.phase[p].hessian[key] = ..."

    def _key_groups(self) -> tuple[tuple[str, dict[Any, Any]], ...]:
        return tuple(
            (f" for phase {p}", phase._hessian) for p, phase in enumerate(self._store.phase)
        )

    def _reset(self) -> None:
        """Empty every phase's Hessian entries."""
        for phase in self._store.phase:
            phase._hessian.clear()


# Define function type aliases with generics
ObjectiveFunctionFloat = Callable[["ObjectiveArg[np.float64]"], None]
ObjectiveFunctionObject = Callable[["ObjectiveArg[np.object_]"], None]
ObjectiveFunction = ObjectiveFunctionFloat | ObjectiveFunctionObject

ObjectiveGradientFunction = Callable[["ObjectiveGradientArg"], None]
ObjectiveHessianFunction = Callable[["ObjectiveHessianArg"], None]

DiscreteFunctionFloat = Callable[["DiscreteArg[np.float64]"], None]
DiscreteFunctionObject = Callable[["DiscreteArg[np.object_]"], None]
DiscreteFunction = DiscreteFunctionFloat | DiscreteFunctionObject

DiscreteJacobianFunction = Callable[["DiscreteJacobianArg"], None]
DiscreteHessianFunction = Callable[["DiscreteHessianArg"], None]

ContinuousFunctionFloat = Callable[["ContinuousArg[np.float64]"], None]
ContinuousFunctionObject = Callable[["ContinuousArg[np.object_]"], None]
ContinuousFunction = ContinuousFunctionFloat | ContinuousFunctionObject

ContinuousJacobianFunction = Callable[["ContinuousJacobianArg"], None]
ContinuousHessianFunction = Callable[["ContinuousHessianArg"], None]


# Update ProblemFunctions with detailed type annotations and generics
class ProblemFunctions(SimpleNamespace):
    """Container for problem functions.

    Attributes
    ----------
    objective: Optional[ObjectiveFunction]
    objective_gradient: Optional[ObjectiveGradientFunction]
    objective_hessian: Optional[ObjectiveHessianFunction]
    continuous: Optional[ContinuousFunction]
    continuous_jacobian: Optional[ContinuousJacobianFunction]
    continuous_hessian: Optional[ContinuousHessianFunction]
    discrete: Optional[DiscreteFunction]
    discrete_jacobian: Optional[DiscreteJacobianFunction]
    discrete_hessian: Optional[DiscreteHessianFunction]
    continuous_jacobian_structure: CJS
    objective_gradient_structure: OGS
    discrete_jacobian_structure: DJS
    continuous_jacobian_structure_cd: CJFDS
    discrete_jacobian_structure_cd: DJFDS
    discrete_hessian_structure: Optional[DHS]
    discrete_hessian_structure_cd: Optional[DHFDS]
    objective_hessian_structure: Optional[OHS]
    continuous_hessian_structure: Optional[CHS]
    continuous_hessian_structure_cd: Optional[CHFDS]
    """

    objective: ObjectiveFunction
    objective_gradient: ObjectiveGradientFunction
    objective_hessian: ObjectiveHessianFunction
    continuous: ContinuousFunction
    continuous_jacobian: ContinuousJacobianFunction
    continuous_hessian: ContinuousHessianFunction
    discrete: DiscreteFunction
    discrete_jacobian: DiscreteJacobianFunction
    discrete_hessian: DiscreteHessianFunction

    continuous_jacobian_structure: CJS
    objective_gradient_structure: OGS
    discrete_jacobian_structure: DJS
    continuous_jacobian_structure_cd: CJFDS
    discrete_jacobian_structure_cd: DJFDS
    discrete_hessian_structure: DHS
    discrete_hessian_structure_cd: DHFDS
    objective_hessian_structure: OHS
    continuous_hessian_structure: CHS
    continuous_hessian_structure_cd: CHFDS
