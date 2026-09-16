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


def call_callback(function: Any, arg: Any, *, reset: bool = True) -> None:
    """Call one user callback, under the rules every callback obeys.

    The callback's outputs are cleared first, so a row the callback does not assign on this
    call is zero rather than whatever the previous call left there, and a value returned
    instead of assigned is refused, naming the idiom to use.

    Every call of a user-supplied function goes through here. Derivative arguments keep their
    ``jacobian``/``hessian`` dictionaries across calls; whether a key may first appear on a
    later call is an open question (E7b), so `BaseArg._reset` leaves them alone.

    ``reset=False`` is for the NLP's shared continuous evaluator alone: it passes one
    `ContinuousArg` to all three continuous callbacks, so the outputs a Jacobian or Hessian
    call finds there belong to the continuous callback and must survive. Distinct runtime
    classes (E7c) would carry that in `ContinuousJacobianArg._reset` and retire the argument.
    """
    if reset:
        arg._reset()
    result = function(arg)
    if result is not None:
        msg = (
            f"the {arg._callback} callback returned a {type(result).__name__}; a callback "
            f"assigns its results to the argument and returns nothing: {arg._results_in}"
        )
        raise TypeError(msg)


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

    def _reset(self) -> None:
        """Clear whatever the callback assigns. Derivative dictionaries are kept (E7b)."""

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        self.auxdata = problem.auxdata
        self._dv: DVStructure[T] = dv
        self._parameter: NDArray[T] = dv.s
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
        self._parameter: NDArray[T] = dv.s
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
        """Initial state of the phase as an immutable copy."""
        return self._initial_state.copy()

    @property
    def final_state(self) -> NDArray[T]:
        """Final state of the phase as an immutable copy."""
        return self._final_state.copy()

    @property
    def integral(self) -> NDArray[T]:
        """Array of integral values for the phase as an immutable copy."""
        return self._integral.copy()


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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the superclass with problem and dv
        super().__init__(problem, dv, np.float64)
        # Initialize hessian as an empty dictionary with values of type T
        self.hessian: dict[tuple[DFIndex, DVKey, DVKey], float] = {}
        self._dv = dv


class ContinuousArg(BaseArg[T], Protected, Generic[T]):
    """Continuous argument to be passed to user-defined continuous constraint function.

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

    _callback = "continuous"
    _results_in = "arg.phase[p].dynamics[i] = ..."

    def _reset(self) -> None:
        """Return every output row of every phase to zero and unassigned."""
        for storage, written in self._output_buffers:
            storage.fill(0)
            written[:] = False

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
            msg = "Numeric ContinuousArg instances require tau_u."
            raise ValueError(msg)
        if nodes is not None and (dtype != np.float64 or len(nodes) != problem.np):
            msg = "nodes are given for numeric ContinuousArg instances only, one array per phase."
            raise ValueError(msg)
        self._nodes = nodes
        self._tau_u = tau_u
        # Initialize _phase with a tuple of ContinuousPhase instances
        self._phase: tuple[ContinuousPhase[T], ...] = tuple(
            ContinuousPhase(problem, dv, q, dtype, None if nodes is None else nodes[q])
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

    def _sync(self, z: NDArray[np.float64]) -> None:
        """Synchronize numeric continuous inputs with an NLP decision vector."""
        if self._dtype != np.float64 or self._tau_u is None:
            msg = "ContinuousArg._sync() is available for numeric arguments only."
            raise TypeError(msg)

        self._dv.z[:] = z
        for p, tau in enumerate(self._tau_u):
            t0 = self._dv.phase[p].t0[0]
            tf = self._dv.phase[p].tf[0]
            phase = self.phase[p]
            if self._nodes is None:
                phase.time[:] = tau * (tf - t0) / 2 + (t0 + tf) / 2
                continue
            # a node subset: the inputs are copies of the selected points
            nodes = self._nodes[p]
            phase.time[:] = tau[nodes] * (tf - t0) / 2 + (t0 + tf) / 2
            for state, values in zip(phase.state, self._dv.phase[p].xc, strict=True):
                state[:] = values[nodes]
            for control, values in zip(phase.control, self._dv.phase[p].u, strict=True):
                control[:] = values[nodes]

    def __getitem__(
        self,
        item: tuple[PhaseIndex, CVName, CVIndex] | tuple[PhaseIndex, CFName, CFIndex],
    ) -> NDArray[T]:
        """Get items from ContinuousArg structure using CVKey or CFKey object.

        The __getitem__ method used here is not guaranteed to be stable and should not
        be used in callback functions. This method is used internally by YAPSS to
        reference user-supplied values.

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
                value = phase.state[i]
            case "u":
                value = phase.control[i]
            case "t":
                value = phase.time
            case "s":
                value = self.parameter[i : i + 1]
            case _:
                assert_never(letter)

        return cast(NDArray[T], value)

    @property
    def phase_list(self) -> tuple[int, ...]:
        """Return phase list."""
        return self._phase_list

    @property
    def phase(self) -> tuple[ContinuousPhase[T], ...]:
        """Return the tuple of ContinuousPhase objects."""
        return self._phase


class ContinuousPhase(Protected, Generic[T]):
    """Continuous phase."""

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
        self.time: NDArray[T]
        if dtype == np.object_:
            self.time = sx_array([SXW(SX.sym("t"))])
        else:
            self.time = numpy.zeros([nt], dtype=dtype)
        self.state: NDArray[Any] = numpy.zeros([problem.nx[q]], dtype=object)
        for i in range(problem.nx[q]):
            self.state[i] = dv.phase[q].xc[i] if nodes is None else numpy.zeros(nt)
        self.control: NDArray[Any] = numpy.zeros([problem.nu[q]], dtype=object)
        for i in range(problem.nu[q]):
            self.control[i] = dv.phase[q].u[i] if nodes is None else numpy.zeros(nt)

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


# Define generically typed ContinuousJacobianArg and ContinuousHessianArg
class ContinuousJacobianArg(ContinuousArg[np.float64]):
    """Continuous argument for user-defined continuous constraint Jacobian function."""

    _callback = "continuous_jacobian"
    _results_in = "arg.phase[p].jacobian[key] = ..."

    def _reset(self) -> None:
        """Keep the outputs: they belong to the continuous callback.

        The NLP's shared evaluator passes one argument to all three continuous callbacks, and
        the function values it holds must survive a Jacobian or Hessian call.
        """


class ContinuousHessianArg(ContinuousArg[np.float64]):
    """Continuous argument for user-defined continuous constraint Hessian function."""

    _callback = "continuous_hessian"
    _results_in = "arg.phase[p].hessian[key] = ..."

    def _reset(self) -> None:
        """Keep the outputs: they belong to the continuous callback."""


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
