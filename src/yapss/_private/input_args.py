"""

Module input_args.

This module defines arguments which are used to call the user-defined callback functions.
"""

# future imports
from __future__ import annotations

# standard inputs
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Union, cast

# third party imports
import numpy  # noqa: ICN001
from casadi import SX

from yapss.math.wrapper import SXW

# package imports
from .types_ import Protected

if TYPE_CHECKING:
    from collections.abc import Sequence, MutableSequence

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


class BaseArg(Generic[T]):
    """Base class for all argument classes.

    Parameters
    ----------
    problem : yapss.problem.Problem
        An instance of the Problem class containing problem data.
    dv : DVStructure
        A structure for decision variables.
    """

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


class DiscretePhase(Generic[T]):
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
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()

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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        # Initialize the DiscreteArgBase with problem and dv
        DiscreteArgBase.__init__(self, problem, dv, dtype)
        # Define the objective, typed as T
        if dtype == np.object_:
            self.objective = cast(T, SXW(0.0))  # Explicitly cast SXW to T
        elif dtype == np.float64:
            self.objective = cast(T, 0.0)  # Explicitly cast float to T
        else:
            msg = f"Unsupported type for objective: {dtype}"
            raise TypeError(msg)
        # Immutable attributes enforcement
        self._allowed_del_attrs = ()
        self._allowed_attrs = ("objective",)


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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the base class with the provided problem and dv
        DiscreteArgBase.__init__(self, problem, dv, np.float64)
        # Initialize the gradient dictionary with the specific type T for values
        self.gradient: dict[DVKey, float | np.floating[Any]] = {}
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()


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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the base class with the provided problem and dv
        DiscreteArgBase.__init__(self, problem, dv, np.float64)
        # Initialize hessian as an empty dictionary with values of type T
        self.hessian: dict[OHSTerm, float] = {}
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()


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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        super().__init__(problem, dv, dtype)
        # Initialize the discrete array with the specified dtype
        self._discrete: NDArray[T] = np.zeros([problem.nd], dtype=dtype)
        self._dv = dv
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ("discrete",)

    # Use the Discrete descriptor with generic typing for consistency

    @property
    def discrete(self) -> NDArray[T] | MutableSequence[Any]:
        """Return the discrete array."""
        return self._discrete

    @discrete.setter
    def discrete(self, value: Sequence[Any]) -> None:
        """Set the discrete array."""
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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the superclass with problem and dv
        super().__init__(problem, dv, np.float64)
        # Initialize jacobian as an empty dictionary with values of type T
        self.jacobian: dict[tuple[DFIndex, DVKey], float] = {}
        self._dv = dv
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()


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

    def __init__(self, problem: yapss.Problem, dv: DVStructure[np.float64]) -> None:
        # Initialize the superclass with problem and dv
        super().__init__(problem, dv, np.float64)
        # Initialize hessian as an empty dictionary with values of type T
        self.hessian: dict[tuple[DFIndex, DVKey, DVKey], float] = {}
        self._dv = dv
        # Attributes to enforce immutability restrictions
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()


class ContinuousArrayDescriptor(Generic[T], Protected):
    """Descriptor for continuous array, supporting flexible data types."""

    name: str

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Set the name of the attribute."""
        self.name = name

    def __get__(self, instance: ContinuousPhase[T], owner: type[Any]) -> NDArray[T]:
        """Get the value of the continuous array as an immutable copy."""
        # Retrieve the value from the dictionary in instance
        return instance._descriptor_values[self.name]

    # def __set__(self, instance: ContinuousPhase[T], value: NDArray[T]) -> None:
    def __set__(self, instance: ContinuousPhase[T], value: Sequence[Any]) -> None:
        """Set the value of the continuous array."""
        # Store or update the value in the dictionary to maintain structure
        instance._descriptor_values[self.name][:] = value


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
    """

    def __init__(self, problem: yapss.Problem, dv: DVStructure[T], dtype: type[T]) -> None:
        super().__init__(problem, dv, dtype)
        # Initialize _phase with a tuple of ContinuousPhase instances
        self._phase: tuple[ContinuousPhase[T], ...] = tuple(
            ContinuousPhase(problem, dv, q, dtype) for q in range(problem.np)
        )
        # Initialize phase list based on problem.np
        self._phase_list: tuple[int, ...] = tuple(range(problem.np))
        self._allowed_del_attrs = ()
        # Update allowed attributes to reflect the correct structure
        self._allowed_attrs = ("_phase_list", "_phase")

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
        if letter == "f":
            value = phase.dynamics[i]
        elif letter == "g":
            value = phase.integrand[i]
        elif letter == "h":
            value = phase.path[i]
        elif letter == "x":
            value = phase.state[i]
        elif letter == "u":
            value = phase.control[i]
        elif letter == "t":
            value = phase.time
        elif letter == "s":
            value = self.parameter[i : i + 1]
        else:
            msg = "Invalid item key '{letter}' in ContinuousArg.__getitem__."
            raise RuntimeError(msg)

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

    dynamics: ContinuousArrayDescriptor[T] = ContinuousArrayDescriptor()
    integrand: ContinuousArrayDescriptor[T] = ContinuousArrayDescriptor()
    path: ContinuousArrayDescriptor[T] = ContinuousArrayDescriptor()

    def __init__(
        self,
        problem: yapss.Problem,
        dv: SimpleNamespace,
        q: int,
        dtype: type[T],
    ) -> None:
        self._descriptor_values: dict[str, NDArray[T]] = {}
        self._p: int = q
        self._nx: int = problem.nx[q]
        self._nq: int = problem.nq[q]
        self._nh: int = problem.nh[q]

        nx = self._nx
        nq = self._nq
        nh = self._nh

        # Determine the number of collocation points (nt) based on the spectral method
        collocation_points = problem.mesh.phase[q].collocation_points
        if dtype == np.object_:
            nt = 1
        elif problem.spectral_method == "lgr":
            nt = sum(collocation_points)
        elif problem.spectral_method == "lgl":
            nt = sum(collocation_points) - len(collocation_points) + 1
        elif problem.spectral_method == "lg":
            nt = sum(collocation_points)
        else:
            raise RuntimeError

        self.time: NDArray[T]
        # TODO: This can't be right -- should get the time from the phase
        if dtype == np.object_:
            self.time = SX.sym("t")
        else:
            self.time = numpy.zeros([nt], dtype=dtype)
        self.state: NDArray[Any] = numpy.zeros([problem.nx[q]], dtype=object)
        for i in range(problem.nx[q]):
            self.state[i] = dv.phase[q].xc[i]
        self.control: NDArray[Any] = numpy.zeros([problem.nu[q]], dtype=object)
        for i in range(problem.nu[q]):
            self.control[i] = dv.phase[q].u[i]

        # initialize input and output arrays
        self._descriptor_values["dynamics"] = ContinuousArray([nx, nt], dtype=dtype)
        self._descriptor_values["integrand"] = ContinuousArray([nq, nt], dtype=dtype)
        self._descriptor_values["path"] = ContinuousArray([nh, nt], dtype=dtype)
        self._hessian: dict[tuple[CFKey, CVKey, CVKey], Any] = {}
        self._jacobian: dict[tuple[CFKey, CVKey], Any] = {}

        self._allowed_del_attrs = ()
        # TODO: "time" should not be in list below but it's a kludge for now
        self._allowed_attrs = ("dynamics", "integrand", "path", "time")

    @property
    def jacobian(self) -> dict[tuple[CFKey, CVKey], Any]:
        """Return jacobian."""
        return self._jacobian

    @property
    def hessian(self) -> dict[tuple[CFKey, CVKey, CVKey], Any]:
        """Return hessian."""
        return self._hessian


class ContinuousArray(NDArray[T], Generic[T]):
    """Custom continuous array that initializes with zeros and handles assignment."""

    def __new__(cls, shape: list[int], dtype: type[T], **kwargs: Any) -> ContinuousArray[T]:
        # Create an instance of ContinuousArray with the specified dtype
        obj = super().__new__(cls, shape, dtype=dtype, **kwargs)
        obj.fill(0)  # Initialize with zeros or appropriate type
        return obj

    def __setitem__(self, item: slice | int, value: Any) -> None:
        """Set item in array, expanding scalar values to match the shape if needed."""
        # If the value is an iterable, expand each element to match the shape
        if isinstance(value, (list, tuple)):
            expanded_value = [np.full(self.shape[1:], v, dtype=self.dtype) for v in value]
        else:
            expanded_value = value  # Assign directly for scalar values
        super().__setitem__(item, expanded_value)  # type: ignore[no-untyped-call, unused-ignore]


# Define generically typed ContinuousJacobianArg and ContinuousHessianArg
class ContinuousJacobianArg(ContinuousArg[np.float64]):
    """Continuous argument for user-defined continuous constraint Jacobian function."""


class ContinuousHessianArg(ContinuousArg[np.float64]):
    """Continuous argument for user-defined continuous constraint Hessian function."""


# Define function type aliases with generics
ObjectiveFunctionFloat = Callable[["ObjectiveArg[np.float64]"], None]
ObjectiveFunctionObject = Callable[["ObjectiveArg[np.object_]"], None]
ObjectiveFunction = Union[ObjectiveFunctionFloat, ObjectiveFunctionObject]

ObjectiveGradientFunction = Callable[["ObjectiveGradientArg"], None]
ObjectiveHessianFunction = Callable[["ObjectiveHessianArg"], None]

DiscreteFunctionFloat = Callable[["DiscreteArg[np.float64]"], None]
DiscreteFunctionObject = Callable[["DiscreteArg[np.object_]"], None]
DiscreteFunction = Union[DiscreteFunctionFloat, DiscreteFunctionObject]

DiscreteJacobianFunction = Callable[["DiscreteJacobianArg"], None]
DiscreteHessianFunction = Callable[["DiscreteHessianArg"], None]

ContinuousFunctionFloat = Callable[["ContinuousArg[np.float64]"], None]
ContinuousFunctionObject = Callable[["ContinuousArg[np.object_]"], None]
ContinuousFunction = Union[ContinuousFunctionFloat, ContinuousFunctionObject]

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
