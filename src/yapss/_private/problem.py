"""

Module problem.

"""

# future imports
from __future__ import annotations

__all__ = ["Problem"]

import inspect

# standard imports
from collections.abc import Sequence
from types import FrameType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable

# third party imports
import numpy as np
from numpy import float64

# package imports
from .bounds import Bounds
from .guess import Guess
from .ipopt_options import IpoptOptions
from .solver import solve
from .types_ import LimitOptions, Protected

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .input_args import (
        ContinuousFunction,
        ContinuousHessianFunction,
        ContinuousJacobianFunction,
        DiscreteFunction,
        DiscreteHessianFunction,
        DiscreteJacobianFunction,
        ObjectiveFunction,
        ObjectiveGradientFunction,
        ObjectiveHessianFunction,
    )
    from .solution import Solution

    Array = NDArray[float64]

# default options
DEFAULT_NUMBER_OF_SEGMENTS = 10
DEFAULT_NUMBER_OF_COLLOCATION_POINTS = 10
DEFAULT_SPECTRAL_METHOD = "lgl"
DEFAULT_DERIVATIVE_METHOD = "auto"
DEFAULT_DERIVATIVE_ORDER = "second"


class Problem(Protected):
    """Instances of the `Problem` class define the optimal control problem.

    Parameters
    ----------
    name : str
        Name of the optimal control problem
    nx : Sequence[int]
        Number of states in each phase
    nu : Optional[Sequence[int]]
        Number of controls in each phase
    nq : Sequence[int], optional
        Number of integrals in each phase
    nh : Sequence[int], optional
        Number of path constraints in each phase
    ns : int, optional
        Number of parameters in problem
    nd : int, optional
        Number of discrete constraints in problem

    Attributes
    ----------
    name : str
        The name of the problem.
    np : int
        The number of phases in the problem.
    nx : tuple[int, ...]
        The number of state variables in each phase.
    nu : tuple[int, ...]
        The number of control variables in each phase.
    nq : tuple[int, ...]
        The number of integrals in each phase.
    nh : tuple[int, ...]
        The number of path constraints in each phase.
    nd : int
        The number of discrete constraints in the problem.
    ns : int
        The number of parameters in the problem.
    auxdata : Auxdata
        The auxiliary data for the problem.
    bounds : Bounds
        The bounds object structure for the problem.
    derivatives : Derivatives
        The derivative options for the problem. Attributes: `method`, `order`.
    functions : UserFunctions
        The user-defined functions for the problem. Attributes: `objective`, `continuous`,
        `discrete`, `objective_gradient`, `objective_hessian`, `continuous_jacobian`,
        `continuous_hessian`, `discrete_jacobian`, `discrete_hessian`.
    guess : Guess
        The initial guess for the problem. **Attributes**: `parameter`, `phase[p].time`,
        `phase[p].state`, `phase[p].control`, `phase[p].integral`.
    ipopt_options : IpoptOptions
        The user-selected Ipopt options for the problem. To select a particular Ipopt
        option, use the `ipopt_options` attribute of the `Problem` class. For example,
        for a Problem instance `problem`, set the `tol` option to 1e-6 by setting
        ``problem.ipopt_options.tol = 1e-6``.
    scale : Scale
        The scaling data structure for the problem.
    mesh : Mesh
        The mesh data structure for the problem.
    spectral_method : {"lg", "lgr", "lgl"}
        The type of interpolation used for the problem.
    """

    auxdata: Auxdata

    catch_keyboard_interrupt: LimitOptions[bool] = LimitOptions((True, False))

    name: str
    np: int
    nx: tuple[int, ...]
    nu: tuple[int, ...]
    nq: tuple[int, ...]
    nh: tuple[int, ...]
    ns: int
    nd: int
    bounds: Bounds
    functions: UserFunctions
    guess: Guess
    derivatives: Derivatives
    ipopt_options: IpoptOptions
    scale: Scale
    mesh: Mesh
    spectral_method: LimitOptions[str] = LimitOptions(("lgl", "lgr", "lg"))

    # TODO: mesh should be ReadOnlyProperty

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str,
        nx: Sequence[int],
        nu: Sequence[int] | None = None,
        nq: Sequence[int] | None = None,
        nh: Sequence[int] | None = None,
        ns: int | None = 0,
        nd: int | None = 0,
    ) -> None:
        """Initialize the optimal control problem name and dimensions.

        Parameters
        ----------
        name : str
            Name of the optimal control problem
        nx : {List[int], tuple[int, ...]}
            Number of states in each phase
        nu : {List[int], tuple[int, ...]}
            Number of controls in each phase
        nq : {List[int], tuple[int, ...]}
            Number of integrals in each phase
        nh : {List[int], tuple[int, ...]}
            Number of path constraints in each phase
        ns : int
            Number of parameters in problem
        nd : int
            Number of discrete constraints in problem
        """
        self.name = name
        self._nx = nx
        self._nu = nu
        self._nq = nq
        self._nh = nh
        self.ns: int = ns if ns is not None else 0
        self.nd: int = nd if nd is not None else 0
        self._np: int

        # validate input and put in canonical form
        self._check_input()

        # fields for user input
        self.auxdata: Auxdata = Auxdata()
        self.bounds = Bounds(self)
        self.derivatives: Derivatives = Derivatives()
        self.functions = UserFunctions()
        self.guess = Guess(self)
        self.ipopt_options = IpoptOptions()
        self.scale = self._init_scale()
        self.mesh = Mesh(self)

        self.__dict__["_ipopt_source"] = "default"

        self._abort: bool = False
        self.catch_keyboard_interrupt = True

        self._allowed_del_attrs = ()
        self._allowed_attrs = (
            "spectral_method",
            "_spectral_method",
            "catch_keyboard_interrupt",
            "ipopt_source",
            "_abort",
            "_catch_keyboard_interrupt",
        )
        self.spectral_method = DEFAULT_SPECTRAL_METHOD

    # ipopt_source getter
    @property
    def ipopt_source(self) -> str:
        value = self.__dict__["_ipopt_source"]
        if not isinstance(value, str):
            msg = "Internal error: 'ipopt_source' must have type 'str'"
            raise TypeError(msg)
        return value

    @ipopt_source.setter
    def ipopt_source(self, value: str) -> None:
        if not isinstance(value, str):
            msg = f"'ipopt_source' must have type 'str', not {type(value)}"  # type: ignore[unreachable]
            raise TypeError(msg)
        self.__dict__["_ipopt_source"] = value

    def solve(self) -> Solution:
        """Solve the optimal control problem.

        Returns
        -------
        solution : Solution
            The solution to the optimal control problem.
        """
        return solve(self)

    def validate(self) -> None:
        """Validate the optimal control problem input.

        Raises
        ------
        ValueError
            If the problem is invalid.
        """
        self.bounds.validate()
        self.guess.validate()
        self.mesh.validate()
        self._validate_functions()

    def _validate_functions(self) -> None:
        """Validate the user-defined functions.

        Raises
        ------
        ValueError
            If the functions are invalid.
        """
        if self.functions.objective is None:
            msg = "'functions.objective' function is required."
            raise ValueError(msg)
        if self.np > 0 and self.functions.continuous is None:
            msg = "'functions.continuous' function is required."
            raise ValueError(msg)
        if self.nd > 0 and self.functions.discrete is None:
            msg = "'functions.discrete' function is required."
            raise ValueError(msg)
        if self.derivatives.method == "user":
            if self.functions.objective_gradient is None:
                msg = "'functions.objective_gradient' function is required."
                raise ValueError(msg)
            if self.np > 0 and self.functions.continuous_jacobian is None:
                msg = "'functions.continuous_jacobian' function is required."
                raise ValueError(msg)
            if self.nd > 0 and self.functions.discrete_jacobian is None:
                msg = "'functions.discrete_jacobian' function is required."
                raise ValueError(msg)
            if self.derivatives.order == "second":
                if self.functions.objective_hessian is None:
                    msg = "'functions.objective_hessian' function is required."
                    raise ValueError(msg)
                if self.np > 0 and self.functions.continuous_hessian is None:
                    msg = "'functions.continuous_hessian' function is required."
                    raise ValueError(msg)
                if self.nd > 0 and self.functions.discrete_hessian is None:
                    msg = "'functions.discrete_hessian' function is required."
                    raise ValueError(msg)

    def __str__(self) -> str:
        """Return repr(self)."""
        return (
            f"Problem(\n"
            f"    name='{self.name}',\n"
            f"    nx={self.nx},\n"
            f"    nu={self.nu},\n"
            f"    nq={self.nq},\n"
            f"    nh={self.nh},\n"
            f"    nd={self.nd},\n"
            f"    ns={self.ns}\n"
            f")"
        )

    def _validate_integer_array(
        self,
        array: Sequence[int] | None,
        arg_name: str,
    ) -> tuple[int, ...]:
        if array is None:
            return self.np * (0,)
        if arg_name == "nx":
            msg = f"Keyword '{arg_name}' must be a tuple or list of positive integers."
        else:
            msg = (
                f"Keyword '{arg_name}' must be a tuple or list of nonnegative integers, "
                f"or None."
            )
        if not isinstance(array, (tuple, list)):
            raise TypeError(msg)
        if not all(isinstance(item, int) for item in array):
            raise TypeError(msg)
        if not all(item >= (0 if arg_name == "nx" else 0) for item in array):
            raise ValueError(msg)
        if arg_name != "nx" and len(array) != self.np:
            msg = f"Length of '{arg_name}' must be the same as length of 'nx'."
            raise ValueError(msg)
        return tuple(array)

    def _check_input(self) -> None:
        """Check the validity of the arguments.

        * The user must supply a value for `name` that is a string
        * The value of `nx` must be a list or tuple of positive integers
        * Each of `nu`, `nq`, and `nh` must be `None`, or a list or tuple of non-negative
          integers the same length as `nx`
        * ns, nd must be nonnegative integers

        Raises
        ------
        TypeError, ValueError
        """
        msg = "Value of keyword 'name' must be a nonempty string."
        if not isinstance(self.name, str):
            raise TypeError(msg)
        if len(self.name) == 0:
            raise ValueError(msg)

        self.nx = self._validate_integer_array(self._nx, "nx")
        self.np = len(self.nx)
        self.nu = self._validate_integer_array(self._nu, "nu")
        self.nq = self._validate_integer_array(self._nq, "nq")
        self.nh = self._validate_integer_array(self._nh, "nh")

        msg = "Argument 'ns' must a nonnegative integer or None."
        if not isinstance(self.ns, (int, type(None))):
            raise TypeError(msg)
        if self.ns < 0:
            raise ValueError(msg)

        msg = "Argument 'nd' must a nonnegative integer or None."
        if not isinstance(self.nd, (int, type(None))):
            raise TypeError(msg)
        if self.nd < 0:
            raise ValueError(msg)

    def _init_scale(self) -> Scale:
        """Initialize the scaling object."""

        def make_scale(n: int) -> SimpleNamespace:
            scale_shift = SimpleNamespace()
            scale_shift.scale = np.ones(n, dtype=float64)
            return scale_shift

        scales = SimpleNamespace()
        scales.parameter = make_scale(self.ns)
        scales.discrete = make_scale(self.nd)
        scales.objective = SimpleNamespace()
        scales.objective.scale = 1.0

        scales.phase = self.np * [None]
        p: int
        scales.phase = []
        for p in range(self.np):
            phase = SimpleNamespace()
            phase.state = make_scale(self.nx[p])
            phase.control = make_scale(self.nu[p])
            phase.dynamics = make_scale(self.nx[p])
            phase.integral = make_scale(self.nq[p])
            phase.path = make_scale(self.nh[p])
            phase.initial_time = make_scale(1)
            phase.final_time = make_scale(1)
            scales.phase.append(phase)

        return Scale(self)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        self._abort = True

    def _intermediate_cb(self, *args: Any) -> bool:  # noqa: ARG002
        if self._abort:
            self._abort = False
            return False
        return True


class Auxdata(SimpleNamespace):
    """Auxiliary problem data, which can be anything."""


class ScaleArray(Protected):
    """Scale array."""

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the attribute."""
        self.name = name

    def __get__(self, instance: ScalePhase | Scale | None, owner: type) -> Array:
        """Get the value of the scale array."""
        if instance is None:
            msg = f"attribute '{self.name}' can be accessed on instance objects only."
            raise AttributeError(msg)
        value = getattr(instance, "_" + self.name)
        if not isinstance(value, np.ndarray):
            msg = f"Internal error: '{self.name}' must be a numpy array."
            raise TypeError(msg)
        return value

    def __set__(
        self,
        instance: ScalePhase | Scale,
        value: Sequence[float] | Array,
    ) -> None:
        """Set the value of the scale array."""
        scale = np.array(value, dtype=float64)
        shape = getattr(instance, "_" + self.name).shape
        if scale.shape != shape:
            if hasattr(instance, "_p"):
                msg = (
                    f"Scale '{self.name}' in phase {instance._p} must be an array of length "
                    f"{shape[0]}."
                )
            else:
                msg = f"Scale '{self.name}' must be an array of length {shape[0]}."
            raise ValueError(msg)
        setattr(instance, "_" + self.name, scale)


class ScalePhase(Protected):
    """One phase of the scaling object.

    Attributes
    ----------
    time : float
    state : ScaleArray
    control : ScaleArray
    integral : ScaleArray
    dynamics : ScaleArray
    path : ScaleArray
    """

    _allowed_attrs = (
        "state",
        "control",
        "integral",
        "dynamics",
        "path",
        "time",
        "_state",
        "_control",
        "_integral",
        "_dynamics",
        "_path",
        "_p",
        "p",
    )

    state: ScaleArray = ScaleArray()
    """State scaling array for a single phase."""
    control: ScaleArray = ScaleArray()
    """Control scaling array for a single phase."""
    integral: ScaleArray = ScaleArray()
    """Integral scaling array for a single phase."""
    dynamics: ScaleArray = ScaleArray()
    """Dynamics scaling array for a single phase."""
    path: ScaleArray = ScaleArray()
    """Path scaling array for a single phase."""

    def __init__(self, problem: yapss.Problem, p: int) -> None:
        """Initialize the scaling object.

        Initialize the scaling object for phase `p` of  the problem, based on the `problem`
        object.

        Parameters
        ----------
        problem : Problem
            The problem object.
        p : int
            The phase index.
        """
        self.time = 1.0
        self._p: int = p
        self._state: Array = np.ones([problem.nx[p]], dtype=float)
        self._control: Array = np.ones([problem.nu[p]], dtype=float)
        self._integral: Array = np.ones([problem.nq[p]], dtype=float)
        self._dynamics: Array = np.ones([problem.nx[p]], dtype=float)
        self._path: Array = np.ones([problem.nh[p]], dtype=float)


class Scale(Protected):
    """Scaling object."""

    _allowed_attrs = (
        "_phase",
        "_discrete",
        "_parameter",
        "objective",
        "phase",
        "discrete",
        "parameter",
    )

    phase: tuple[ScalePhase, ...]
    discrete: ScaleArray = ScaleArray()
    parameter: ScaleArray = ScaleArray()

    def __init__(self, ocp: Problem) -> None:
        self.phase = tuple(ScalePhase(ocp, p) for p in range(ocp.np))
        self._discrete: Array = np.ones([ocp.nd], dtype=float)
        self._parameter: Array = np.ones([ocp.ns], dtype=float)
        self.objective = 1.0

    def __getitem__(self, item: tuple[int, str, int]) -> float:  # TODO: not correct
        """Get the scale value for a given item."""
        p, v, i = item
        if v == "s":
            return float(self._parameter[i])
        if v in ("x", "x0", "xf"):
            return float(self.phase[p].state[i])
        if v == "u":
            return float(self.phase[p].control[i])
        if v in ("t", "t0", "tf"):
            return self.phase[p].time
        if v == "q":
            return float(self.phase[p].integral[i])
        raise RuntimeError


class Derivatives(Protected):
    """Derivative class.

    Attributes
    ----------
    method : {"auto", "central-difference", "central-difference-full", "user"}
    order : {"first", "second"}
    """

    _allowed_attrs = ("_method", "_order", "method", "order")

    order: LimitOptions[str] = LimitOptions(("first", "second"))
    """Order of derivatives used in search for optimum."""

    method: LimitOptions[str] = LimitOptions(
        ("auto", "central-difference", "central-difference-full", "user"),
    )
    """Method used to compute derivatives."""

    def __init__(self) -> None:
        """Initialize the derivative object with default values."""
        super().__init__()
        self._method = DEFAULT_DERIVATIVE_METHOD
        self._order = DEFAULT_DERIVATIVE_ORDER


class UserFunctions:
    """Container for the user-defined callback functions and their derivatives.

    The `functions` attribute of a `Problem` instance is an instance of the `UserFunctions`
    class, which stores the user-defined callback functions and their derivatives. Every
    optimal control problem must have at least an objective function. Most problems will have
    one or more phases with dynamics, path constraints, and/or integrands, and these problems
    require at least a `continuous` callback function. Problems with discrete constraints
    require at least a `discrete` callback function.

    For problems that use automatic differentiation, or differentiation by finite differences,
    no further callbacks are required. For problems that use user-supplied derivatives,
    additional callback functions are required. The `objective_gradient` callback is required
    for problems that use user-supplied gradients, and the `objective_hessian` callback is
    required for problems that use user-supplied Hessians. The `continuous_jacobian`,
    `continuous_hessian`, `discrete_jacobian`, and `discrete_hessian` callbacks are required
    as appropriate for problems that use user-supplied derivatives.

    Attributes
    ----------
    objective : ObjectiveFunction | None
    continuous : ContinuousFunction | None
    discrete : DiscreteFunction | None
    objective_gradient : ObjectiveGradientFunction | None
    continuous_jacobian : ContinuousJacobianFunction | None
    discrete_jacobian : DiscreteJacobianFunction | None
    objective_hessian : ObjectiveHessianFunction | None
    continuous_hessian : ContinuousHessianFunction | None
    discrete_hessian : DiscreteHessianFunction | None
    """

    def __init__(self) -> None:
        """Initialize the user functions."""
        self.objective: ObjectiveFunction | None = None
        self.objective_gradient: ObjectiveGradientFunction | None = None
        self.objective_hessian: ObjectiveHessianFunction | None = None
        self.continuous: ContinuousFunction | None = None
        self.continuous_jacobian: ContinuousJacobianFunction | None = None
        self.continuous_hessian: ContinuousHessianFunction | None = None
        self.discrete: DiscreteFunction | None = None
        self.discrete_jacobian: DiscreteJacobianFunction | None = None
        self.discrete_hessian: DiscreteHessianFunction | None = None

    _function_names = (
        "objective",
        "objective_gradient",
        "objective_hessian",
        "continuous",
        "continuous_jacobian",
        "continuous_hessian",
        "discrete",
        "discrete_jacobian",
        "discrete_hessian",
    )

    def __setattr__(self, key: str, value: Callable[[Any], None] | None) -> None:
        """Set the value of the attribute."""
        if key not in self._function_names:
            msg = f"Cannot set 'UserFunctions' attribute '{key}'"
            raise AttributeError(msg)

        # None is a valid value
        if value is None:
            super().__setattr__(key, value)
            return

        # Check if the value is a callable object with one argument
        msg = f"Value of '{key}' must be a callable object with one argument, or None."
        if not callable(value):
            raise TypeError(msg)
        sig = inspect.signature(value)
        # Count the parameters in the signature
        params = sig.parameters
        if len(params) != 1 or not all(
            p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            for p in params.values()
        ):
            raise TypeError(msg)

        super().__setattr__(key, value)

    def __delattr__(self, key: str) -> None:
        """Raise an error if the attribute is deleted."""
        if key not in self._function_names:
            msg = f"Cannot delete function '{key}'; set to None instead."
            raise AttributeError(msg)
        msg = f"Cannot delete 'UserFunctions' attribute '{key}'"
        raise AttributeError(msg)


class MeshPhase(Protected):
    """MeshPhase instances represent the mesh structure of a phase of the NLP."""

    _allowed_attrs = (
        "collocation_points",
        "fraction",
        "_fraction",
        "_collocation_points",
    )
    _allowed_del_attrs = ()

    def __init__(self) -> None:
        self._fraction: Sequence[float] = 10 * (0.1,)
        self._collocation_points: Sequence[int] = 10 * (10,)

    @property
    def fraction(self) -> Sequence[float]:
        return self._fraction

    @fraction.setter
    def fraction(self, value: Sequence[float]) -> None:
        if not all(isinstance(f, (int, float)) and f > 0 for f in value):
            msg = "fraction must be a sequence of floats"
            raise TypeError(msg)
        s = sum(value)
        if not np.isclose(s, 1.0, atol=0.01):
            msg = f"Sum of mesh fractions must be close to 1.0. Sum is {s}"
            raise ValueError(msg)
        self._fraction = tuple(fraction / s for fraction in value)

    @property
    def collocation_points(self) -> Sequence[int]:
        return self._collocation_points

    @collocation_points.setter
    def collocation_points(self, value: Sequence[int]) -> None:
        if not isinstance(value, Sequence):
            msg = f"collocation_points must be a sequence of positive integers, not {value}"  # type: ignore[unreachable]
            raise TypeError(msg)
        # TODO: Change to minimum number of collocation points. 3?
        if not all(isinstance(i, int) and i > 0 for i in value):
            msg = "collocation_points must be a sequence of positive integers"
            raise ValueError(msg)
        self._collocation_points = tuple(value)


class Mesh(Protected):
    """Mesh data for a problem."""

    phase: tuple[MeshPhase, ...]

    def __init__(self, problem: yapss.Problem) -> None:
        """Initialize the mesh object."""
        self.phase = tuple(MeshPhase() for _ in range(problem.np))
        segments = DEFAULT_NUMBER_OF_SEGMENTS
        points = DEFAULT_NUMBER_OF_COLLOCATION_POINTS
        for p in range(problem.np):
            self.phase[p].collocation_points = segments * (segments,)
            self.phase[p].fraction = segments * (1 / points,)

    # TODO: should just init mesh inside class?

    def validate(self) -> None:
        """Validate the user-supplied mesh geometry.

        The method raises a `ValueError` or `TypeError` if the user supplied _mesh data is
        invalid.

        Raises
        ------
        ValueError
        """
        # fix below to use enumerate

        for p, phase in enumerate(self.phase):
            cp = phase.collocation_points
            f = phase.fraction
            if len(cp) != len(f):
                # TODO: Check error message
                msg = (
                    "mesh.phase[{}].col_points and mesh.phase[{}].fraction must be the same length"
                )
                raise ValueError(msg.format(p, p))
