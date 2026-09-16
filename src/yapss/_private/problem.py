"""

Module problem.

"""

# future imports
from __future__ import annotations

import warnings

__all__ = ["Problem"]

import inspect

# standard imports
from collections.abc import Callable, Sequence
from types import FrameType, SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, assert_never, cast, get_args

# third party imports
import numpy as np
from numpy import float64

# package imports
from .bounds import Bounds
from .coercion import integer_scalar, integer_sequence, real_array, real_scalar
from .exceptions import YapssWarning
from .guess import Guess
from .ipopt_options import IpoptOptions
from .solution import warn_if_not_converged
from .solver import solve
from .types_ import (
    DerivativeMethod,
    DerivativeOrder,
    LimitOptions,
    Protected,
    Sense,
    SpectralMethod,
    set_private,
)

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
    from .types_ import CVName, DVName

    Array = NDArray[float64]

# default options
DEFAULT_NUMBER_OF_SEGMENTS = 10
DEFAULT_NUMBER_OF_COLLOCATION_POINTS = 10
DEFAULT_SPECTRAL_METHOD: SpectralMethod = "lgl"
DEFAULT_DERIVATIVE_METHOD: DerivativeMethod = "auto"
DEFAULT_DERIVATIVE_ORDER: DerivativeOrder = "second"
DEFAULT_SENSE: Sense = "minimize"


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
    catch_keyboard_interrupt : bool
        Whether ``solve()`` installs a SIGINT handler so that Ctrl-C asks Ipopt to stop
        at the next iteration and return the current iterate, rather than interrupting
        Python inside the solver. Defaults to ``True``. The handler can only be installed
        on the main thread; on any other thread the solve runs without it.
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
    sense : {"minimize", "maximize"}
        Whether the objective should be minimized or maximized. Defaults to
        ``"minimize"``. This is the only supported way to flip the sign of the
        objective -- ``scale.objective`` must be positive and controls magnitude
        conditioning only.
    """

    # Removed in 0.3.0 with the cyipopt backend. Remove this entry in 0.4.0 or after
    # 2027-09, whichever is later.
    _removed_attrs: ClassVar[dict[str, str]] = {
        "ipopt_source": (
            "'ipopt_source' was removed in YAPSS 0.3.0. YAPSS always uses the Ipopt library "
            "that CasADi loads, after verifying it, so this line can be deleted."
        ),
    }

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
    spectral_method: LimitOptions[SpectralMethod] = LimitOptions(get_args(SpectralMethod))
    sense: LimitOptions[Sense] = LimitOptions(get_args(Sense))

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
        self.scale = Scale(self)
        self.mesh = Mesh(self)

        self._abort: bool = False
        self.catch_keyboard_interrupt = True

        self.spectral_method = DEFAULT_SPECTRAL_METHOD
        self.sense = DEFAULT_SENSE

    def solve(self) -> Solution:
        """Solve the optimal control problem.

        A `Solution` is returned whatever Ipopt reports. If Ipopt did not converge,
        an `IpoptConvergenceWarning` is emitted -- the returned trajectory looks
        perfectly ordinary otherwise.

        Returns
        -------
        solution : Solution
            The solution to the optimal control problem.

        Warns
        -----
        IpoptConvergenceWarning
            If Ipopt reported a status other than 0 (optimal), 1 (acceptable level)
            or 6 (feasible point for a square problem).
        """
        solution = solve(self)
        # stacklevel=3: warn -> warn_if_not_converged -> this method -> user code.
        warn_if_not_converged(solution, stacklevel=3)
        return solution

    def validate(self) -> None:
        """Validate the optimal control problem input.

        Every part of the problem is checked, and all the failures are reported together:
        fixing one thing only to be told about the next is a poor way to find out that four
        things are wrong. Each part stops at its own first failure, so the report has at most
        one entry per part.

        Raises
        ------
        ValueError
            If the problem is invalid.
        """
        parts = (
            self.bounds.validate,
            self.guess.validate,
            self.scale.validate,
            self.mesh.validate,
            self._validate_functions,
        )
        problems = []
        for check in parts:
            try:
                check()
            except ValueError as error:
                problems.append(str(error))
        if not problems:
            return
        if len(problems) == 1:
            raise ValueError(problems[0])
        listed = "\n".join(f"  - {problem}" for problem in problems)
        msg = f"The problem is not ready to solve, for {len(problems)} reasons:\n{listed}"
        raise ValueError(msg)

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

    def __repr__(self) -> str:
        """Return the problem as it would be constructed.

        The default repr says only ``<yapss._private.problem.Problem object at 0x...>``,
        which in a debugger or a notebook does not even say which problem it is.
        """
        counts = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in ("nx", "nu", "nq", "nh", "ns", "nd")
        )
        return f"Problem(name={self.name!r}, {counts})"

    def __str__(self) -> str:
        """Return a short summary of the problem."""
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
        # the same rule as mesh collocation points: any sequence of integers, which covers
        # NumPy integers, ndarrays and range, but not floats or bools
        counts = integer_sequence(
            array,
            arg_name,
            minimum=0,
            allow_empty=True,  # nx=[] is a problem with no phases
        )
        if arg_name != "nx" and len(counts) != self.np:
            msg = (
                f"Length of '{arg_name}' must be the same as length of 'nx', "
                f"{self.np}, but it has {len(counts)}."
            )
            raise ValueError(msg)
        return counts

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

        for arg_name in ("ns", "nd"):
            value = integer_scalar(getattr(self, arg_name), arg_name)
            if value < 0:
                msg = f"{arg_name} must be a nonnegative integer, got {value}."
                raise ValueError(msg)
            set_private(self, arg_name, value)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:  # noqa: ARG002
        set_private(self, "_abort", value=True)

    def _intermediate_cb(self, *args: Any) -> bool:  # noqa: ARG002
        if self._abort:
            set_private(self, "_abort", value=False)
            return False
        return True


class Auxdata(SimpleNamespace):
    """Auxiliary problem data, which can be anything."""


def _check_scale(name: str, value: Array | float) -> None:
    """Raise unless every scale factor is finite and positive.

    A scale factor is a characteristic magnitude: the NLP divides by it, and the
    finite-difference methods size their steps with it. Zero divides by zero, and a NaN
    passes through Ipopt's user scaling unchecked and crashes the process (``not value >
    0`` catches NaN; ``value <= 0`` does not). A negative factor is rejected because Ipopt
    cannot honor the sign: it scales the bound vectors x_L, x_U, d_L, and d_U elementwise
    without swapping them (``OrigIpoptNLP::InitializeStructures`` via
    ``StandardScalingBase::apply_vector_scaling_x``, Ipopt 3.14), so a negative variable
    scale inverts the variable's bounds, and a negative constraint scale inverts the bounds
    of an inequality. The sign would have no effect on conditioning anyway: the scaled KKT
    matrix is a congruence of the unscaled one. The objective's sign is ``Problem.sense``.
    """
    if not np.all(np.isfinite(value)) or not np.all(np.asarray(value) > 0):
        msg = f"{name} must be finite and positive, got {value!r}."
        raise ValueError(msg)


def _check_scale_elements(name: str, value: Array) -> None:
    """Raise unless every element of a stored scale array is finite and positive.

    The array setters check a whole assignment with `_check_scale`, but the getters return
    the stored array itself, so an element or slice assignment such as
    ``problem.scale.phase[0].dynamics[0] = 0`` never passes through a setter. This is the
    check that catches it, from `Problem.validate`, before the scale reaches the NLP. The
    message names the first offending element so that it points at the assignment that
    produced it.
    """
    array = np.asarray(value)
    bad = ~(np.isfinite(array) & (array > 0))
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        msg = f"{name}[{i}] must be finite and positive, got {float(array[i])!r}."
        raise ValueError(msg)


class ScaleArray:
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
        shape = getattr(instance, "_" + self.name).shape
        if hasattr(instance, "_p"):
            label = f"Scale '{self.name}' in phase {instance._p}"
        else:
            label = f"Scale '{self.name}'"
        scale = real_array(value, label, shape=shape)
        _check_scale(label, scale)
        set_private(instance, "_" + self.name, scale)


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

    _time: float

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
        self._p: int = p
        self.time = 1.0
        self._state: Array = np.ones([problem.nx[p]], dtype=float)
        self._control: Array = np.ones([problem.nu[p]], dtype=float)
        self._integral: Array = np.ones([problem.nq[p]], dtype=float)
        self._dynamics: Array = np.ones([problem.nx[p]], dtype=float)
        self._path: Array = np.ones([problem.nh[p]], dtype=float)

    @property
    def time(self) -> float:
        """Time scale factor for the phase, shared by ``t``, ``t0``, and ``tf``."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        label = f"Scale 'time' in phase {self._p}"
        scale = real_scalar(value, label)
        _check_scale(label, scale)
        set_private(self, "_time", scale)

    def reset(self) -> None:
        """Reset every scale factor of the phase to 1.0.

        The arrays are filled with ones in place, as `Bounds.reset` does, so an array read
        before the reset still refers to the phase's scale factors.
        """
        for name in ("state", "control", "integral", "dynamics", "path"):
            getattr(self, "_" + name)[:] = 1.0
        set_private(self, "_time", 1.0)

    def validate(self) -> None:
        """Check every scale factor of the phase, including elements set in place.

        Raises
        ------
        ValueError
            If any scale factor is not finite and positive.
        """
        prefix = f"scale.phase[{self._p}]"
        for name in ("state", "control", "integral", "dynamics", "path"):
            _check_scale_elements(f"{prefix}.{name}", getattr(self, "_" + name))
        _check_scale(f"{prefix}.time", self._time)


class Scale(Protected):
    """Scaling object."""

    _objective: float

    phase: tuple[ScalePhase, ...]
    discrete: ScaleArray = ScaleArray()
    parameter: ScaleArray = ScaleArray()

    def __init__(self, ocp: Problem) -> None:
        self.phase = tuple(ScalePhase(ocp, p) for p in range(ocp.np))
        self._discrete: Array = np.ones([ocp.nd], dtype=float)
        self._parameter: Array = np.ones([ocp.ns], dtype=float)
        self.objective = 1.0

    @property
    def objective(self) -> float:
        """Objective scale factor. Magnitude conditioning only -- must be positive.

        Use `Problem.sense` to select minimization or maximization; this factor no
        longer carries sign.
        """
        return self._objective

    @objective.setter
    def objective(self, value: float) -> None:
        value = real_scalar(value, "scale.objective")
        if not np.isfinite(value) or not value > 0:
            msg = (
                f"'scale.objective' must be positive, got {value!r}. "
                "Use 'problem.sense = \"maximize\"' to maximize the objective instead "
                "of a negative scale factor."
            )
            raise ValueError(msg)
        set_private(self, "_objective", float(value))

    def reset(self) -> None:
        """Reset every scale factor to 1.0, in every phase (see `ScalePhase.reset`)."""
        for phase in self.phase:
            phase.reset()
        self._discrete[:] = 1.0
        self._parameter[:] = 1.0
        set_private(self, "_objective", 1.0)

    def validate(self) -> None:
        """Check every scale factor, including array elements set in place.

        Whole assignments are checked when they are made; an element or slice assignment
        is not, because the array getters return the stored array. `Problem.validate`
        calls this so that such a value is reported before it reaches the NLP, where a zero
        becomes an infinite Ipopt scaling factor and a negative one inverts the bounds of
        the variable or constraint it scales.

        Raises
        ------
        ValueError
            If any scale factor is not finite and positive.
        """
        for phase in self.phase:
            phase.validate()
        _check_scale_elements("scale.discrete", self._discrete)
        _check_scale_elements("scale.parameter", self._parameter)
        _check_scale("scale.objective", self._objective)

    def __getitem__(self, item: tuple[int, CVName | DVName, int]) -> float:
        """Return the characteristic magnitude of one decision variable.

        Convenience accessor used by the finite-difference derivative methods to size
        their perturbation steps. It relies on the scaling model having one scale per
        state and one per phase time: ``x``, ``x0``, and ``xf`` share the state scale,
        and ``t``, ``t0``, and ``tf`` share the time scale. If separate endpoint scales
        are ever introduced, this mapping must be split accordingly.
        """
        p, v, i = item
        match v:
            case "s":
                return float(self._parameter[i])
            case "x" | "x0" | "xf":
                return float(self.phase[p].state[i])
            case "u":
                return float(self.phase[p].control[i])
            case "t" | "t0" | "tf":
                return float(self.phase[p].time)
            case "q":
                return float(self.phase[p].integral[i])
            case _:
                assert_never(v)


class Derivatives(Protected):
    """Derivative class.

    Attributes
    ----------
    method : {"auto", "central-difference", "central-difference-full", "user"}
        Method used to compute derivatives.
    order : {"first", "second"}
        Order of derivatives used in search for optimum.
    """

    order: LimitOptions[DerivativeOrder] = LimitOptions(get_args(DerivativeOrder))
    """Order of derivatives used in search for optimum."""

    method: LimitOptions[DerivativeMethod] = LimitOptions(get_args(DerivativeMethod))
    """Method used to compute derivatives."""

    def __init__(self) -> None:
        """Initialize the derivative object with default values."""
        super().__init__()
        self._method = DEFAULT_DERIVATIVE_METHOD
        self._order = DEFAULT_DERIVATIVE_ORDER


F = TypeVar("F", bound=Callable[..., Any])


class Callback(Generic[F]):
    """A `UserFunctions` slot: a callable taking exactly one argument, or None."""

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name."""
        self.name = name

    def __get__(self, instance: UserFunctions | None, owner: type) -> F | None:
        """Return the callback, or None if it has not been set."""
        if instance is None:
            return self  # type: ignore[return-value]
        return cast("F | None", instance.__dict__.get("_" + self.name))

    def __set__(self, instance: UserFunctions, value: F | None) -> None:
        """Set the callback after checking it can be called with one argument."""
        if value is not None:
            msg = f"Value of '{self.name}' must be a callable object with one argument, or None."
            if not callable(value):
                raise TypeError(msg)
            # the question is whether it can be *called* with one argument, not how many
            # parameters it has: extra parameters with defaults, and *args/**kwargs, are
            # all fine. `bind` answers exactly that.
            signature = inspect.signature(value)
            try:
                signature.bind(None)
            except TypeError:
                msg = (
                    f"Value of '{self.name}' must be a callable object with one argument, "
                    f"or None; {getattr(value, '__name__', value)}{signature} cannot be "
                    f"called with one."
                )
                raise TypeError(msg) from None
        set_private(instance, "_" + self.name, value)

    def __delete__(self, instance: UserFunctions) -> None:
        """Refuse deletion, saying how to unset a callback."""
        msg = f"cannot delete 'UserFunctions' attribute '{self.name}'; set to None instead"
        raise AttributeError(msg)


class UserFunctions(Protected):
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

    objective: Callback[ObjectiveFunction] = Callback()
    objective_gradient: Callback[ObjectiveGradientFunction] = Callback()
    objective_hessian: Callback[ObjectiveHessianFunction] = Callback()
    continuous: Callback[ContinuousFunction] = Callback()
    continuous_jacobian: Callback[ContinuousJacobianFunction] = Callback()
    continuous_hessian: Callback[ContinuousHessianFunction] = Callback()
    discrete: Callback[DiscreteFunction] = Callback()
    discrete_jacobian: Callback[DiscreteJacobianFunction] = Callback()
    discrete_hessian: Callback[DiscreteHessianFunction] = Callback()


# The point above which `LargeSegmentWarning` suggests splitting a segment. It is a
# judgement, not a cliff: nothing fails at 26 points. The figure is set well above
# published practice and well below where the cost becomes painful.
#
# Published hp-adaptive methods cap the degree per interval far lower: the method is
# parameterized as hp-Method(Nmin, Nmax) with "a user-specified upper limit Nmax >= 2 ...
# to prevent the polynomial degree from growing unreasonably large", and GPOPS-II's
# examples use ph-(4, 10) -- a maximum of 10 (Darby, Hager and Rao, "An hp-adaptive
# pseudospectral method for solving optimal control problems", Optimal Control
# Applications and Methods 32, 2011; Patterson and Rao, "GPOPS-II", ACM TOMS 41, 2014).
# Conditioning is the milder constraint: the first-derivative differentiation matrix
# conditions as O(N^2), so N = 100 costs about four digits, which double precision
# absorbs.
#
# What bites in YAPSS is the mesh setup. `quadrature.py` computes the nodes with mpmath,
# memoized per (method, count), and the cost grows quadratically: measured on an M-series
# Mac, LGL takes 0.02 s at 10 points, 0.03 s at 15, 0.08 s at 25, 0.32 s at 50, 1.2 s at
# 100, and 21 s at 400.
LARGE_SEGMENT_THRESHOLD = 15

# A fraction sequence is rescaled to sum to exactly 1, so that the segment boundaries are
# exact. Only rounding error is absorbed silently: seven sevenths sum to 0.9999999999999998,
# which must be accepted, while [0.5, 0.495] is a mistake the user should hear about.
#
# The measured worst case for n equal fractions, n = 2 to 2000, is 6.7e-16, so this leaves
# eight orders of margin. It is deliberately not tighter than that: the tolerance is there
# to read the user's intent to sum to 1, and a value that only just clears the rounding
# error would turn a slightly different way of computing the same fractions into an error.
FRACTION_SUM_TOLERANCE = 1e-8


class LargeSegmentWarning(YapssWarning):
    """A mesh segment has more collocation points than it probably should.

    The collocation points of a segment are the roots of a polynomial of that degree, so a
    segment with many points is a high-order fit over the whole segment. Published
    hp-adaptive methods raise the degree only to about 10 per interval before splitting the
    interval instead, and YAPSS computes the quadrature rule for a segment in high-precision
    arithmetic, at a cost that grows quadratically with the count. More, shorter segments
    are usually both more accurate and faster to set up.

    This is advice, not a limit: nothing fails above the threshold, and a deliberate
    single-segment (global) method is a legitimate thing to want. Silence it with
    ``warnings.simplefilter("ignore", yapss.LargeSegmentWarning)``.
    """


class MeshPhase(Protected):
    """MeshPhase instances represent the mesh structure of a phase of the NLP."""

    def __init__(self, phase_index: int = 0) -> None:
        self._p = phase_index
        self._fraction: Sequence[float] = 10 * (0.1,)
        self._collocation_points: Sequence[int] = 10 * (10,)

    @property
    def fraction(self) -> Sequence[float]:
        return self._fraction

    @fraction.setter
    def fraction(self, value: Sequence[float]) -> None:
        label = f"mesh.phase[{self._p}].fraction"
        fractions = real_array(value, label, finite=True)
        if fractions.ndim != 1 or fractions.size == 0:
            msg = f"{label} must be a non-empty one-dimensional sequence of positive numbers."
            raise ValueError(msg)
        bad = np.flatnonzero(fractions <= 0)
        if bad.size:
            msg = f"{label}[{bad[0]}] must be positive, got {fractions[bad[0]]}."
            raise ValueError(msg)
        total = float(fractions.sum())
        if abs(total - 1.0) > FRACTION_SUM_TOLERANCE:
            msg = f"{label} must sum to 1, but sums to {total}."
            raise ValueError(msg)
        # rescale by the residual rounding error, so the boundaries are exact
        set_private(self, "_fraction", tuple(float(f) / total for f in fractions))

    @property
    def collocation_points(self) -> Sequence[int]:
        return self._collocation_points

    @collocation_points.setter
    def collocation_points(self, value: Sequence[int]) -> None:
        # The true minimum depends on the spectral method: LG and LGR quadrature will
        # accept 1 point, but LGL requires at least 2. Since the spectral method can be
        # set independently of (and after) the mesh, we enforce the higher, universal
        # floor of 2 here so a mesh is never silently invalid for whichever method ends
        # up being selected. Values of 2 or 3 work but are rarely a good choice in
        # practice -- 4 or more collocation points per segment is recommended.
        label = f"mesh.phase[{self._p}].collocation_points"
        points = integer_sequence(value, label, minimum=2)
        for i, n in enumerate(points):
            if n > LARGE_SEGMENT_THRESHOLD:
                msg = (
                    f"{label}[{i}] is {n}, above the {LARGE_SEGMENT_THRESHOLD} points above "
                    f"which a segment is usually better split. A segment is fitted by a "
                    f"single polynomial of that degree, and its quadrature rule costs more "
                    f"to compute the larger it is; hp-adaptive methods raise the degree only "
                    f"to about 10 before splitting instead. Nothing fails above this, so "
                    f"filter yapss.LargeSegmentWarning if the mesh is deliberate."
                )
                warnings.warn(msg, LargeSegmentWarning, stacklevel=3)
        set_private(self, "_collocation_points", points)


class Mesh(Protected):
    """Mesh data for a problem."""

    phase: tuple[MeshPhase, ...]

    def __init__(self, problem: yapss.Problem) -> None:
        """Initialize the mesh object."""
        self.phase = tuple(MeshPhase(p) for p in range(problem.np))
        segments = DEFAULT_NUMBER_OF_SEGMENTS
        points = DEFAULT_NUMBER_OF_COLLOCATION_POINTS
        for p in range(problem.np):
            self.phase[p].collocation_points = segments * (points,)
            self.phase[p].fraction = segments * (1 / segments,)

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
                    f"mesh.phase[{p}].collocation_points has {len(cp)} segments but "
                    f"mesh.phase[{p}].fraction has {len(f)}; they must be the same length."
                )
                raise ValueError(msg)
