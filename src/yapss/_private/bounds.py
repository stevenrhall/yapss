"""

Module bounds.

This module provides the interface that allows the user to set the bounds on the problem
decision variables and constraints. It also provides methods to reset the bounds, and to
validate that the bounds are consistent, so that for example no lower bound is greater than
the corresponding upper bound.

"""

# future imports
from __future__ import annotations

# standard library imports
from dataclasses import dataclass
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from numpy import float64

# package imports
from .structure import DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .types_ import Protected

if TYPE_CHECKING:
    # standard library imports

    # third party imports
    from numpy.typing import ArrayLike, NDArray

    # package imports
    import yapss

    FloatArray = NDArray[float64]


class ArrayBound:
    """Class to represent upper or lower bounds on an array of decision variables or constraints."""

    name: str
    private_name: str

    def __set_name__(self, owner: ArrayBounds, name: str) -> None:
        """Set the name of the attribute."""
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj: ArrayBounds, obj_type: type | None = None) -> NDArray[np.float64]:
        """Get the value of the attribute."""
        value = getattr(obj, self.private_name)
        assert isinstance(value, np.ndarray)  # noqa: S101
        return value

    def __set__(self, obj: ArrayBounds, value: ArrayLike) -> None:
        """Set the value of the attribute."""
        bound: NDArray[np.float64] = np.asarray(value, dtype=np.float64)
        if bound.shape != (obj._n,):
            msg = f"ArrayBound must be a sequence of floats of length {obj._n}."
            raise ValueError(msg)
        setattr(obj, self.private_name, bound)


class ArrayBounds(Protected):
    """Class to represent the bounds on a vector of decision variables or constraints.

    Attributes
    ----------
    lower : ArrayBound
        Lower bound on the decision variables or constraints.
    upper : ArrayBound
        Upper bound on the decision variables or constraints.
    """

    lower: ArrayBound = ArrayBound()
    upper: ArrayBound = ArrayBound()

    def __init__(self, phase_index: int, name: str, n: int) -> None:
        self._n = n
        self._p = phase_index
        self._name = name
        self._lower: NDArray[np.float64] = np.array(n * [-np.inf], dtype=float64)
        self._upper: NDArray[np.float64] = np.array(n * [+np.inf], dtype=float64)

        self._allowed_del_attrs = ()
        self._allowed_attrs = ("upper", "lower", "_upper", "_lower")

    def reset(self) -> None:
        """Reset the bounds to their default values."""
        self.lower[:] = -np.inf
        self.upper[:] = +np.inf

    def validate(self) -> None:
        """Validate the bounds."""
        diff = self.upper - self.lower
        indices = np.where(diff < 0)[0]
        if len(indices) > 0:
            if self._p >= 0:
                msg = (
                    "bounds.phase[{}].{}.lower[i] is greater than bounds.phase[{}].{}.upper[i] "
                    "for indices i in {}"
                )
                msg = msg.format(self._p, self._name, self._p, self._name, indices)
            else:
                msg = "bounds.{}.lower[i] is greater than bounds.{}.upper[i] for indices i in {}"
                msg = msg.format(self._name, self._name, indices)
            raise ValueError(msg)


class ScalarBound:
    """Class to represent upper or lower bound on a scalar decision variable or constraint."""

    def __set_name__(self, owner: ScalarBounds, name: str) -> None:
        """Set the name of the attribute."""
        self._name = name

    def __get__(self, obj: ScalarBounds, obj_type: type | None = None) -> float:
        """Get the value of the attribute."""
        return float(getattr(obj, "_" + self._name))

    def __delete__(self, obj: ScalarBounds) -> None:
        """Raise an error on attempt to delete the attribute."""
        msg = "can't delete attribute"
        raise AttributeError(msg)

    def __set__(self, obj: ScalarBounds, value: float) -> None:
        """Set the value of the attribute."""
        if not isinstance(value, (int, float)):
            msg = f"attribute '{self._name} must be a float, not {type(value)}"  # type: ignore[unreachable]
            raise TypeError(msg)
        setattr(obj, "_" + self._name, float(value))


class ScalarBounds(Protected):
    """Upper and lower bound pair for a scalar.

    Attributes
    ----------
    upper : float
        Upper bound.
    lower : float
        Lower bound.
    """

    _allowed_attrs = ("__dict__", "upper", "lower", "_upper", "_lower", "_name", "_p")

    upper: ScalarBound = ScalarBound()
    lower: ScalarBound = ScalarBound()

    def __init__(self, name: str, phase: int) -> None:
        """Initialize the upper and lower bounds to `+inf` and `-inf`, respectively."""
        self._upper: float = +float("inf")
        self._lower: float = -float("inf")
        self._name = name
        self._p = phase
        self.__dict__["lower"] = -float("inf")
        self.__dict__["upper"] = +float("inf")

    def reset(self) -> None:
        """Reset the bounds to their default values."""
        if self._name == "duration":
            self.lower = 0.0
        else:
            self.lower = -np.inf
        self.upper = np.inf

    def validate(self) -> None:
        """Validate the bounds."""
        if self.lower > self.upper:
            msg = "bounds.phase[{}].{}.lower is greater than bounds.phase[{}].{}.upper"
            msg = msg.format(self._p, self._name, self._p, self._name)
            raise ValueError(msg)
        if self._name == "duration" and self.upper < 0:
            msg = "bounds.phase[{}].duration.upper cannot be less than zero"
            msg = msg.format(self._p)
            raise ValueError(msg)


@dataclass(frozen=True)
class PhaseBounds:
    """Container for the bounds of a single phase of the optimal control problem.

    Attributes
    ----------
    initial_time : ScalarBounds
        Upper and lower bound on the initial time.
    final_time : ScalarBounds
        Upper and lower bound on the final time.
    duration : ScalarBounds
        Upper and lower bound on the duration.
    state : ArrayBounds
        Upper and lower bounds on the state variables.
    initial_state : ArrayBounds
        Upper and lower bounds on the initial state variables.
    final_state : ArrayBounds
        Upper and lower bounds on the final state variables.
    control : ArrayBounds
        Upper and lower bounds on the control variables.
    integral : ArrayBounds
        Upper and lower bounds on the integrals.
    path : ArrayBounds
        Upper and lower bounds on the path variables.
    zero_mode : ArrayBounds
        Upper and lower bounds on the zero modes.
    """

    initial_time: ScalarBounds
    final_time: ScalarBounds
    duration: ScalarBounds
    state: ArrayBounds
    initial_state: ArrayBounds
    final_state: ArrayBounds
    control: ArrayBounds
    integral: ArrayBounds
    path: ArrayBounds
    zero_mode: ArrayBounds

    def reset(self) -> None:
        """Reset the bounds to their default values."""
        self.initial_time.reset()
        self.final_time.reset()
        self.duration.reset()
        self.state.reset()
        self.initial_state.reset()
        self.final_state.reset()
        self.control.reset()
        self.integral.reset()
        self.path.reset()
        self.zero_mode.reset()

    def validate(self) -> None:
        """Validate the bounds."""
        self.initial_time.validate()
        self.final_time.validate()
        self.duration.validate()
        self.state.validate()
        self.initial_state.validate()
        self.final_state.validate()
        self.control.validate()
        self.integral.validate()
        self.path.validate()
        self.zero_mode.validate()

        # check that time bounds are feasible
        msg = None
        if self.final_time.upper - self.initial_time.lower < self.duration.lower:
            # TODO: phase should be ._p not .p
            #       also maybe should be an attribute of the phase?
            msg = (
                "Time bounds are infeasible:\nbounds.phase[{}].final_time.upper - "
                "bounds.phase[{}].initial_time.lower < bounds.phase[{}].duration.lower."
            )
        elif self.final_time.lower - self.initial_time.upper > self.duration.upper:
            msg = (
                "Time bounds are infeasible:\nbounds.phase[{}].final_time.lower - "
                "bounds.phase[{}].initial_time.upper > bounds.phase[{}].duration.upper."
            )
        if msg is not None:
            p = self.initial_time._p
            msg = msg.format(p, p, p)
            raise ValueError(msg)


class Bounds(Protected):
    """Represents bounds on variables and constraints in an optimal control problem.

    The `Bounds` class organizes and manages the bounds for decision variables and constraints
    in a hierarchical structure tailored to the optimal control problem. Each variable or
    constraint in the hierarchy has associated `upper` and `lower` bounds, along with methods
    to validate and reset the bounds.

    Each bounded variable in the hierarchy includes the following attributes:

        upper : float or np.ndarray
            The upper bound.
        lower : float or np.ndarray
            The lower bound.
        validate() : method
            Validates user-defined bounds, raising a `ValueError` if they are infeasible.
        reset() : method
            Resets bounds to default values (`+inf` for upper bounds, `-inf` for lower
            bounds), except for phase durations where the lower bound is set to zero.

    Attributes
    ----------
    parameter : ArrayBounds
        Bounds for the parameters in the optimal control problem.
    discrete : ArrayBounds
        Bounds for discrete constraint functions within the problem.
    phase : tuple of PhaseBounds
        A tuple of `PhaseBounds` instances, where each `PhaseBounds` instance represents the bounds
        associated with one phase in the problem. Each `PhaseBounds` instance has the attributes:

        control : ArrayBounds
            Bounds for control variables within the phase.
        state : ArrayBounds
            Bounds for state variables within the phase.
        initial_state : ArrayBounds
            Bounds on the initial state for the phase.
        final_state : ArrayBounds
            Bounds on the final state for the phase.
        initial_time : ScalarBounds
            Bounds on the initial time of the phase.
        final_time : ScalarBounds
            Bounds on the final time of the phase.
        duration : ScalarBounds
            Bounds on the duration of the phase, with a default lower bound of zero.
        integral : ArrayBounds
            Bounds for any integral values defined over the phase.
        path : ArrayBounds
            Bounds for path constraints applied to the phase.
        zero_mode : ArrayBounds
            Bounds for the state "zero modes"
    """

    discrete: ArrayBounds
    parameter: ArrayBounds
    phase: tuple[PhaseBounds, ...]

    def __init__(self, problem: yapss.Problem) -> None:
        """Initialize the bounds instance."""
        phase_bounds: list[PhaseBounds] = []

        for p in range(problem.np):
            phase = PhaseBounds(
                initial_time=ScalarBounds("initial_time", p),
                final_time=ScalarBounds("final_time", p),
                duration=ScalarBounds("duration", p),
                state=ArrayBounds(p, "state", problem.nx[p]),
                initial_state=ArrayBounds(p, "initial_state", problem.nx[p]),
                final_state=ArrayBounds(p, "final_state", problem.nx[p]),
                control=ArrayBounds(p, "control", problem.nu[p]),
                integral=ArrayBounds(p, "integral", problem.nq[p]),
                path=ArrayBounds(p, "path", problem.nh[p]),
                zero_mode=ArrayBounds(p, "zero_mode", problem.nx[p]),
            )
            phase.duration.lower = 0
            phase_bounds.append(phase)

        self.phase = tuple(phase_bounds)
        self.discrete = ArrayBounds(-1, "discrete", problem.nd)
        self.parameter = ArrayBounds(-1, "parameter", problem.ns)
        self._allowed_del_attrs = ()
        self._allowed_attrs = ()

    def reset(self) -> None:
        """Reset all bounds to default values across phases, parameters, and constraints."""
        for phase in self.phase:
            phase.reset()
        self.discrete.reset()
        self.parameter.reset()

    def validate(self) -> None:
        """Validate the bounds for each variable and constraint in the problem.

        Raises
        ------
        ValueError
            If infeasible bounds are detected.
        """
        for phase in self.phase:
            phase.validate()
        self.discrete.validate()
        self.parameter.validate()


# TODO: validate consistency of x, x0, xf


def get_nlp_decision_variable_bounds(problem: yapss.Problem) -> tuple[FloatArray, FloatArray]:
    """Determine the upper and lower bounds on the NLP decision variables.

    Function to determine the upper and lower bounds on the NLP decision variables based
    on the upper and  lower bounds in the optimal control problem statement.

    Parameters
    ----------
    problem : Problem
        The user-defined optimal control problem

    Returns
    -------
    tuple[NDArray, NDArray]
        The upper and lower bounds on the NLP decisionv variables. The length of each is
        the same as the number of decision variables.
    """
    # make structure that allows easy translation from problem statement bounds to NLP
    # bounds

    lb: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    ub: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)

    # do for each phase
    for p in range(problem.np):
        self_phase = problem.bounds.phase[p]

        # state bounds
        for i in range(problem.nx[p]):
            if problem.spectral_method == "lg":
                lb.phase[p].xa[i][:] = self_phase.state.lower[i]
                ub.phase[p].xa[i][:] = self_phase.state.upper[i]
            else:
                lb.phase[p].x[i][:] = self_phase.state.lower[i]
                ub.phase[p].x[i][:] = self_phase.state.upper[i]

            if problem.spectral_method == "lgl":
                lb.phase[p].xs[i][:] = self_phase.zero_mode.lower[i]
                ub.phase[p].xs[i][:] = self_phase.zero_mode.upper[i]

        # overwrite boundary value bounds
        lb.phase[p].x0[:] = np.maximum(self_phase.initial_state.lower, self_phase.state.lower)
        ub.phase[p].x0[:] = np.minimum(self_phase.initial_state.upper, self_phase.state.upper)
        lb.phase[p].xf[:] = np.maximum(self_phase.final_state.lower, self_phase.state.lower)
        ub.phase[p].xf[:] = np.minimum(self_phase.final_state.upper, self_phase.state.upper)

        # control bounds
        for i in range(problem.nu[p]):
            lb.phase[p].u[i][:] = self_phase.control.lower[i]
            ub.phase[p].u[i][:] = self_phase.control.upper[i]

        # integral bounds
        lb.phase[p].q[:] = self_phase.integral.lower
        ub.phase[p].q[:] = self_phase.integral.upper

        # boundary time bounds
        lb.phase[p].t0[:] = self_phase.initial_time.lower
        ub.phase[p].t0[:] = self_phase.initial_time.upper
        lb.phase[p].tf[:] = self_phase.final_time.lower
        ub.phase[p].tf[:] = self_phase.final_time.upper

    # parameter bounds
    lb.s[:] = problem.bounds.parameter.lower
    ub.s[:] = problem.bounds.parameter.upper

    return ub.z, lb.z


def get_nlp_constraint_function_bounds(
    problem: yapss.Problem,
) -> tuple[FloatArray, FloatArray]:
    """Determine the upper and lower bounds on the NLP decision variables.

    Parameters
    ----------
    problem : Problem
        The user-defined optimal control problem

    Returns
    -------
    tuple[NDArray, NDArray]
    """
    from .structure import CFStructure

    lb: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    ub: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)

    for p in range(problem.np):
        # state equation defect
        for i in range(problem.nx[p]):
            lb.phase[p].defect[i][:] = 0.0
            ub.phase[p].defect[i][:] = 0.0

        # integral equation defect
        lb.phase[p].integral[:] = 0.0
        ub.phase[p].integral[:] = 0.0

        # path
        for i in range(problem.nh[p]):
            lb.phase[p].path[i][:] = problem.bounds.phase[p].path.lower[i]
            ub.phase[p].path[i][:] = problem.bounds.phase[p].path.upper[i]

        # duration
        lb.phase[p].duration[:] = problem.bounds.phase[p].duration.lower
        ub.phase[p].duration[:] = problem.bounds.phase[p].duration.upper

        # zero mode
        if problem.spectral_method in ("lg", "lgr"):
            for i in range(problem.nx[p]):
                lb.phase[p].zero_mode[i][:] = problem.bounds.phase[p].zero_mode.lower[i]
                ub.phase[p].zero_mode[i][:] = problem.bounds.phase[p].zero_mode.upper[i]

    # discrete constraints
    lb.discrete[:] = problem.bounds.discrete.lower
    ub.discrete[:] = problem.bounds.discrete.upper

    return ub.c, lb.c
