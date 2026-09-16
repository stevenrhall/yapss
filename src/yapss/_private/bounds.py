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
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np
from numpy import float64

from .checked_array import CheckedArray, raise_if_invalid
from .coercion import real_array, real_scalar

# package imports
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .types_ import Protected, set_private

if TYPE_CHECKING:
    # standard library imports

    # third party imports
    from numpy.typing import ArrayLike, NDArray

    # package imports
    import yapss

    FloatArray = NDArray[float64]


@dataclass(frozen=True)
class BoundCheck:
    """What one side of a bound allows on its own: no NaN, and no infinity on the wrong side.

    Whether the two sides are consistent (lower not above upper) depends on both, so it is
    left to `ArrayBounds.validate`, and the sides can be assigned in either order.
    """

    side: str  # "lower" or "upper"

    def invalid(self, values: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return a mask of NaN values and of infinities on the wrong side."""
        wrong_side = np.inf if self.side == "lower" else -np.inf
        invalid: NDArray[np.bool_] = np.isnan(values) | (values == wrong_side)
        return invalid

    def describe(self, value: float) -> str:
        """Say what is wrong with a refused value."""
        return _describe_bound(self.side, value)


def _describe_bound(side: str, value: float) -> str:
    if np.isnan(value):
        return "is NaN."
    if side == "lower":
        return "is +inf; a lower bound must be less than +inf."
    return "is -inf; an upper bound must be greater than -inf."


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
        assert isinstance(value, np.ndarray)
        return value

    def __set__(self, obj: ArrayBounds, value: ArrayLike) -> None:
        """Set the value of the attribute."""
        # infinity is allowed on its own side: an unbounded side is +-inf
        label = f"{obj._path}.{self.name}"
        check = BoundCheck(self.name)
        bound = real_array(value, label, shape=(obj._n,))
        raise_if_invalid(check, label, bound)
        set_private(obj, self.private_name, CheckedArray.create(bound, label, check))


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
        self._path = f"bounds.phase[{phase_index}].{name}" if phase_index >= 0 else f"bounds.{name}"
        self._lower: NDArray[np.float64] = CheckedArray.create(
            np.full(n, -np.inf), f"{self._path}.lower", BoundCheck("lower")
        )
        self._upper: NDArray[np.float64] = CheckedArray.create(
            np.full(n, np.inf), f"{self._path}.upper", BoundCheck("upper")
        )

    def reset(self) -> None:
        """Reset the bounds to their default values."""
        self.lower[:] = -np.inf
        self.upper[:] = +np.inf

    def validate(self) -> None:
        """Validate the bounds.

        Raises
        ------
        ValueError
            If a bound is NaN, if a lower bound is ``+inf`` or an upper bound is ``-inf``
            (no value can satisfy it), or if a lower bound is greater than its upper bound.
        """
        path = self._path
        # NaN first: every comparison below is false for NaN, so a NaN bound would pass them
        # and reach Ipopt, whose interface rejects it with a message that names no bound.
        for side, values in (("lower", self.lower), ("upper", self.upper)):
            indices = np.flatnonzero(np.isnan(values))
            if len(indices) > 0:
                msg = f"{path}.{side}[i] is NaN for indices i in {indices}"
                raise ValueError(msg)
        # A lower bound of +inf or an upper bound of -inf leaves no feasible value. Equal
        # infinite bounds are not caught by the comparison below.
        for side, values, bad, sign, relation in (
            ("lower", self.lower, np.inf, "+inf", "less than +inf"),
            ("upper", self.upper, -np.inf, "-inf", "greater than -inf"),
        ):
            indices = np.flatnonzero(values == bad)
            if len(indices) > 0:
                msg = (
                    f"{path}.{side}[i] is {sign} for indices i in {indices}; "
                    f"a{'n' if side == 'upper' else ''} {side} bound must be {relation}"
                )
                raise ValueError(msg)
        # Compare directly rather than subtracting, which warns on inf - inf.
        indices = np.flatnonzero(self.lower > self.upper)
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

    def __set__(self, obj: ScalarBounds, value: float | np.floating[Any] | np.integer[Any]) -> None:
        """Set the value of the attribute, refusing a value that is wrong on its own."""
        label = f"{obj._path}.{self._name}"
        bound = real_scalar(value, label)
        if np.isnan(bound) or bound == (np.inf if self._name == "lower" else -np.inf):
            msg = f"{label} {_describe_bound(self._name, bound)}"
            raise ValueError(msg)
        # A negative duration bound would let the phase run backward in time: the duration
        # constraint's lower bound of zero is all that prevents tf < t0.
        if obj._name == "duration" and bound < 0:
            msg = f"{label} cannot be less than zero."
            raise ValueError(msg)
        set_private(obj, "_" + self._name, bound)


class ScalarBounds(Protected):
    """Upper and lower bound pair for a scalar.

    Attributes
    ----------
    upper : float
        Upper bound.
    lower : float
        Lower bound.
    """

    upper: ScalarBound = ScalarBound()
    lower: ScalarBound = ScalarBound()

    def __init__(self, name: str, phase: int) -> None:
        """Initialize the upper and lower bounds to `+inf` and `-inf`, respectively."""
        self._upper: float = +float("inf")
        self._lower: float = -float("inf")
        self._name = name
        self._p = phase
        self._path = f"bounds.phase[{phase}].{name}" if phase >= 0 else f"bounds.{name}"

    def reset(self) -> None:
        """Reset the bounds to their default values."""
        if self._name == "duration":
            self.lower = 0.0
        else:
            self.lower = -np.inf
        self.upper = np.inf

    def validate(self) -> None:
        """Validate the bounds.

        A value wrong on its own (NaN, an infinity on the wrong side, a negative duration
        bound) is refused by the setter, which is the only way to write a scalar bound, so
        only the relation between the two sides is left to check here.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        """
        if self.lower > self.upper:
            msg = "bounds.phase[{}].{}.lower is greater than bounds.phase[{}].{}.upper"
            msg = msg.format(self._p, self._name, self._p, self._name)
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
    # Bounds on the state "zero modes" -- the extra degree of freedom needed so the LGL
    # discretization is not overconstrained. Not user-facing: there is no legitimate
    # reason for a user to set these, so the attribute is private. Kept (rather than
    # removed) for research use.
    _zero_mode: ArrayBounds

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
        self._zero_mode.reset()

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
        self._zero_mode.validate()

        # The NLP bounds on the boundary states are the intersection of the state bounds
        # with the initial (final) state bounds, so each pair must overlap even when both
        # are individually consistent.
        p = self.initial_time._p
        for boundary in (self.initial_state, self.final_state):
            lower = np.maximum(boundary.lower, self.state.lower)
            upper = np.minimum(boundary.upper, self.state.upper)
            indices = np.where(upper < lower)[0]
            if len(indices) > 0:
                overlap_msg = (
                    "bounds.phase[{p}].{name} and bounds.phase[{p}].state do not overlap "
                    "for indices i in {indices}"
                )
                raise ValueError(overlap_msg.format(p=p, name=boundary._name, indices=indices))

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
                _zero_mode=ArrayBounds(p, "zero_mode", problem.nx[p]),
            )
            phase.duration.lower = 0
            phase_bounds.append(phase)

        self.phase = tuple(phase_bounds)
        self.discrete = ArrayBounds(-1, "discrete", problem.nd)
        self.parameter = ArrayBounds(-1, "parameter", problem.ns)

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

        # state bounds at every time point, and zero-mode bounds (empty unless LGL)
        for i in range(problem.nx[p]):
            lb.phase[p].x[i][:] = self_phase.state.lower[i]
            ub.phase[p].x[i][:] = self_phase.state.upper[i]
            lb.phase[p].xs[i][:] = self_phase._zero_mode.lower[i]
            ub.phase[p].xs[i][:] = self_phase._zero_mode.upper[i]

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

    # discrete constraints
    lb.discrete[:] = problem.bounds.discrete.lower
    ub.discrete[:] = problem.bounds.discrete.upper

    return ub.c, lb.c
