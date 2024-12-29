"""

Module that defines the `Guess` class and its associated methods.

"""

# future imports
from __future__ import annotations

# standard library imports
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from scipy.interpolate import interp1d

# package imports
from .structure import DVStructure, get_nlp_dv_structure
from .types_ import Protected

if TYPE_CHECKING:
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import ArrayLike, NDArray

    # package imports
    import yapss

    from .problem import Problem
    from .solution import Solution

    # Float array
    Array = NDArray[np.float64]


class PhaseArrayGuess:
    """Descriptor class for the guess of a single phase."""

    def __set_name__(self, owner: PhaseGuess, name: str) -> None:
        """Save attribute name and create private attribute names."""
        self.name = name
        self.private_name = "_" + name
        self.len_name = "_n_" + name

    def __get__(self, instance: PhaseGuess, owner: type) -> Array:
        """Get the value of the guess."""
        value = getattr(instance, self.private_name)
        assert isinstance(value, np.ndarray)  # noqa: S101
        return value

    def __set__(
        self,
        instance: PhaseGuess,
        value: Sequence[Sequence[float | int]] | Sequence[Array] | Array,
    ) -> None:
        """Set the value of the guess."""
        value = np.asarray(value, dtype=float)
        shape = value.shape
        array_dimensions = 2
        if len(shape) != array_dimensions:
            msg = f"'guess.phase[{instance._p}].{self.name}' must be a 2-dimensional array."
            raise ValueError(msg)
        if shape[0] != getattr(instance, self.len_name):
            msg = (
                f"Expected '{self.name}' in 'guess.phase[{instance._p}]' to have "
                f"{getattr(instance, self.len_name)} rows, but got {shape[0]}."
            )
            raise ValueError(msg)
        if shape[1] < array_dimensions:
            msg = f"'guess.phase[{instance._p}].{self.name}' must have at least 2 columns."
            raise ValueError(msg)
        setattr(instance, self.private_name, value)


class Parameter(Protected):
    """Parameter descriptor."""

    name: str | None

    def __init__(self) -> None:
        self.name = None  # Name of the attribute, set in __set_name__

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the attribute."""
        self.name = name

    def __get__(self, instance: Guess, owner: type) -> NDArray[np.float64]:
        """Get the value of the parameter array."""
        if instance is None:
            # Return descriptor itself if accessed on the class
            return self  # type: ignore[unreachable]

        # Construct the private attribute name
        private_name = f"_attr_{self.name}"

        # Initialize _attr_parameter if it doesn't exist
        if not hasattr(instance, private_name):
            setattr(instance, private_name, np.zeros(instance._ns, dtype=np.float64))

        # Retrieve and return the attribute value
        value = getattr(instance, private_name)
        assert isinstance(value, np.ndarray)  # noqa: S101
        return value

    def __set__(self, instance: Guess, value: ArrayLike) -> None:
        """Set the value of the parameter array."""
        private_name = f"_attr_{self.name}"
        array_value = np.asarray(value, dtype=np.float64)

        # Check array shape
        if array_value.shape != (instance._ns,):
            msg = f"'guess.{self.name}' must be a 1-dimensional array of length {instance._ns}."
            raise ValueError(msg)

        # Set the parameter array on the instance
        setattr(instance, private_name, array_value)


class Guess:
    """Class that forms the interface to the user guess.

    Attributes
    ----------
    phase : tuple[PhaseGuess, ...]
        The guesses for each phase.
    parameter : NDArray[float]
        The guess for the problem parameters.
    """

    def __init__(self, problem: yapss.Problem) -> None:
        """Initialize the guess object.

        Parameters
        ----------
        problem : Problem
            The problem object.
        """
        # store information about the problem dimensions
        self._ns = problem.ns
        self._nx = problem.nx
        self._nu = problem.nu
        self._nq = problem.nq
        self._problem = problem

        # initialize the parameter guess
        self._parameter: Array = np.zeros([problem.ns], dtype=float)

        # initialize the guess for each phase
        phase = [PhaseGuess(problem, p) for p in range(len(problem.nx))]
        self._phase = tuple(phase)

    @property
    def phase(self) -> tuple[PhaseGuess, ...]:
        """The guesses for each phase."""
        return self._phase

    parameter = Parameter()

    def validate(self) -> None:
        """Validate user-provided initial guess."""
        for phase in self._phase:
            phase.validate()

    def __call__(self, solution: Solution) -> None:
        return self.from_solution(solution)

    def from_solution(self, solution: Solution) -> None:
        """Set the guess from a solution object.

        Parameters
        ----------
        solution : Solution
            A previous solution that will serve as the initial guess for a new solution.
        """
        # set the guess for each phase
        for p, phase in enumerate(solution.phase):
            self.phase[p].time = phase.time
            self.phase[p].state = phase.state
            self.phase[p].control = phase.control
            self.phase[p].integral = phase.integral

        # set the guess for the problem parameters
        self.parameter = solution.parameter


class TimeGuess(Protected):
    """Time descriptor."""

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        """Set the name of the attribute."""
        self.name = name

    def __get__(self, instance: PhaseGuess, owner: type) -> Array | None:
        """Get the value of the time array."""
        return getattr(instance, "_" + self.name, None)

    def __set__(self, instance: PhaseGuess, value: Sequence[float] | Array) -> None:
        """Set the value of the time array."""
        min_length = 2
        p = instance._p
        t: Array = np.asarray(value, dtype=float)
        shape = t.shape
        base_msg = (
            f"Expected 'guess.phase[{p}].time' to be a strictly increasing, 1-dimensional array "
            f"with at least {min_length} elements, "
        )
        if len(shape) != 1 or t.shape[0] < min_length:
            msg = base_msg + f"received shape {t.shape}."
            raise ValueError(msg)
        if np.any(np.diff(t) <= 0):
            msg = base_msg + "but the values were not strictly increasing."
            raise ValueError(msg)
        setattr(instance, "_" + self.name, t)
        instance._nt = len(t)


class PhaseGuess(Protected):
    """Class that forms the interface to the user guess."""

    _p: int
    _n_state: int
    _n_control: int
    _nq: int
    _nt: int | None
    _time: Array | None
    _state: Array | None
    _control: Array | None
    _integral: Array

    state: PhaseArrayGuess = PhaseArrayGuess()
    control: PhaseArrayGuess = PhaseArrayGuess()
    time: TimeGuess = TimeGuess()

    _allowed_attrs = (
        "_n_state",
        "_n_control",
        "_nq",
        "_p",
        "_nt",
        "_time",
        "_state",
        "_control",
        "_integral",
        "time",
        "state",
        "control",
        "integral",
    )

    def __init__(self, problem: yapss.Problem, p: int) -> None:
        self._p = p
        self._n_state = problem.nx[p]
        self._n_control = problem.nu[p]
        self._nq = problem.nq[p]
        self._nt: int | None = None
        self._time: Array | None = None
        self._state: Array | None = None
        self._control: Array | None = None
        self._integral: Array = np.zeros([self._nq])

    @property
    def integral(self) -> Array:
        """The guess for the integral values of the phase."""
        return self._integral

    @integral.setter
    def integral(self, value: ArrayLike) -> None:
        self._integral[:] = value

    def validate(self) -> None:
        """Validate the user-supplied guess for a phase."""
        p = self._p
        if self._time is None:
            msg = f"guess.phase[{p}].time has not been set."
            raise ValueError(msg)
        assert isinstance(self._nt, int)  # noqa: S101
        if self._state is None:
            self._state = np.zeros([self._n_state, self._nt], dtype=float)
        if self._control is None:
            self._control = np.zeros([self._n_control, self._nt], dtype=float)
        if self._state.shape != (self._n_state, self._nt):
            msg = (
                f"guess.phase[{p}].state must be a 2-dimensional array of "
                f"shape ({self._n_state}, {self._nt})."
            )
            raise ValueError(msg)
        if self._control.shape != (self._n_control, self._nt):
            msg = (
                f"guess.phase[{p}].control must be a 2-dimensional array of "
                f"shape ({self._n_control}, {self._nt})."
            )
            raise ValueError(msg)


def make_initial_guess_nlp(problem: Problem, computational_mesh: yapss._private.mesh.Mesh) -> Array:
    """Make initial guess for the NLP solution from the user-provided initial guess.

    This method takes the initial guess provided by the user and interpolates to produce an
    initial guess for the NLP solver.

    Returns
    -------
    NDArray
        Initial guess of the NLP decision variable array
    """
    # problem = guess._problem
    guess = problem.guess
    mesh = computational_mesh
    nlp_dv_guess: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)

    # guess for each phase
    for p, phase in enumerate(nlp_dv_guess.phase):
        tau_x = mesh.tau_x[p]
        tau_u = mesh.tau_u[p]
        time = guess.phase[p].time
        assert time is not None  # noqa: S101
        t0 = time[0]
        tf = time[-1]
        # tau is defined over the interval [-1, 1], so we need to scale and shift it to the
        # interval [t0, tf]
        t_x = (tf - t0) / 2 * tau_x + (t0 + tf) / 2
        t_u = (tf - t0) / 2 * tau_u + (t0 + tf) / 2

        phase.t0[0] = t0
        phase.tf[0] = tf

        # interpolate state and control variables
        state = guess.phase[p].state
        for i in range(problem.nx[p]):
            if problem.spectral_method != "lg":
                f = interp1d(time, state[i], fill_value="extrapolate")
                phase.x[i][:] = f(t_x)
            else:
                f = interp1d(time, state[i], fill_value="extrapolate")
                phase.xa[i][:] = f(t_x)

            if problem.spectral_method == "lgl":
                phase.xs[i][:] = 0.0

        control = guess.phase[p].control
        for i in range(problem.nu[p]):
            f = interp1d(time, control[i], fill_value="extrapolate")
            phase.u[i][:] = f(t_u)

        phase.q[:] = guess.phase[p].integral

    # guess for parameter
    nlp_dv_guess.s[:] = guess.parameter

    return nlp_dv_guess.z
