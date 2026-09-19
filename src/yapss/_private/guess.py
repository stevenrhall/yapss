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

from .coercion import real_array

# package imports
from .layout import problem_layout
from .structure import DVStructure, get_nlp_dv_structure
from .types_ import Protected, set_private

if TYPE_CHECKING:
    from collections.abc import Sequence

    # third party imports
    from numpy.typing import ArrayLike, NDArray

    from .mesh import Mesh
    from .problem import Problem
    from .solution import Solution
    from .spec import ProblemSpec

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
        """Get the value of the guess, creating the all-zeros default on first read.

        The phase's time array fixes the default's shape, so reading an unset guess before
        the time array is set raises; a guess that has been assigned is returned whether or
        not the time array is set. The default is stored, so indexing and slicing assign into it as
        expected (``state[0, :] = ...``). An array that is still all zeros is treated as
        the default when the time array changes length and is regenerated at the new
        length (see ``TimeGuess.__set__``); through 0.2.2 the stored default froze at the
        length it was created with and a later change to the time array alone then
        failed validation.
        """
        value = getattr(instance, self.private_name)
        if value is None and instance._nt is None:
            msg = (
                f"guess.phase[{instance._p}].{self.name} cannot be read before "
                f"guess.phase[{instance._p}].time is set: the time array's length fixes "
                f"the array's shape ({getattr(instance, self.len_name)} rows by the number "
                f"of time points)."
            )
            raise ValueError(msg)
        if value is None:
            assert instance._nt is not None
            value = np.zeros([getattr(instance, self.len_name), instance._nt], dtype=float)
            set_private(instance, self.private_name, value)
        assert isinstance(value, np.ndarray)
        return value

    def __set__(
        self,
        instance: PhaseGuess,
        value: Sequence[Sequence[float | int]] | Sequence[Array] | Array,
    ) -> None:
        """Set the value of the guess."""
        # copy: the guess must not alias the caller's array (a Solution's, typically)
        value = real_array(value, f"guess.phase[{instance._p}].{self.name}", finite=True)
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
        set_private(instance, self.private_name, value)


class Parameter:
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
            set_private(instance, private_name, np.zeros(instance._ns, dtype=np.float64))

        # Retrieve and return the attribute value
        value = getattr(instance, private_name)
        assert isinstance(value, np.ndarray)
        return value

    def __set__(self, instance: Guess, value: ArrayLike) -> None:
        """Set the value of the parameter array."""
        private_name = f"_attr_{self.name}"
        # copy, never alias the caller's
        array_value = real_array(
            value,
            f"guess.{self.name}",
            shape=(instance._ns,),
            finite=True,
        )

        # Set the parameter array on the instance
        set_private(instance, private_name, array_value)


class Guess(Protected):
    """Class that forms the interface to the user guess.

    Attributes
    ----------
    phase : tuple[PhaseGuess, ...]
        The guesses for each phase.
    parameter : NDArray[float]
        The guess for the problem parameters.
    """

    # `Protected` rejects assignment to any other name, so a misspelled attribute such as
    # `problem.guess.parmeter = ...` raises instead of being stored and ignored. The
    # private names are the fields set in `__init__` and the `Parameter` descriptor's
    # backing store.

    def __init__(self, problem: Problem) -> None:
        """Initialize the guess object.

        Parameters
        ----------
        problem : ProblemSpec
            The problem object.
        """
        # store information about the problem dimensions
        self._ns = problem.ns
        self._nx = problem.nx
        self._nu = problem.nu
        self._nq = problem.nq
        self._problem = problem

        # initialize the guess for each phase
        phase = [PhaseGuess(problem, p) for p in range(len(problem.nx))]
        self._phase = tuple(phase)

    @property
    def phase(self) -> tuple[PhaseGuess, ...]:
        """The guesses for each phase."""
        return self._phase

    parameter = Parameter()

    def reset(self) -> None:
        """Reset the guess to its state when the problem was created.

        Every phase is reset (see `PhaseGuess.reset`), and the parameter guess returns to
        zeros.
        """
        for phase in self._phase:
            phase.reset()
        set_private(self, "_attr_parameter", np.zeros(self._ns, dtype=np.float64))

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

            # control is defined on phase.time_c, which only coincides with phase.time
            # for the lgl spectral method. Interpolate/extrapolate onto phase.time so
            # the guess has state and control on a common time grid.
            control_interp = interp1d(
                phase.time_c,
                phase.control,
                axis=1,
                fill_value="extrapolate",
            )(phase.time)
            self.phase[p].control = control_interp

            self.phase[p].integral = phase.integral

        # set the guess for the problem parameters
        self.parameter = solution.parameter


class TimeGuess:
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
        # copy, never alias the caller's; a time guess must be finite, unlike a bound
        t: Array = real_array(value, f"guess.phase[{p}].time", finite=True)
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
        set_private(instance, "_" + self.name, t)
        set_private(instance, "_nt", len(t))
        # A stored state or control guess that is still all zeros is the default,
        # whether created on read or assigned as zeros; if its length no longer matches,
        # drop it so it is regenerated at the new length on the next read. An array with
        # values in it is kept: validate() reports the mismatch, since its values cannot
        # be resized on the user's behalf.
        for name in ("_state", "_control"):
            stored = getattr(instance, name)
            if stored is not None and stored.shape[1] != len(t) and not np.any(stored):
                set_private(instance, name, None)


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

    def __init__(self, problem: Problem, p: int) -> None:
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
        set_private(
            self,
            "_integral",
            real_array(
                value,
                f"guess.phase[{self._p}].integral",
                shape=(self._nq,),
                finite=True,
            ),
        )

    def reset(self) -> None:
        """Reset the guess for the phase to its state when the problem was created.

        The time, state, and control guesses become unset, so the time array must be set
        again before the state or control guess can be read, and the integral guess
        returns to zeros. New arrays are stored: an array read before the reset is no
        longer part of the guess.
        """
        set_private(self, "_time", None)
        set_private(self, "_nt", None)
        set_private(self, "_state", None)
        set_private(self, "_control", None)
        set_private(self, "_integral", np.zeros([self._nq]))

    def validate(self) -> None:
        """Validate the user-supplied guess for a phase."""
        p = self._p
        if self._time is None:
            msg = f"guess.phase[{p}].time has not been set."
            raise ValueError(msg)
        assert isinstance(self._nt, int)
        # an unset state or control guess is zeros, supplied on read; only a set one
        # can disagree with the time array
        if self._state is not None and self._state.shape != (self._n_state, self._nt):
            msg = (
                f"guess.phase[{p}].state must be a 2-dimensional array of "
                f"shape ({self._n_state}, {self._nt})."
            )
            raise ValueError(msg)
        if self._control is not None and self._control.shape != (self._n_control, self._nt):
            msg = (
                f"guess.phase[{p}].control must be a 2-dimensional array of "
                f"shape ({self._n_control}, {self._nt})."
            )
            raise ValueError(msg)


def make_initial_guess_nlp(problem: ProblemSpec, computational_mesh: Mesh) -> Array:
    """Make initial guess for the NLP solution from the user-provided initial guess.

    This method takes the initial guess provided by the user and interpolates to produce an
    initial guess for the NLP solver.

    Returns
    -------
    NDArray
        Initial guess of the NLP decision variable array
    """
    mesh = computational_mesh
    nlp_dv_guess: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)

    # guess for each phase
    for p, phase in enumerate(nlp_dv_guess.phase):
        tau_x = mesh.tau_x[p]
        tau_u = mesh.tau_u[p]
        time = problem.phases[p].guess_time
        t0 = time[0]
        tf = time[-1]
        # tau is defined over the interval [-1, 1], so we need to scale and shift it to the
        # interval [t0, tf]
        t_x = (tf - t0) / 2 * tau_x + (t0 + tf) / 2
        t_u = (tf - t0) / 2 * tau_u + (t0 + tf) / 2

        phase.t0[0] = t0
        phase.tf[0] = tf

        # interpolate state and control variables
        state = problem.phases[p].guess_state
        # tau_x is in time order; the stored order differs under LG, which time_order maps
        time_order = problem_layout(problem)[p].time_order
        for i in range(problem.nx[p]):
            f = interp1d(time, state[i], fill_value="extrapolate")
            phase.x[i][time_order] = f(t_x)
            phase.xs[i][:] = 0.0  # zero modes (empty unless LGL)

        control = problem.phases[p].guess_control
        for i in range(problem.nu[p]):
            f = interp1d(time, control[i], fill_value="extrapolate")
            phase.u[i][:] = f(t_u)

        phase.q[:] = problem.phases[p].guess_integral

    # guess for parameter
    nlp_dv_guess.s[:] = problem.guess_parameter

    return nlp_dv_guess.z
