"""

What a callback is given and what it fills in.

Every callback receives a fresh argument object, and the ones that produce several named
results receive a fresh output object as well. Neither exposes an array, so there is nothing
to write into out of turn, nothing held across calls, and nothing to reset. Inputs are
read-only; an output is filled field by field and checked for completeness when the callback
returns.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from .containers import suggest

if TYPE_CHECKING:
    from .vector import Vector

__all__ = [
    "DiscreteOutput",
    "Endpoint",
    "EndpointArg",
    "Endpoints",
    "PhaseArg",
    "PhaseOutput",
]


class _Frozen:
    """Base of the argument objects: named slots, read-only, with suggestions on a typo."""

    __slots__: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> Any:
        """Refuse an unknown name with a suggestion."""
        if name.startswith("_"):
            raise AttributeError(name)
        names = getattr(self, "_names", None) or tuple(
            n for n in self.__slots__ if not n.startswith("_")
        )
        msg = f"{type(self).__name__} has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment."""
        del value
        msg = f"{type(self).__name__}.{name} is an input and cannot be assigned"
        raise AttributeError(msg)


class PhaseArg(_Frozen):
    """What a continuous callback is given, for one phase at one set of time points.

    Attributes
    ----------
    phase : Phase
        The phase this call is for. A callback shared between phases branches on it.
    time : numpy.ndarray
        The time points.
    state, control, parameter : Vector
        The values at those points, one row per field.
    """

    __slots__ = ("control", "parameter", "phase", "state", "time")

    def __init__(
        self, phase: Any, time: Any, state: Vector, control: Vector, parameter: Vector
    ) -> None:
        for name, value in (
            ("phase", phase),
            ("time", time),
            ("state", state),
            ("control", control),
            ("parameter", parameter),
        ):
            object.__setattr__(self, name, value)


class PhaseOutput(_Frozen):
    """What a continuous callback fills in: the dynamics, path constraints, and integrands.

    Each is a vector of the class the phase was declared with, so ``out.dynamics`` has the
    state's field names.
    """

    __slots__ = ("dynamics", "integrand", "path")

    def __init__(self, dynamics: Vector, path: Vector, integrand: Vector) -> None:
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "integrand", integrand)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse replacing an output vector, explaining how to fill it."""
        del value
        if name in self.__slots__:
            fields = getattr(getattr(self, name), "_fields", ())
            example = f"out.{name}.{fields[0]}" if fields else f"out.{name}[:]"
            msg = (
                f"out.{name} cannot be replaced; fill it field by field, for example "
                f"'{example} = ...'."
            )
            raise AttributeError(msg)
        msg = f"out has no '{name}'.{suggest(name, self.__slots__)}"
        raise AttributeError(msg)

    def _is_complete(self) -> bool:
        """Report whether every output field has been assigned."""
        return bool(
            self.dynamics._is_complete()
            and self.path._is_complete()
            and self.integrand._is_complete()
        )

    def _missing(self) -> list[str]:
        """Return ``"<output>.<field>"`` for every field left unassigned."""
        missing: list[str] = []
        for name in self.__slots__:
            vector: Vector = getattr(self, name)
            missing.extend(f"{name}.{field}" for field in vector.missing())
        return missing


class Endpoints(Protocol):
    """How `EndpointArg` reaches one phase's endpoint values: by its handle."""

    def __getitem__(self, handle: Any) -> Endpoint:
        """Return the endpoint values of `handle`."""
        ...


class Endpoint(_Frozen):
    """The endpoint values of one phase, as an endpoint callback sees them.

    The state vectors read the transcription's own arrays where they are, so this object stays
    correct from one evaluation to the next and is built once rather than per call. The times
    are numbers rather than arrays, so they are read afresh on each access; a cached copy of
    one would go stale, which is the whole hazard this design exists to remove.

    Attributes
    ----------
    initial_time, final_time, duration : float or symbolic
        The phase's time endpoints, and their difference.
    initial_state, final_state, integral : Vector
        The state at each endpoint, and the phase's integrals.
    """

    __slots__ = ("_data", "final_state", "initial_state", "integral")

    _names = (
        "duration",
        "final_state",
        "final_time",
        "initial_state",
        "initial_time",
        "integral",
    )

    def __init__(
        self, data: Any, initial_state: Vector, final_state: Vector, integral: Vector
    ) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "final_state", final_state)
        object.__setattr__(self, "integral", integral)

    @property
    def initial_time(self) -> Any:
        """Return the time at which the phase begins."""
        return self._data.initial_time

    @property
    def final_time(self) -> Any:
        """Return the time at which the phase ends."""
        return self._data.final_time

    @property
    def duration(self) -> Any:
        """Return how long the phase lasts."""
        return self._data.final_time - self._data.initial_time


class EndpointArg(_Frozen):
    """What the objective and discrete callbacks are given: every phase's endpoints.

    A phase's endpoints are reached by indexing with its handle, ``arg[phases.coast]``.
    """

    __slots__ = ("_endpoints", "parameter")

    def __init__(self, endpoints: Endpoints, parameter: Vector) -> None:
        object.__setattr__(self, "_endpoints", endpoints)
        object.__setattr__(self, "parameter", parameter)

    def __getitem__(self, phase: Any) -> Endpoint:
        """Return the endpoint values of `phase`, which is a phase handle."""
        endpoints: Endpoints = object.__getattribute__(self, "_endpoints")
        try:
            return endpoints[phase]
        except (KeyError, TypeError):
            msg = (
                f"arg[...] takes a phase handle, such as 'problem.phases.<name>'; " f"got {phase!r}"
            )
            raise KeyError(msg) from None


class DiscreteOutput(_Frozen):
    """What the discrete callback fills in: the discrete constraint groups."""

    __slots__ = ("discrete",)

    def __init__(self, discrete: Vector) -> None:
        object.__setattr__(self, "discrete", discrete)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse replacing the discrete vector, explaining how to fill it."""
        del value
        if name == "discrete":
            fields = getattr(self.discrete, "_fields", ())
            example = f"out.discrete.{fields[0]}" if fields else "out.discrete[:]"
            msg = (
                f"out.discrete cannot be replaced; fill it field by field, for example "
                f"'{example} = ...'."
            )
            raise AttributeError(msg)
        msg = f"out has no '{name}'.{suggest(name, self.__slots__)}"
        raise AttributeError(msg)

    def _is_complete(self) -> bool:
        """Report whether every group has been assigned."""
        return bool(self.discrete._is_complete())

    def _missing(self) -> list[str]:
        """Return ``"discrete.<field>"`` for every group left unassigned."""
        return [f"discrete.{field}" for field in self.discrete.missing()]
