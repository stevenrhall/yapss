"""

What a callback is given and what it fills in.

Every callback receives a fresh argument object, and the ones that produce several named
results receive a fresh output object as well. Neither exposes an array, so there is nothing
to write into out of turn, nothing held across calls, and nothing to reset. Inputs are
read-only; an output is filled field by field and checked for completeness when the callback
returns.

Each class is generic in the declarations it carries, so a callback can be annotated and
checked, down to the field:

    @ph.register.continuous
    def continuous(
        arg: yapss.ContinuousArg[State, Control], out: yapss.ContinuousOut[State]
    ) -> None:
        out.dynamics.h = arg.state.v      # checked: State has h and v

The parameters name the vectors, not the phase's shape. The shape would be shorter, and the
vectors can be recovered from it by matching it against a protocol, which mypy follows; but
PyCharm's own type engine does not, and there an annotated callback got no completion and no
warnings at all. Plain type parameters are followed by every checker and every editor.
Unparameterized, every class degrades to `Any`, so an unannotated callback, or one annotated
with the bare class, is exactly as unchecked as before and never falsely reported.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Protocol

# `TypeVar` from typing_extensions for PEP 696 defaults, as in `declare`.
from typing_extensions import TypeVar

from .containers import suggest

if TYPE_CHECKING:
    from .declare import AnyPhase
    from .vector import Control, Discrete, Integral, Parameter, Path, State, Vector

__all__ = [
    "ContinuousArg",
    "ContinuousOut",
    "DiscreteArg",
    "DiscreteOut",
    "Endpoint",
    "Endpoints",
]

# What a class is generic in: the declarations it carries. Each is bounded by its role, so a
# state named where a control belongs is reported, and each defaults to `Any`, so the bare
# class checks nothing rather than refusing every name.
S_co = TypeVar("S_co", bound="State", covariant=True, default=Any)
"""The phase's state declaration."""
C_co = TypeVar("C_co", bound="Control", covariant=True, default=Any)
"""The phase's control declaration."""
P_co = TypeVar("P_co", bound="Path", covariant=True, default=Any)
"""The phase's path constraint declaration."""
PR_co = TypeVar("PR_co", bound="Parameter", covariant=True, default=Any)
"""The problem's parameter declaration."""
D_co = TypeVar("D_co", bound="Discrete", covariant=True, default=Any)
"""The problem's discrete constraint declaration."""
I_co = TypeVar("I_co", bound="Integral", covariant=True, default=Any)
"""A phase's integral declaration."""

_S_co = TypeVar("_S_co", bound="State", covariant=True)
_V_co = TypeVar("_V_co", bound="Integral", covariant=True)


class _HasEndpoints(Protocol[_S_co, _V_co]):
    """A phase handle, as far as its endpoints: what a shape's ``state`` and ``integral`` satisfy.

    The one place a protocol is still matched against a shape. ``arg[ph]`` differs by phase,
    so nothing written on the callback could type it; the handle can. Where a checker does not
    follow the match -- PyCharm's engine does not -- ``arg[ph]`` is merely unchecked.
    """

    @property
    def state(self) -> _S_co: ...
    @property
    def integral(self) -> _V_co: ...


class _Frozen:
    """Base of the argument objects: named slots, read-only, with suggestions on a typo."""

    __slots__: tuple[str, ...] = ()

    # Hidden from type checkers, as `Container`'s is: one that sees a reader answering any name
    # stops reporting misspellings. `ContinuousArg` declares its own, for the one name it cannot
    # know statically.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Refuse an unknown name with a suggestion."""
            if name.startswith("_"):
                raise AttributeError(name)
            names = getattr(self, "_names", None) or tuple(
                n for n in self.__slots__ if not n.startswith("_")
            )
            msg = f"{type(self).__name__} has no '{name}'.{suggest(name, names)}"
            raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment, saying why: an input, the objective, or no such name."""
        del value
        kind = type(self).__name__
        if hasattr(self, name):
            msg = f"{kind}.{name} is an input and cannot be assigned"
        elif name == "objective":
            # 0.3.0 wrote `arg.objective = ...`; the 0.4 objective callback returns it
            msg = (
                "the objective is returned from the objective callback, not assigned: 'return ...'"
            )
        else:
            names = getattr(self, "_names", None) or tuple(
                n for n in self.__slots__ if not n.startswith("_")
            )
            msg = f"{kind} has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion."""
        msg = f"{type(self).__name__}.{name} is an input and cannot be deleted"
        raise AttributeError(msg)


class ContinuousArg(_Frozen, Generic[S_co, C_co, PR_co]):
    """What a continuous callback is given, for one phase at one set of time points.

    Annotated ``yapss.ContinuousArg[State, Control, Parameter]`` -- the phase's state and
    control, and the problem's parameters -- a type checker follows ``arg.state``,
    ``arg.control`` and ``arg.parameter`` to their declarations. Trailing ones may be left out.

    Attributes
    ----------
    phase : Phase
        The phase this call is for. A callback shared between phases branches on it.
    time : numpy.ndarray
        The points the phase is evaluated at: its independent variable, whatever it measures.
    state, control, parameter : Vector
        The values at those points, one row per field.
    """

    __slots__ = ("control", "parameter", "phase", "state", "time")

    if TYPE_CHECKING:
        phase: AnyPhase
        state: S_co
        control: C_co
        parameter: PR_co
        time: Any

    _names = ("phase", "time", "state", "control", "parameter")

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


class ContinuousOut(_Frozen, Generic[S_co, P_co, I_co]):
    """What a continuous callback fills in: the dynamics, path constraints, and integrands.

    Each is a vector of the class the phase was declared with, so ``out.dynamics`` has the
    state's field names -- and, annotated ``yapss.ContinuousOut[State, Path, Integral]``, a type
    checker knows it. Trailing ones may be left out.
    """

    __slots__ = ("dynamics", "integrand", "path")

    if TYPE_CHECKING:
        dynamics: S_co
        path: P_co
        integrand: I_co

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

    def __delattr__(self, name: str) -> None:
        """Refuse deleting an output vector."""
        msg = f"out.{name} cannot be deleted; it is filled field by field"
        raise AttributeError(msg)

    def _is_complete(self) -> bool:
        """Report whether every output field has been assigned."""
        vectors: list[Vector] = [getattr(self, name) for name in self.__slots__]
        return all(vector._is_complete() for vector in vectors)

    def _missing(self) -> list[str]:
        """Return ``"<output>.<field>"`` for every field left unassigned."""
        missing: list[str] = []
        for name in self.__slots__:
            vector: Vector = getattr(self, name)
            missing.extend(f"{name}.{field}" for field in vector.missing())
        return missing


class Endpoints(Protocol):
    """How `DiscreteArg` reaches one phase's endpoint values: by its handle."""

    def __getitem__(self, handle: Any) -> Endpoint:
        """Return the endpoint values of `handle`."""
        ...


class Endpoint(_Frozen, Generic[S_co, I_co]):
    """The endpoint values of one phase, as an endpoint callback sees them.

    Typed by the phase's state and integral declarations, which `DiscreteArg` reads from the
    handle's shape. The state vectors read the transcription's own arrays where they are, so the
    endpoint is built once and keeps reading the current point; a time is a number, read afresh
    on each access, since a cached copy of one would go stale.

    Attributes
    ----------
    initial_state, final_state : Vector
        The phase's state at each end, one value per field.
    initial_time, final_time : Any
        The phase's time at each end: a float, or a symbol under the ``"auto"`` trace.
    integral : Vector
        The phase's integrals, which belong to the phase rather than to either end of it.
    """

    __slots__ = ("_data", "final_state", "initial_state", "integral")

    _names = ("initial_state", "initial_time", "final_state", "final_time", "integral")

    if TYPE_CHECKING:
        initial_state: S_co
        final_state: S_co
        integral: I_co

    def __init__(
        self, data: Any, initial_state: Vector, final_state: Vector, integral: Vector
    ) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "final_state", final_state)
        object.__setattr__(self, "integral", integral)

    @property
    def initial_time(self) -> Any:
        """The phase's time at its start."""
        return object.__getattribute__(self, "_data").initial_time

    @property
    def final_time(self) -> Any:
        """The phase's time at its end."""
        return object.__getattribute__(self, "_data").final_time


class DiscreteArg(_Frozen, Generic[PR_co]):
    """What the objective and discrete callbacks are given: what is evaluated once, not over time.

    That is every phase's endpoints and integrals, and the problem's parameters -- the discrete
    side of the problem, as `ContinuousArg` is the continuous side.

    A phase's endpoints are reached by indexing with its handle, ``arg[phases.coast]``.
    Annotated ``yapss.DiscreteArg[Parameter]``, a type checker follows ``arg.parameter``; what
    ``arg[ph]`` holds is typed from the handle itself, so it needs no parameter of its own.
    """

    __slots__ = ("_endpoints", "parameter")

    if TYPE_CHECKING:
        parameter: PR_co

    def __init__(self, endpoints: Endpoints, parameter: Vector) -> None:
        object.__setattr__(self, "_endpoints", endpoints)
        object.__setattr__(self, "parameter", parameter)

    def __getitem__(self, phase: _HasEndpoints[_S_co, _V_co]) -> Endpoint[_S_co, _V_co]:
        """Return the endpoint values of `phase`, which is a phase handle."""
        endpoints: Endpoints = object.__getattribute__(self, "_endpoints")
        try:
            return endpoints[phase]
        except (KeyError, TypeError):
            msg = (
                f"arg[...] takes a phase handle, such as 'problem.phases.<name>'; " f"got {phase!r}"
            )
            raise KeyError(msg) from None


class DiscreteOut(_Frozen, Generic[D_co]):
    """What the discrete callback fills in: the discrete constraint groups.

    Annotated ``yapss.DiscreteOut[Discrete]``, a type checker follows ``out.discrete``.
    """

    __slots__ = ("discrete",)

    if TYPE_CHECKING:
        discrete: D_co

    def __init__(self, discrete: Vector) -> None:
        object.__setattr__(self, "discrete", discrete)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse replacing the discrete vector, explaining how to fill it."""
        del value
        if name == "discrete":
            fields = getattr(getattr(self, "discrete"), "_fields", ())  # noqa: B009
            example = f"out.discrete.{fields[0]}" if fields else "out.discrete[:]"
            msg = (
                f"out.discrete cannot be replaced; fill it field by field, for example "
                f"'{example} = ...'."
            )
            raise AttributeError(msg)
        msg = f"out has no '{name}'.{suggest(name, self.__slots__)}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse deleting the discrete vector."""
        msg = f"out.{name} cannot be deleted; it is filled field by field"
        raise AttributeError(msg)

    def _is_complete(self) -> bool:
        """Report whether every group has been assigned."""
        discrete: Vector = getattr(self, "discrete")  # noqa: B009
        return bool(discrete._is_complete())

    def _missing(self) -> list[str]:
        """Return ``"discrete.<field>"`` for every group left unassigned."""
        discrete: Vector = getattr(self, "discrete")  # noqa: B009
        return [f"discrete.{field}" for field in discrete.missing()]
