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

from functools import cache
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
    "DiscreteOut",
    "Endpoint",
    "EndpointArg",
    "EndpointValues",
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

_V_co = TypeVar("_V_co", bound="Integral", covariant=True)


class _HasIntegral(Protocol[_V_co]):
    """A phase handle, as far as its integrals: what ``integral: Effort`` in a shape satisfies.

    The one place a protocol is still matched against a shape. ``arg[ph]`` differs by phase,
    so nothing written on the callback could type it; the handle can. Where a checker does not
    follow the match -- PyCharm's engine does not -- ``arg[ph].integral`` is merely unchecked.
    """

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
        """Refuse every assignment."""
        del value
        msg = f"{type(self).__name__}.{name} is an input and cannot be assigned"
        raise AttributeError(msg)


class ContinuousArg(_Frozen, Generic[S_co, C_co, PR_co]):
    """What a continuous callback is given, for one phase at one set of time points.

    Annotated ``yapss.ContinuousArg[State, Control, Parameter]`` -- the phase's state and
    control, and the problem's parameters -- a type checker follows ``arg.state``,
    ``arg.control`` and ``arg.parameter`` to their declarations. Trailing ones may be left out.

    The independent variable is reached by the name the phase gave it, so this class is
    generated per name by `phase_arg_class`: a subclass with the points under a property of
    that name. Generating rather than intercepting keeps it an ordinary attribute read, which
    is what a callback that uses it does at every point of every call.

    Attributes
    ----------
    phase : Phase
        The phase this call is for. A callback shared between phases branches on it.
    time : numpy.ndarray
        The points the phase is evaluated at, under whatever the phase calls its independent
        variable -- ``time`` unless it was named otherwise.
    state, control, parameter : Vector
        The values at those points, one row per field.
    """

    __slots__ = ("_points", "control", "parameter", "phase", "state")

    _independent = "time"

    if TYPE_CHECKING:
        phase: AnyPhase
        state: S_co
        control: C_co
        parameter: PR_co
        time: Any

        # The independent variable is reached by the name the phase gave it, which nothing in
        # a type parameter can carry, so `arg.r` must not be an error. This reader answers for
        # it, at the top level only: `arg.state.x` is still checked. The cost is that
        # `arg.stat` passes the checker, and is refused at run time with a suggestion.
        def __getattr__(self, name: str) -> Any: ...

    def __init__(
        self, phase: Any, points: Any, state: Vector, control: Vector, parameter: Vector
    ) -> None:
        for name, value in (
            ("phase", phase),
            ("_points", points),
            ("state", state),
            ("control", control),
            ("parameter", parameter),
        ):
            object.__setattr__(self, name, value)

    @property
    def _names(self) -> tuple[str, ...]:
        return ("phase", "state", "control", "parameter", type(self)._independent)


@cache
def phase_arg_class(name: str) -> type[ContinuousArg[Any, Any, Any]]:
    """Return the `ContinuousArg` subclass whose independent variable is called `name`."""
    return type(
        f"ContinuousArg_{name}",
        (ContinuousArg,),
        {
            "__slots__": (),
            "_independent": name,
            name: property(lambda self: object.__getattribute__(self, "_points")),
        },
    )


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
    """How `EndpointArg` reaches one phase's endpoint values: by its handle."""

    def __getitem__(self, handle: Any) -> Endpoint:
        """Return the endpoint values of `handle`."""
        ...


class EndpointValues(_Frozen):
    """One end of a phase: its state there, and its independent variable there.

    The two are one namespace, because that is what they are -- the phase's variables at a
    point, which is also what the endpoint columns of a Jacobian index. The state vector reads
    the transcription's own array where it is, so it is built once and keeps reading the
    current point; the independent variable is a number, read afresh on each access, since a
    cached copy of one would go stale.

    A name is looked for on the state first and then matched against the independent variable,
    so the state's own error message is what a misspelling gets. Positions address the state's
    rows, which the independent variable is not one of: it is a scalar in the same namespace,
    reached by name.
    """

    __slots__ = ("_independent", "_name", "_state")

    def __init__(self, state: Vector, independent: Any, name: str) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_independent", independent)
        object.__setattr__(self, "_name", name)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the constructor's arguments. Only a solution's endpoints are ever pickled."""
        return (
            EndpointValues,
            tuple(object.__getattribute__(self, n) for n in ("_state", "_independent", "_name")),
        )

    def __getattr__(self, name: str) -> Any:
        """Return a state by name, or the independent variable by its own name."""
        if name.startswith("_"):
            raise AttributeError(name)
        independent = object.__getattribute__(self, "_name")
        if name == independent:
            return object.__getattribute__(self, "_independent")()
        state = object.__getattribute__(self, "_state")
        try:
            return getattr(state, name)
        except AttributeError:
            # the state's own message would offer only the state's names, and this namespace
            # holds one more: the independent variable
            names = (*type(state)._fields, independent)
            msg = f"{type(state)._label} has no '{name}'.{suggest(name, names)}"
            raise AttributeError(msg) from None

    def __getitem__(self, index: Any) -> Any:
        """Return the state's rows by position."""
        return object.__getattribute__(self, "_state")[index]

    def __len__(self) -> int:
        """Return the number of state rows."""
        return len(object.__getattribute__(self, "_state"))


class Endpoint(_Frozen, Generic[I_co]):
    """The endpoint values of one phase, as an endpoint callback sees them.

    Typed by the phase's integral declaration, which `EndpointArg` reads from the handle's
    shape. ``initial`` and ``final`` are not typed: each holds the state's fields *and* the
    independent variable, one namespace at run time, and a type that is one class plus one more
    name is an intersection, which Python's typing cannot write. Typing them as the state would
    report ``arg[ph].final.time`` as an error, and that is working code.

    Attributes
    ----------
    initial, final : EndpointValues
        The phase's variables at each end: its state there, and its independent variable.
    integral : Vector
        The phase's integrals, which belong to the phase rather than to either end of it.
    """

    __slots__ = ("_data", "final", "initial", "integral")

    _names = ("final", "initial", "integral")

    if TYPE_CHECKING:
        initial: Any
        final: Any
        integral: I_co

    def __init__(
        self, data: Any, initial: EndpointValues, final: EndpointValues, integral: Vector
    ) -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "final", final)
        object.__setattr__(self, "integral", integral)


class EndpointArg(_Frozen, Generic[PR_co]):
    """What the objective and discrete callbacks are given: every phase's endpoints.

    A phase's endpoints are reached by indexing with its handle, ``arg[phases.coast]``.
    Annotated ``yapss.EndpointArg[Parameter]``, a type checker follows ``arg.parameter``; what
    ``arg[ph]`` holds is typed from the handle itself, so it needs no parameter of its own.
    """

    __slots__ = ("_endpoints", "parameter")

    if TYPE_CHECKING:
        parameter: PR_co

    def __init__(self, endpoints: Endpoints, parameter: Vector) -> None:
        object.__setattr__(self, "_endpoints", endpoints)
        object.__setattr__(self, "parameter", parameter)

    def __getitem__(self, phase: _HasIntegral[_V_co]) -> Endpoint[_V_co]:
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

    def _is_complete(self) -> bool:
        """Report whether every group has been assigned."""
        discrete: Vector = getattr(self, "discrete")  # noqa: B009
        return bool(discrete._is_complete())

    def _missing(self) -> list[str]:
        """Return ``"discrete.<field>"`` for every group left unassigned."""
        discrete: Vector = getattr(self, "discrete")  # noqa: B009
        return [f"discrete.{field}" for field in discrete.missing()]
