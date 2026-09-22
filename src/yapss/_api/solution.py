"""

The solution, read under the names the problem declared.

Every quantity of a phase is a read-only vector with that phase's own field names, so a state is
``ps.state.h`` wherever it is reached, and a helper written against a callback's endpoint
values also accepts a solution's.

A solution is data and holds nothing but data. It keeps no callbacks, no reference to the
problem, and not even the classes the problem was declared with: a declaration may be made
inside a function, and a class made there cannot be pickled by reference. What it keeps of each
declaration is the *shape* -- its role, its name, and its fields with their sizes, which are
plain data -- and it answers under those names through a declaration rebuilt from the shape.
That costs nothing a user can see, since a vector class holds no behavior, except that
``isinstance(ps.state, State)`` is false. What it buys is that every solution pickles, however
its problem was declared.

"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from .args import EndpointValues
from .containers import suggest
from .kinds import ReadOnlyRows
from .vector import ROLES, Vector, role_of, scalar, vector

if TYPE_CHECKING:
    from .spec import PhaseSpec, ProblemSpec

__all__ = [
    "EndpointMultiplier",
    "PhaseMultiplier",
    "PhaseSolution",
    "ProblemMultiplier",
    "Solution",
]

Shape = tuple[str, str, tuple[tuple[str, int | None], ...]]
"""What a solution keeps of a declaration: its role, its name, and its fields with their sizes."""


def shape_of(declaration: type[Vector]) -> Shape:
    """Return the shape of `declaration`, which is plain data and pickles by value."""
    fields = tuple((name, declaration._meta[name].size) for name in declaration._fields)
    return (role_of(declaration) or "", declaration.__name__, fields)


def _reduce_vector(self: Vector) -> tuple[Any, ...]:
    """Pickle a solution vector as its shape, its label and its rows."""
    shape: Shape = getattr(type(self), "_shape")  # noqa: B009
    return (_rebuild_vector, (shape, self._label, self._source))


@cache
def _declaration(shape: Shape) -> type[Vector]:
    """Return a declaration with the fields `shape` names, rebuilt from the shape alone.

    It has the user's class name, so a message reads as it would have, and it answers every read
    the user's class would. It is made once per shape and cached, and it pickles its instances
    as their shape, so nothing about it needs to be importable.
    """
    role, name, fields = shape
    base = next((cls for cls in ROLES if cls._role == role), Vector)
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": name,
        "_shape": shape,
        "__reduce__": _reduce_vector,
    }
    for field, size in fields:
        namespace[field] = scalar() if size is None else vector(size)
    declaration: type[Vector] = type(name, (base,), namespace)
    return declaration


def _rebuild_vector(shape: Shape, label: str, rows: Any) -> Any:
    """Return a read-only solution vector of `shape` holding `rows`. Also what unpickling calls."""
    obj = _declaration(shape)._new(ReadOnlyRows, label, None)
    obj._fill(rows)
    return obj


def _vector(declaration: type[Vector], rows: Any, label: str) -> Any:
    """Return a read-only vector with the fields of `declaration`, holding `rows`."""
    return _rebuild_vector(shape_of(declaration), label, rows)


class _Fixed:
    """A value read at an endpoint of a solution, which does not change, unlike a callback's."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self) -> Any:
        return self.value

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Fixed, (self.value,))


def _endpoint(phase: PhaseSpec, rows: Any, independent: Any, label: str) -> EndpointValues:
    """Return one end of a phase, read as an endpoint callback reads it."""
    return EndpointValues(
        _vector(phase.state, rows, f"{label} state"), _Fixed(independent), phase.independent
    )


class _Record:
    """Base of the groups a solution holds: named slots, read-only, pickled as their values."""

    __slots__: tuple[str, ...] = ()
    _label = "the solution"

    def __init__(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def _names(self) -> tuple[str, ...]:
        return tuple(n for n in self.__slots__ if not n.startswith("_"))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the values, which are data all the way down."""
        return (type(self), ({n: object.__getattribute__(self, n) for n in self.__slots__},))

    def __getattr__(self, name: str) -> Any:
        """Refuse an unknown name with a suggestion."""
        if name.startswith("_"):
            raise AttributeError(name)
        msg = f"{self._label} has no '{name}'.{suggest(name, self._names())}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: a solution is a record of what was solved."""
        del value
        msg = f"a solution is read-only; '{name}' cannot be assigned"
        raise AttributeError(msg)


_STATE_BOUNDS_OWED = (
    "A state's bound multiplier has to be divided by the quadrature weight and, at a collocated "
    "endpoint, folded into the costate, and that is still to be done."
)


class EndpointMultiplier(_Record):
    """The multipliers at one end of a phase, in the namespace `ps.initial` and `ps.final` use.

    The independent variable's is reported, under the phase's name for it. A state's -- the
    multiplier of its initial or final condition -- is owed, and asking for one says so rather
    than calling the name unknown.
    """

    __slots__ = ("_fields", "_independent", "_value")
    _label = "the endpoint multipliers"

    def __getattr__(self, name: str) -> Any:
        """Return the independent variable's multiplier, or explain a state's absence."""
        if name.startswith("_"):
            raise AttributeError(name)
        independent = object.__getattribute__(self, "_independent")
        if name == independent:
            return object.__getattribute__(self, "_value")
        fields: tuple[str, ...] = object.__getattribute__(self, "_fields")
        if name in fields:
            msg = f"the multiplier of '{name}' at this endpoint is not reported yet. " + (
                _STATE_BOUNDS_OWED
            )
            raise AttributeError(msg)
        msg = f"{self._label} have no '{name}'.{suggest(name, (*fields, independent))}"
        raise AttributeError(msg)


class PhaseMultiplier(_Record):
    """A phase's multipliers, in the shapes of what they belong to: ``ps.multiplier``.

    Attributes
    ----------
    dynamics : Vector
        The costate: the multiplier of the dynamics, named by the state's fields. The same
        object as ``ps.costate``.
    control : Vector
        The multipliers of the controls' bounds, densities in time.
    path : Vector
        The multipliers of the path constraints, densities in time.
    integral : Vector
        One multiplier per integral.
    initial, final : EndpointMultiplier
        The multipliers at each end of the phase.
    duration : float
        The multiplier of the phase's extent.
    """

    __slots__ = ("control", "duration", "dynamics", "final", "initial", "integral", "path")
    _label = "the phase multipliers"

    def __getattr__(self, name: str) -> Any:
        """Explain the owed state-bound multipliers, or refuse an unknown name."""
        if name == "state":
            msg = (
                f"the multipliers of the state's bounds are not reported yet. {_STATE_BOUNDS_OWED}"
            )
            raise AttributeError(msg)
        return super().__getattr__(name)


class ProblemMultiplier(_Record):
    """The problem's multipliers: ``solution.multiplier``.

    Attributes
    ----------
    parameter : Vector
        The multipliers of the parameters' bounds.
    discrete : Vector
        The multipliers of the discrete constraints.
    """

    __slots__ = ("discrete", "parameter")
    _label = "the problem multipliers"


class PhaseSolution:
    """One phase of a solution.

    Attributes
    ----------
    time : numpy.ndarray
        The points every quantity of the phase is given on, under whatever the phase calls
        its independent variable -- `time` unless it was named otherwise.
    state, dynamics : Vector
        Arrays over `time`, named by the phase's state class.
    control : Vector
        Arrays over `time`, named by the phase's control class.
    path : Vector
        Arrays over `time`, named by the phase's path class.
    integrand : Vector
        Arrays over `time`, named by the phase's integral class.
    integral : Vector
        One value per integral.
    multiplier : PhaseMultiplier
        The multipliers, in the same shapes: ``ps.multiplier.path.g``.
    costate : Vector
        The multiplier of the dynamics, which is ``ps.multiplier.dynamics`` under the name the
        field uses for it -- the same object.
    initial, final : EndpointValues
        The phase's variables at each end, read as a callback reads them.
    duration : float
        The extent of the phase.
    hamiltonian : numpy.ndarray
        The Hamiltonian over `time`.
    mesh : Mesh
        The mesh the phase was solved on.
    """

    __slots__ = (
        "_independent",
        "_points",
        "control",
        "costate",
        "duration",
        "dynamics",
        "final",
        "hamiltonian",
        "initial",
        "integral",
        "integrand",
        "mesh",
        "multiplier",
        "path",
        "state",
    )

    def __init__(self, independent: str, values: dict[str, Any]) -> None:
        object.__setattr__(self, "_independent", independent)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def _from(cls, phase: PhaseSpec, data: Any) -> PhaseSolution:
        """Return the solution of `phase` from the back end's record of it."""
        label = f"phase '{phase.name}' solution"
        state = np.asarray(data.state)
        costate = _vector(phase.state, data.costate, f"{label} costate")
        fields = phase.state._fields

        def at_end(value: float) -> EndpointMultiplier:
            return EndpointMultiplier(
                {"_fields": fields, "_independent": phase.independent, "_value": value}
            )

        multiplier = PhaseMultiplier(
            {
                "dynamics": costate,
                "control": _vector(
                    phase.control, data.control_multiplier, f"{label} control multiplier"
                ),
                "path": _vector(phase.path, data.path_multiplier, f"{label} path multiplier"),
                "integral": _vector(
                    phase.integral, data.integral_multiplier, f"{label} integral multiplier"
                ),
                "initial": at_end(data.initial_time_multiplier),
                "final": at_end(data.final_time_multiplier),
                "duration": data.duration_multiplier,
            }
        )
        return cls(
            phase.independent,
            {
                "_points": data.time,
                "state": _vector(phase.state, state, f"{label} state"),
                "costate": costate,
                "multiplier": multiplier,
                "dynamics": _vector(phase.state, data.dynamics, f"{label} dynamics"),
                "control": _vector(phase.control, data.control, f"{label} control"),
                "path": _vector(phase.path, data.path, f"{label} path"),
                "integrand": _vector(phase.integral, data.integrand, f"{label} integrand"),
                "integral": _vector(phase.integral, data.integral, f"{label} integral"),
                "initial": _endpoint(phase, state[:, 0], data.time[0], f"{label} initial"),
                "final": _endpoint(phase, state[:, -1], data.time[-1], f"{label} final"),
                "duration": data.time[-1] - data.time[0],
                "hamiltonian": data.hamiltonian,
                "mesh": phase.mesh,
            },
        )

    def _values(self) -> dict[str, Any]:
        return {
            name: object.__getattribute__(self, name)
            for name in self.__slots__
            if name != "_independent"
        }

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the phase solution as its independent variable's name and its values."""
        return (PhaseSolution, (object.__getattribute__(self, "_independent"), self._values()))

    def __getattr__(self, name: str) -> Any:
        """Return the points under the independent variable's name, or refuse the name."""
        if name.startswith("_"):
            raise AttributeError(name)
        independent = object.__getattribute__(self, "_independent")
        if name == independent:
            return object.__getattribute__(self, "_points")
        names = (*(n for n in self.__slots__ if not n.startswith("_")), independent)
        msg = f"the phase solution has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: a solution is a record of what was solved."""
        del value
        msg = f"a solution is read-only; '{name}' cannot be assigned"
        raise AttributeError(msg)


class Solution:
    """The result of a solve.

    Attributes
    ----------
    objective : float
        The objective value.
    converged : bool
        Whether Ipopt reported a converged solve.
    status : IpoptStatus
        What Ipopt reported.
    name : str
        The name of the problem this is a solution to.
    method : str
        The spectral method it was solved with: ``"lgl"``, ``"lgr"`` or ``"lg"``.
    parameter, discrete : Vector
        The problem-level values, named by the classes the problem declared.
    multiplier : ProblemMultiplier
        Their multipliers, in the same shapes: ``solution.multiplier.discrete.d``.
    """

    __slots__ = (
        "_names",
        "_phases",
        "converged",
        "discrete",
        "method",
        "multiplier",
        "name",
        "objective",
        "parameter",
        "status",
    )

    def __init__(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def _from(cls, spec: ProblemSpec, record: Any) -> Solution:
        """Return the solution of `spec` from the back end's record of the solve.

        Only data is kept: nothing reached from here refers to the problem, its callbacks, or
        the classes it was declared with.
        """
        return cls(
            {
                "_names": tuple(phase.name for phase in spec.phases),
                "_phases": tuple(
                    PhaseSolution._from(phase, record.phase[phase.index]) for phase in spec.phases
                ),
                "objective": record.objective,
                "converged": record.converged,
                "status": record.status,
                "name": spec.name,
                "method": spec.method,
                "parameter": _vector(spec.parameter, record.parameter, "parameter"),
                "discrete": _vector(spec.discrete, record.discrete, "discrete"),
                "multiplier": ProblemMultiplier(
                    {
                        "parameter": _vector(
                            spec.parameter, record.parameter_multiplier, "parameter multiplier"
                        ),
                        "discrete": _vector(
                            spec.discrete, record.discrete_multiplier, "discrete multiplier"
                        ),
                    }
                ),
            }
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the solution as its values, which are data all the way down."""
        values = {name: object.__getattribute__(self, name) for name in self.__slots__}
        return (Solution, (values,))

    def __getitem__(self, phase: Any) -> PhaseSolution:
        """Return the solution for `phase`: a phase handle, or the phase's name.

        A handle is matched by its position and its name, not by identity, and a name is enough
        on its own, so a solution unpickled in another process is readable with no problem at
        all: ``solution["boost"]``.
        """
        names: tuple[str, ...] = object.__getattribute__(self, "_names")
        phases: tuple[PhaseSolution, ...] = object.__getattribute__(self, "_phases")
        if isinstance(phase, str):
            if phase in names:
                return phases[names.index(phase)]
            hint = suggest(phase, names) if names else " The problem declared no phases."
            msg = f"the solution has no phase {phase!r}.{hint}"
            raise KeyError(msg)
        index = getattr(phase, "_index", None)
        name = getattr(phase, "_name", None)
        if isinstance(index, int) and 0 <= index < len(names) and names[index] == name:
            return phases[index]
        msg = (
            f"solution[...] takes a phase handle, such as 'problem.phases.<name>', or a phase's "
            f"name; got {phase!r}"
        )
        raise KeyError(msg)

    def __getattr__(self, name: str) -> Any:
        """Refuse an unknown name with a suggestion."""
        if name.startswith("_"):
            raise AttributeError(name)
        names = tuple(n for n in self.__slots__ if not n.startswith("_"))
        msg = f"the solution has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: a solution is a record of what was solved."""
        del value
        msg = f"a solution is read-only; '{name}' cannot be assigned"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        """Return a short representation naming the objective and status."""
        return f"<Solution objective={self.objective!r} converged={self.converged!r}>"
