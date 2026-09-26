"""

Phase declarations and the setup aspects reached from a phase.

A phase's shape is a class, whose annotations name the vectors it is built from, and a
problem's phases are a class whose annotations name each phase and give it a shape::

    class Arc(yapss.Phase):
        state: Rocket
        control: Thrust

    class Singular(yapss.Phase):
        state: Rocket
        control: Thrust
        path: SingularArc

    class Phases(yapss.Phases):
        boost: Arc
        singular: Singular
        coast: Arc

A shape is not a phase: two phases may share one, as boost and coast do. Declaring both in
class bodies is what lets a type checker follow them -- ``problem.phases.boost`` is an `Arc`,
and its ``state`` a `Rocket` -- and what refuses a state annotated where a control belongs,
since the base class declares each slot's role and the override must agree with it.

The shape fixes what every aspect of a phase contains, so ``ph.state.x.bounds``,
``ph.state.x.guess``, and the ``dynamics`` a callback fills all carry the same field names.

"""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, cast, overload

# `TypeVar` from typing_extensions, not typing: PEP 696 defaults are native only from
# Python 3.13, and the floor is 3.11. The defaults are what let `yapss.Phase[Slide, Angle]`
# be written without naming the two declarations a phase usually does not have.
from typing_extensions import TypeVar

from .containers import (
    Container,
    FillerT,
    HasRegistry,
    Registry,
    is_callable,
    is_subclass,
    suggest,
)
from .fields import Fields
from .kinds import Bounds, Guess, ScalarGuess, Scale, is_bool, is_pair, is_real
from .mesh import Mesh
from .vector import Control, Integral, Path, State, Vector, role_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["AnyPhase", "Independent", "Phase", "Phases"]

# Each parameter is bounded by its role, which is what makes a state handed over as a control a
# type error. Each default is the role itself: a role class declares no fields, so it says what
# an omitted argument means -- no field of this is known -- and it is an *ancestor* of every
# declaration of that role, which a default has to be. (The old `Empty` could not be: it was a
# sibling of every user declaration, and a covariant parameter whose default is not a supertype
# of what a declaration supplies makes `Problem[Mine, ...]` unassignable to the bare
# `yapss.Problem`.) Covariance is sound because all of these are reached only for reading.
S_co = TypeVar("S_co", bound=State, default=State, covariant=True)
"""The class declaring a phase's states."""
C_co = TypeVar("C_co", bound=Control, default=Control, covariant=True)
"""The class declaring a phase's controls."""
P_co = TypeVar("P_co", bound=Path, default=Path, covariant=True)
"""The class declaring a phase's path constraints."""
I_co = TypeVar("I_co", bound=Integral, default=Integral, covariant=True)
"""The class declaring a phase's integrals."""

AnyPhase: TypeAlias = "Phase[Any, Any, Any, Any]"
"""A phase whose declarations are not known statically, as when phases are iterated.

The parameters are `Any` rather than `Vector` on purpose: a phase reached this way says
nothing about which fields were declared, and `Vector` would make every field of it an error
rather than an unknown.
"""


@dataclass(frozen=True, slots=True)
class PhaseDeclaration:
    """What a `Phase` subclass declares, read from its annotations when the class is made."""

    state: type[Vector]
    control: type[Vector]
    path: type[Vector]
    integral: type[Vector]
    independent: str
    """What the phase calls its independent variable, which is `time` unless it named it."""


def declared_role(
    value: object,
    owner: str,
    slot: str,
    slots: dict[str, type[Vector]],
    *,
    annotation: bool = False,
) -> type[Vector]:
    """Return `value` if it is a declaration of `role`, or refuse it naming what it is.

    The role is read from the class hierarchy, so a state put where a control belongs is refused
    at the line that did it -- which, without the check, would build and solve a problem
    mislabelled throughout, and never raise at all.

    The likeliest way to get this wrong is to swap two, so when the class belongs in another
    slot of the same owner the message names that slot. Changing the class's base would also
    silence the error, and would be the wrong fix.

    Parameters
    ----------
    value : object
        What was given.
    owner : str
        What it was given to: a call such as ``"Problem"``, or a class such as ``"Slide"``.
    slot : str
        The keyword or annotation it was given as.
    slots : dict
        Every role slot of the same owner and the role each takes, so the slot's own role is
        known and a swapped one can be named.
    annotation : bool, default False
        Whether the slot is an annotation in a class body rather than a call's keyword, which
        is only a matter of how the message spells it.
    """
    role = slots[slot]
    base = f"yapss.{role.__name__}"
    where = f"{owner}.{slot} is annotated" if annotation else f"{owner}({slot}=) takes"
    if not is_subclass(value, Vector):
        if annotation:
            msg = f"{where} {value!r}; it takes a subclass of {base}, such as 'class X({base})'"
        else:
            msg = f"{where} a subclass of {base}, such as 'class X({base})'; got {value!r}"
        raise TypeError(msg)
    cls = cast("type[Vector]", value)
    actual = role_of(cls)
    if actual == role._role:
        return cls
    if actual is None:
        msg = (
            f"{owner}.{slot} takes a subclass of {base}, but {cls.__name__} has no role. "
            f"Declare it as 'class {cls.__name__}({base})'."
        )
        raise TypeError(msg)
    if annotation:
        msg = (
            f"{where} {cls.__name__}, which subclasses yapss.{actual.title()}; it takes a "
            f"subclass of {base}."
        )
        if actual in slots:
            msg += f" Did you mean '{actual}: {cls.__name__}'?"
    else:
        msg = (
            f"{where} a subclass of {base}, but {cls.__name__} subclasses "
            f"yapss.{actual.title()}."
        )
        if actual in slots:
            msg += f" Did you mean {actual}={cls.__name__}?"
    raise TypeError(msg)


_ROLES: dict[str, type[Vector]] = {
    "state": State,
    "control": Control,
    "path": Path,
    "integral": Integral,
}
"""A phase's slots and the role each takes, in the order a phase is declared."""

_RESERVED = ("mesh", "register", "name", "index")
"""A phase's own attributes, which its independent variable cannot also be called."""


def _not_a_slot(owner: str, name: str, value: object) -> str:
    """Return the message for an annotation that is neither a role slot nor `Independent`.

    A vector class under an unknown name is a misspelled slot, and is answered as one, since
    that is far likelier than an independent variable annotated with the wrong type.
    """
    hint = suggest(name, tuple(_ROLES))
    if is_subclass(value, Vector):
        role = role_of(cast("type[Vector]", value))
        if role in _ROLES and not hint:
            hint = f" {_named(value)} is a yapss.{role.title()}: did you mean '{role}'?"
        return (
            f"{owner}.{name} is not one of a phase's slots, which are "
            f"{', '.join(_ROLES)}.{hint}"
        )
    return (
        f"{owner}.{name} is annotated {_named(value)}. A phase annotates its vectors as "
        f"{', '.join(_ROLES)}, and names its independent variable with "
        f"'{name}: yapss.Independent'.{hint}"
    )


def own_annotations(cls: type, scope: dict[str, Any]) -> dict[str, Any]:
    """Return the annotations `cls` declares itself, resolved to the objects they name.

    They are read from the finished class, not its namespace, which is the one place Python 3.11
    through 3.15 agree: from 3.14, Python evaluates annotations lazily and puts
    ``__annotate_func__`` in the namespace in place of ``__annotations__``. Only the class's
    own are read, so what a declaration left out can be told from what it said.

    A module that begins ``from __future__ import annotations`` gives strings. They are resolved
    against the module and against `scope`, the namespace the class statement ran in, so a
    class declared inside a function may name another declared there.

    Parameters
    ----------
    cls : type
        The class being declared.
    scope : dict
        The local namespace of the code that declared it.

    Returns
    -------
    dict
        Each annotated name, mapped to what it names.
    """
    try:
        return inspect.get_annotations(cls, eval_str=True, locals=scope)
    except NameError as exc:
        msg = (
            f"{cls.__name__}: an annotation names {exc.name!r}, which is not defined where "
            f"{cls.__name__} is. The classes a declaration names are declared before it."
        )
        raise NameError(msg) from None


def _declaring_scope() -> dict[str, Any]:
    """Return the local namespace of the class statement being executed.

    Called from an ``__init_subclass__``, which Python calls from inside the creation of the
    class, so the caller's caller is the code that wrote ``class ...:``.
    """
    return dict(sys._getframe(2).f_locals)


def _check_namespace(
    owner: str, state: type[Vector], control: type[Vector], independent: str
) -> None:
    """Refuse a phase whose states, controls and independent variable share a name.

    Those three are one namespace, because that is what they are: the columns of the phase's
    Jacobian, which a derivative names without saying which vector it came from. A name
    belonging to two of them would name two columns.

    The check is made where the classes are brought together, in the phase's class body, and
    the message names the two classes rather than the phase, since renaming a member of one of
    them is the fix. The parameters are not here to be checked; they arrive as an argument to
    `Problem`, which checks them against this namespace there.

    Path and integral names are not in it. They are outputs, so they appear on the other side
    of a derivative and may collide with a variable freely.
    """
    shared = sorted(set(state._fields) & set(control._fields))
    if shared:
        msg = (
            f"{owner}: its state {state.__name__} and its control {control.__name__} both "
            f"declare {shared[0]!r}. A phase's states, controls and independent variable are one "
            f"namespace, so their names must differ; rename it in one of the two classes."
        )
        raise ValueError(msg)
    for role, declaration in (("state", state), ("control", control)):
        if independent in declaration._fields:
            msg = (
                f"{owner}: its {role} {declaration.__name__} declares {independent!r}, and that "
                f"is also the phase's independent variable, which you named. They are one "
                f"namespace, so their names must differ; rename "
                f"the {role}, or name the independent variable something else with "
                f"'<name>: yapss.Independent'."
            )
            raise ValueError(msg)


def _assigned(cls: type) -> list[str]:
    """Return the public names a declaring class assigns in its body.

    A declaration is made of annotations alone, and an assignment among them is nearly always
    an annotation mistyped -- ``state = Position`` for ``state: Position`` -- which would
    otherwise leave the slot silently empty. Refusing now is what can be relaxed later; the
    other way is not.
    """
    return [name for name in vars(cls) if not name.startswith("_")]


def _named(value: object) -> str:
    """Return how an annotation is named in a message: a class by its name, else its repr."""
    return value.__name__ if isinstance(value, type) else repr(value)


class Phases:
    """Base of a problem's phases. Subclass it and annotate each phase with its shape.

    Each annotation names a phase and gives it a `Phase` subclass, and the order of the
    annotations is the order of the phases::

        class Phases(yapss.Phases):
            boost: Arc
            coast: Arc

    The subclass is passed to `Problem`, which instantiates it; each name then gives the phase,
    an instance of its shape, used to set that phase up and to reach it in callbacks and
    solutions.
    """

    _declared: dict[str, type[AnyPhase]] = {}  # noqa: RUF012

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect the declared phases, in declaration order."""
        super().__init_subclass__(**kwargs)
        for base in cls.__bases__:
            if base is not Phases and issubclass(base, Phases):
                msg = (
                    f"{cls.__name__} cannot inherit from {base.__name__}, which already "
                    f"declares a problem's phases. Subclass yapss.Phases."
                )
                raise TypeError(msg)
        assigned = _assigned(cls)
        if assigned:
            msg = (
                f"{cls.__name__}.{assigned[0]} is assigned. A problem's phases are annotated "
                f"with their shape, not assigned: write '{assigned[0]}: <a yapss.Phase "
                f"subclass>'."
            )
            raise TypeError(msg)
        declared: dict[str, type[AnyPhase]] = {}
        for name, value in own_annotations(cls, _declaring_scope()).items():
            if name.startswith("_"):
                continue
            if not (is_subclass(value, Phase) and value is not Phase):
                msg = (
                    f"{cls.__name__}.{name} is annotated {_named(value)}, which is not a "
                    f"phase's shape. Each phase is annotated with a subclass of yapss.Phase, "
                    f"such as 'class {name.title().replace('_', '')}(yapss.Phase)'."
                )
                raise TypeError(msg)
            declared[name] = value
        # A class that declares no phases is allowed: zero is a count, and nothing about the
        # transcription changes shape there. What it states is a problem in the parameters and
        # the discrete constraints alone -- an ordinary nonlinear program, which is how a
        # problem like hs071 is written.
        cls._declared = declared

    def __init__(self) -> None:
        """Build each phase, an instance of its shape. Called by `Problem`."""
        handles: dict[str, AnyPhase] = {
            name: shape(name, index)
            for index, (name, shape) in enumerate(type(self)._declared.items())
        }
        object.__setattr__(self, "_handles", handles)

    # Hidden from type checkers for the reason `Container`'s are. A subclass annotates its
    # phases, so a type checker reads the phases from the declaration and reports a name that
    # was never declared. At runtime the annotations are not attributes, and are answered here.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Return the phase of that name."""
            if name.startswith("_"):
                raise AttributeError(name)
            handles = object.__getattribute__(self, "_handles")
            if name not in handles:
                msg = (
                    f"{type(self).__name__} has no phase '{name}'."
                    f"{suggest(name, tuple(handles))}"
                )
                raise AttributeError(msg)
            return handles[name]

        def __setattr__(self, name, value):
            """Refuse every assignment: phases are declared, not assigned."""
            del value
            msg = f"{type(self).__name__}.{name} cannot be assigned; phases are declared"
            raise AttributeError(msg)

        def __delattr__(self, name):
            """Refuse every deletion: phases are declared, not assigned."""
            msg = f"{type(self).__name__}.{name} cannot be deleted; phases are declared"
            raise AttributeError(msg)

    def _all(self) -> dict[str, AnyPhase]:
        handles: dict[str, AnyPhase] = object.__getattribute__(self, "_handles")
        return handles

    def _example(self) -> str:
        """Return a setting written the way a phase's are, for a message.

        A phase is not a field of `phases`: it is reached by name and set up through its own
        fields, so the example has to go two levels deeper than a vector's would.
        """
        for phase in self:
            declaration = type(phase)._declaration
            if declaration is None:  # pragma: no cover - a phase always has its shape
                break
            field = declaration.state._fields[0] if declaration.state._fields else "<field>"
            return f"{phase.name}.state.{field}.bounds"
        return "<phase>.state.<field>.bounds"

    def __iter__(self) -> Iterator[AnyPhase]:
        """Iterate over the phases, in declaration order."""
        return iter(self._all().values())

    def __len__(self) -> int:
        """Return the number of phases."""
        return len(self._all())

    def __getitem__(self, index: int) -> AnyPhase:
        """Return the phase at `index`, in declaration order."""
        return list(self._all().values())[index]


# -- aspects ---------------------------------------------------------------------------------


class StateAspects(Container):
    """The state of one phase: its bounds, its endpoint bounds, its guess, and its scales.

    There are two scales. ``scale`` says how large the state itself typically is; while
    ``defect_scale`` says how large the collocation defect is -- the residual of the dynamics,
    which is a constraint rather than a variable and can be of a quite different size.

    Every aspect is an instance of the declaration itself, which is what makes the fields
    reachable by the names they were declared with. It is read field first, through
    `fields.Fields`, and a type checker never sees it: `ph.state` is typed as the declaration.
    """

    _held = ("bounds", "initial", "final", "guess", "scale", "defect_scale")


class ControlAspects(Container):
    """The control of one phase: its bounds, its guess, and its scale."""

    _held = ("bounds", "guess", "scale")


class PathAspects(Container):
    """The path constraints of one phase: their bounds and their scales."""

    _held = ("bounds", "scale")


class IntegralAspects(Container):
    """The integrals of one phase: their bounds, their guesses, and their scales."""

    _held = ("bounds", "guess", "scale")


class Independent(Container):
    """A phase's independent variable: its initial and final values, and the guess for them.

    Every phase has one, called ``time`` unless the phase names it otherwise, which it does
    with an annotation of this type::

        class Nose(yapss.Phase):
            state: Body
            r: yapss.Independent

    ``initial`` and ``final`` are bounds, written the same way as any other bound. ``guess`` is
    a ``(t0, tf)`` pair, and is the only source of the phase's guessed duration, so a sampled
    guess carries no times of its own.
    """

    _settable = ("initial", "final", "guess", "scale")

    if TYPE_CHECKING:
        initial: Any
        final: Any
        guess: tuple[float, float] | None
        scale: float

    def __init__(self, label: str) -> None:
        self._label = label
        self._hold("initial", Bounds.default)
        self._hold("final", Bounds.default)
        self._hold("guess", None)
        self._hold("scale", 1.0)

    def _check(self, name: str, value: Any) -> Any:
        if name == "guess":
            return self._check_guess(value)
        if name == "scale":
            return Scale.check(value, label=self._label, name=name, npoints=None)
        return Bounds.check(value, label=self._label, name=name, npoints=None)

    def _check_guess(self, value: Any) -> tuple[float, float]:
        """Validate the phase's guessed extent, which is a pair of plain numbers.

        Not a bound: `t0` and `tf` are where the phase is guessed to start and end, so each
        side is one number rather than an interval of its own.
        """
        if not is_pair(value):
            msg = f"{self._label} guess is a (t0, tf) pair; got {value!r}"
            raise TypeError(msg)
        for side in value:
            if is_bool(side) or not is_real(side):
                msg = f"{self._label} guess: t0 and tf are numbers; got {side!r}"
                raise TypeError(msg)
        t0, tf = (float(side) for side in value)
        if not (math.isfinite(t0) and math.isfinite(tf)):
            msg = f"{self._label} guess: t0 and tf must be finite; got ({t0}, {tf})"
            raise ValueError(msg)
        if not t0 < tf:
            msg = f"{self._label} guess: t0 {t0} is not less than tf {tf}"
            raise ValueError(msg)
        return (t0, tf)


class PhaseRegistry(Registry):
    """A phase's callbacks. Reached as ``ph.register``."""

    _registrations = ("continuous",)

    def __init__(self, phase: AnyPhase) -> None:
        self._phase = phase
        self._label = f"{phase._label} callbacks"

    def _register(self, which: str, function: Callable[..., Any] | None) -> Any:
        # See `Problem._register`: registering is setting a value, and the last one wins.
        phase = self._phase
        attribute = f"_{which}"

        def register(callback: Callable[..., Any]) -> Callable[..., Any]:
            if not is_callable(callback):
                msg = f"{phase._label} {which} callback must be callable; got {callback!r}"
                raise TypeError(msg)
            object.__setattr__(phase, attribute, callback)
            return callback

        return register if function is None else register(function)

    @overload
    def continuous(self, function: FillerT, /) -> FillerT: ...
    @overload
    def continuous(self, function: None = None, /) -> Callable[[FillerT], FillerT]: ...
    def continuous(self, function: Callable[..., Any] | None = None, /) -> Any:
        """Register the phase's continuous callback, as a decorator or as a call.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator. Registering a second one
            replaces the first, as setting any other value twice does.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("continuous", function)


class Phase(HasRegistry, Generic[S_co, C_co, P_co, I_co]):
    """A phase's shape: the vectors it is built from. Subclass it and annotate them.

    ``state`` is required; ``control``, ``path`` and ``integral`` are optional, and a phase
    that omits one has none of it. One further annotation, of type `Independent`, names the
    phase's independent variable, which is otherwise ``time``::

        class Slide(yapss.Phase):
            state: Position
            control: Angle

    The annotations are also what a type checker reads: ``ph.state`` is a ``Position``, and its
    fields are checked from there. They override annotations of this base class that bound
    each by its role, so ``state: Angle`` is reported before anything runs, as well as refused
    when it does.

    A shape is not a phase. The phases are named in a `Phases` class, and each is an instance
    of its shape, so two phases may share one.
    """

    # `register` and the independent variable's name are added per instance, since the
    # latter is whatever the phase called it
    _held = ("state", "control", "path", "integral")
    _settable = ("mesh",)
    _declaration: PhaseDeclaration | None = None

    if TYPE_CHECKING:
        # Typed as the declarations themselves, which is what makes a setting reachable field
        # first -- `ph.state.x.bounds` -- and checkable: `State.x` is the marker `scalar()` or
        # `vector(n)` returned, and the marker's type says which settings that rank takes. At
        # runtime each is a `fields.Fields`, forwarding to the aspect containers below.
        state: S_co
        control: C_co
        path: P_co
        integral: I_co
        mesh: Mesh
        register: PhaseRegistry

        # No `time` here: every shape annotates its own independent variable, so the name a
        # checker sees is the name that phase has. Declaring `time` on the base would make
        # `ph.time` resolve on a phase that runs over a radius, and fail only at runtime.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Read the shape from the subclass's annotations, refusing what is not one."""
        super().__init_subclass__(**kwargs)
        owner = cls.__name__
        for base in cls.__mro__[1:]:
            if base is not Phase and issubclass(base, Phase) and base._declaration is not None:
                msg = (
                    f"{owner} cannot inherit from {base.__name__}, which is already a phase's "
                    f"shape. Subclass yapss.Phase and annotate the vectors again."
                )
                raise TypeError(msg)
        assigned = _assigned(cls)
        if assigned:
            msg = (
                f"{owner}.{assigned[0]} is assigned. A phase's shape is annotated, not "
                f"assigned: write '{assigned[0]}: <a class>'."
            )
            raise TypeError(msg)
        roles: dict[str, type[Vector]] = {}
        independent: list[str] = []
        for name, value in own_annotations(cls, _declaring_scope()).items():
            if name.startswith("_"):
                continue
            if name in _ROLES:
                roles[name] = declared_role(value, owner, name, _ROLES, annotation=True)
            elif value is Independent:
                independent.append(name)
            else:
                raise TypeError(_not_a_slot(owner, name, value))
        if "state" not in roles:
            msg = f"{owner} declares no state. Every phase has one: write 'state: <a yapss.State>'."
            raise TypeError(msg)
        if len(independent) > 1:
            names = ", ".join(repr(name) for name in independent)
            msg = f"{owner}: a phase has one independent variable, but {names} are annotated so"
            raise TypeError(msg)
        if not independent:
            # A phase always has an independent variable, so nothing is saved by defaulting
            # its name: the choice between 'time', 't' and 's' is the user's problem's, not
            # YAPSS's preference. Defaulting it also cost a type-checker hole -- `time` had to
            # be declared on this class for `ph.time` to resolve, which made `ph.time` check
            # on a phase that runs over a radius and fail only at runtime.
            msg = (
                f"The class declaration for {owner} does not name its independent variable. "
                f"Write 'time: yapss.Independent' if you want to name the independent "
                f"variable 'time'."
            )
            raise TypeError(msg)
        name = independent[0]
        if name in _RESERVED:
            msg = (
                f"{owner}.{name} names the independent variable, but '{name}' is already a "
                f"phase's own attribute. Name it something else."
            )
            raise TypeError(msg)
        state = roles["state"]
        control = roles.get("control", Control)
        _check_namespace(owner, state, control, name)
        cls._declaration = PhaseDeclaration(
            state=state,
            control=control,
            path=roles.get("path", Path),
            integral=roles.get("integral", Integral),
            independent=name,
        )

    def __init__(self, name: str, index: int) -> None:
        declaration = type(self)._declaration
        if declaration is None:
            msg = (
                "yapss.Phase is a phase's shape to subclass, not a phase: declare "
                "'class Slide(yapss.Phase)', annotate its vectors, and name the phases in a "
                "yapss.Phases."
            )
            raise TypeError(msg)
        self._name = name
        self._index = index
        self._continuous: Callable[..., Any] | None = None
        self._independent = declaration.independent
        self._label = f"phase '{name}'"
        self._hold("mesh", Mesh.uniform())

        state = StateAspects()
        state._label = f"{self._label} state"
        for aspect, kind in (
            ("bounds", Bounds),
            ("initial", Bounds),
            ("final", Bounds),
            ("guess", Guess),
            ("scale", Scale),
            ("defect_scale", Scale),
        ):
            state._hold(
                aspect, declaration.state._new(kind, f"{state._label} {aspect}", aspect=aspect)
            )
        self._hold("state", Fields(state, declaration.state, state._label))

        control = ControlAspects()
        control._label = f"{self._label} control"
        control._hold(
            "bounds", declaration.control._new(Bounds, f"{control._label} bounds", aspect="bounds")
        )
        control._hold(
            "guess", declaration.control._new(Guess, f"{control._label} guess", aspect="guess")
        )
        control._hold(
            "scale", declaration.control._new(Scale, f"{control._label} scale", aspect="scale")
        )
        self._hold("control", Fields(control, declaration.control, control._label))

        path = PathAspects()
        path._label = f"{self._label} path"
        path._hold(
            "bounds", declaration.path._new(Bounds, f"{path._label} bounds", aspect="bounds")
        )
        path._hold("scale", declaration.path._new(Scale, f"{path._label} scale", aspect="scale"))
        self._hold("path", Fields(path, declaration.path, path._label))

        integral = IntegralAspects()
        integral._label = f"{self._label} integral"
        integral._hold(
            "bounds",
            declaration.integral._new(Bounds, f"{integral._label} bounds", aspect="bounds"),
        )
        integral._hold(
            "guess",
            declaration.integral._new(ScalarGuess, f"{integral._label} guess", aspect="guess"),
        )
        integral._hold(
            "scale", declaration.integral._new(Scale, f"{integral._label} scale", aspect="scale")
        )
        self._hold("integral", Fields(integral, declaration.integral, integral._label))

        name = declaration.independent
        self._hold(name, Independent(f"{self._label} {name}"))
        self._hold("register", PhaseRegistry(self))
        object.__setattr__(self, "_held", (*Phase._held, name, "register"))

    @property
    def name(self) -> str:
        """str: The name the phase was declared with."""
        return self._name

    @property
    def index(self) -> int:
        """int: The position of the phase in declaration order."""
        return self._index

    def _check(self, name: str, value: Any) -> Any:
        del name
        if not isinstance(value, Mesh):
            msg = (
                f"{self._label} mesh must be a Mesh, for example "
                f"'yapss.Mesh.uniform(segments=10, points=10)'; got {value!r}"
            )
            raise TypeError(msg)
        return value

    def __repr__(self) -> str:
        """Return a short representation naming the phase and its shape."""
        return f"<{type(self).__name__} {self._name!r} index={self._index}>"
