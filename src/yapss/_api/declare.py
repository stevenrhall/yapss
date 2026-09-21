"""

Phase declarations and the setup aspects reached from a phase.

A phase is declared by naming the vector classes it uses::

    class Phases(yapss.Phases):
        boost = yapss.phase(state=Rocket, control=Thrust)
        singular = yapss.phase(state=Rocket, control=Thrust, path=SingularArc)

The declaration fixes what every aspect of that phase contains, so ``ph.state.bounds``,
``ph.state.guess``, and the ``dynamics`` a callback fills all carry the same field names.

"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, cast

# `TypeVar` from typing_extensions, not typing: PEP 696 defaults are native only from
# Python 3.13, and the floor is 3.11. The defaults are what let `yapss.Phase[Slide, Angle]`
# be written without naming the two declarations a phase usually does not have.
from typing_extensions import TypeVar

from .containers import Container, HasRegistry, Registry, is_callable, is_subclass, suggest
from .fields import Fields
from .kinds import Bounds, Guess, ScalarGuess, Scale, is_bool, is_pair, is_real
from .mesh import Mesh
from .vector import Control, Field, Integral, Path, State, Vector, role_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["AnyPhase", "Phase", "Phases", "phase"]

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
    """What `phase` records. Replaced by a `Phase` when the problem is built."""

    state: type[Vector]
    control: type[Vector]
    path: type[Vector]
    integral: type[Vector]
    independent: str
    """What the phase calls its independent variable, which is `time` unless it named it."""
    independent_field: Field
    """The field that named it, which records its size and nothing else."""


def declared_role(
    value: object, call: str, argument: str, role: type[Vector], keywords: tuple[str, ...]
) -> type[Vector]:
    """Return `value` if it is a declaration of `role`, or refuse it naming what it is.

    The role is read from the class hierarchy, so a state handed over as a control is refused
    here, at the line that made the mistake -- which, without the check, would build and solve a
    problem mislabelled throughout, and never raise at all.

    The likeliest way to get this wrong is to swap two arguments, so when the class belongs to
    another keyword of the same call the message names that keyword. Changing the class's base
    would also silence the error, and would be the wrong fix.

    Parameters
    ----------
    value : object
        What was passed.
    call : str
        The call it was passed to, for the message, such as ``"phase"``.
    argument : str
        The keyword it was passed as.
    role : type[Vector]
        The role that keyword takes.
    keywords : tuple of str
        Every role keyword of the same call, so a swapped argument can be named.
    """
    base = f"yapss.{role.__name__}"
    if not is_subclass(value, Vector):
        msg = (
            f"{call}({argument}=) takes a subclass of {base}, such as "
            f"'class X({base})'; got {value!r}"
        )
        raise TypeError(msg)
    cls = cast("type[Vector]", value)
    actual = role_of(cls)
    if actual == role._role:
        return cls
    if actual is None:
        msg = (
            f"{call}({argument}=) takes a subclass of {base}, but {cls.__name__} has no role. "
            f"Declare it as 'class {cls.__name__}({base})'."
        )
    else:
        msg = f"{call}({argument}=) takes a subclass of {base}, but {cls.__name__} subclasses "
        msg += f"yapss.{actual.title()}."
        if actual in keywords:
            msg += f" Did you mean {actual}={cls.__name__}?"
    raise TypeError(msg)


DEFAULT_INDEPENDENT = "time"
"""What a phase's independent variable is called when the phase does not name it."""

_ROLES = ("state", "control", "path", "integral")


# A type checker solves a type parameter from an argument that was passed, never from the
# default of one that was not, so an omitted `control`, `path` or `integral` resolves to the
# type parameter's own default -- the role, which declares no fields -- rather than leaving the
# whole declaration untyped.
def phase(
    *,
    state: type[S_co],
    # The role itself is the declared default of each of these parameters, so an omitted
    # argument resolves to it rather than leaving the phase untyped. A type checker still
    # measures the default *value* against `type[C_co]` for an arbitrary `C`, which it cannot
    # satisfy, hence the three waivers.
    control: type[C_co] = Control,  # type: ignore[assignment]
    path: type[P_co] = Path,  # type: ignore[assignment]
    integral: type[I_co] = Integral,  # type: ignore[assignment]
    **independent: Any,
) -> Phase[S_co, C_co, P_co, I_co]:
    """Declare one phase of a problem.

    Parameters
    ----------
    state : type[State]
        The class naming the phase's states.
    control : type[Control], optional
        The class naming the phase's controls; none, if omitted.
    path : type[Path], optional
        The class naming the phase's path constraints; none, if omitted.
    integral : type[Integral], optional
        The class naming the phase's integrals; none, if omitted.
    **independent : Field
        One further keyword names the phase's independent variable, which is otherwise
        ``time``. The keyword is the name, as it is in a vector's class body, and its value is
        a scalar: ``yapss.phase(state=Nose, r=yapss.scalar())``.

    Returns
    -------
    Phase
        A marker recording the declaration. YAPSS replaces it with the `Phase` of that name
        when the problem is built, so the marker itself is never seen again. It is *typed* as
        the `Phase` it becomes, which is what lets a type checker follow a declared phase from
        ``problem.phases.<name>`` down to the fields of its state and control.
    """
    name, marker = _independent(independent)
    _check_namespace(
        declared_role(state, "phase", "state", State, _ROLES),
        declared_role(control, "phase", "control", Control, _ROLES),
        name,
    )
    return cast(
        "Phase[S_co, C_co, P_co, I_co]",
        PhaseDeclaration(
            state=declared_role(state, "phase", "state", State, _ROLES),
            control=declared_role(control, "phase", "control", Control, _ROLES),
            path=declared_role(path, "phase", "path", Path, _ROLES),
            integral=declared_role(integral, "phase", "integral", Integral, _ROLES),
            independent=name,
            independent_field=marker,
        ),
    )


def _check_namespace(state: type[Vector], control: type[Vector], independent: str) -> None:
    """Refuse a phase whose states, controls and independent variable share a name.

    Those three are one namespace, because that is what they are: the columns of the phase's
    Jacobian, which a derivative names without saying which vector it came from. A name
    belonging to two of them would name two columns.

    The check is here, at the call that brought the classes together, because that is where the
    collision was made -- and the message names the two classes rather than the phase, since
    renaming a member of one of them is the fix. The parameters are not here to be checked;
    they arrive as an argument to `Problem`, which checks them against this namespace there.

    Path and integral names are not in it. They are outputs, so they appear on the other side
    of a derivative and may collide with a variable freely.
    """
    shared = sorted(set(state._fields) & set(control._fields))
    if shared:
        msg = (
            f"phase(state={state.__name__}, control={control.__name__}): both declare "
            f"{shared[0]!r}. A phase's states, controls and independent variable are one "
            f"namespace, so their names must differ; rename it in one of the two classes."
        )
        raise ValueError(msg)
    for role, declaration in (("state", state), ("control", control)):
        if independent in declaration._fields:
            whose = (
                "the phase's independent variable, which you named"
                if independent != DEFAULT_INDEPENDENT
                else "the phase's independent variable, which is called 'time' by default"
            )
            msg = (
                f"phase({role}={declaration.__name__}): {declaration.__name__} declares "
                f"{independent!r} as a {role}, and that is also {whose}. They are one "
                f"namespace, so their names must differ; rename the {role}, or name the "
                f"independent variable something else with "
                f"'phase(..., <name>=yapss.scalar())'."
            )
            raise ValueError(msg)


def _independent(given: dict[str, Any]) -> tuple[str, Field]:
    """Return the name and the field of the phase's independent variable.

    A keyword `phase` does not know is the independent variable's name -- which is what makes
    the name arrive the way a field's name always does, from where it is bound. The value must
    be a `scalar()`, and that is what keeps a misspelled role a misspelled role: `contrl=Thrust`
    passes a vector class, is not a field, and is refused with the suggestion.
    """
    if not given:
        return DEFAULT_INDEPENDENT, Field()
    for name, value in given.items():
        if not isinstance(value, Field):
            msg = (
                # a vector class under an unknown keyword is a misspelled role, not an
                # independent variable, and that is the message it should get
                f"phase() got an unexpected keyword '{name}'.{suggest(name, _ROLES)}"
                if is_subclass(value, Vector)
                else (
                    f"phase({name}=) names the phase's independent variable, so it takes a "
                    f"scalar: '{name}=yapss.scalar()'; got {value!r}."
                    f"{suggest(name, _ROLES)}"
                )
            )
            raise TypeError(msg)
    if len(given) > 1:
        names = ", ".join(repr(name) for name in given)
        msg = f"a phase has one independent variable, but {names} were given as fields"
        raise TypeError(msg)
    name, marker = next(iter(given.items()))
    if marker.size is not None:
        msg = (
            f"phase({name}=) names the independent variable, which is one value: write "
            f"'{name}=yapss.scalar()', not 'yapss.vector({marker.size})'"
        )
        raise TypeError(msg)
    return name, marker


class Phases:
    """Base of a problem's phase declaration. Subclass it and name the phases.

    The subclass is passed to `Problem`, which instantiates it; each name then gives the
    `Phase` handle used to set that phase up and to reach it in callbacks and solutions.
    """

    _declared: dict[str, PhaseDeclaration] = {}  # noqa: RUF012

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Collect the declared phases, in declaration order."""
        super().__init_subclass__(**kwargs)
        for base in cls.__bases__:
            if base is not Phases and issubclass(base, Phases) and base._declared:
                msg = f"{cls.__name__} cannot inherit from the phase declaration {base.__name__}"
                raise TypeError(msg)
        annotated = [n for n in inspect.get_annotations(cls) if not n.startswith("_")]
        if annotated:
            msg = (
                f"{cls.__name__}.{annotated[0]} is annotated. Phases are declared without "
                f"annotations: write '{annotated[0]} = yapss.phase(state=...)'."
            )
            raise TypeError(msg)
        declared: dict[str, PhaseDeclaration] = {}
        for name, value in list(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            if not isinstance(value, PhaseDeclaration):
                msg = (
                    f"{cls.__name__}.{name} is not a phase. A phase declaration holds only "
                    f"phases; write '{name} = yapss.phase(state=...)'."
                )
                raise TypeError(msg)
            declared[name] = value
        for name in declared:
            delattr(cls, name)
        # A class that declares no phases is allowed: zero is a count, and nothing about the
        # transcription changes shape there. What it states is a problem in the parameters and
        # the discrete constraints alone -- an ordinary nonlinear program, which is how a
        # problem like hs071 is written.
        cls._declared = declared

    def __init__(self) -> None:
        """Build one `Phase` per declared name. Called by `Problem`."""
        handles: dict[str, AnyPhase] = {
            name: Phase(name, index, declaration)
            for index, (name, declaration) in enumerate(type(self)._declared.items())
        }
        object.__setattr__(self, "_handles", handles)

    if TYPE_CHECKING:
        # A declaration reached through the base class -- as it is from the bare
        # `yapss.Problem`, whose phase parameter defaults to `Phases` -- cannot say which
        # phases were declared, so it answers with a phase of unknown declarations rather than
        # refusing every name. A subclass's own phases are declared attributes and resolve
        # ahead of this, so nothing is lost where the declaration is known; a misspelled phase
        # name is caught at runtime, with a suggestion.
        def __getattr__(self, name: str) -> AnyPhase: ...

    # Hidden from type checkers for the reason `Container`'s are. A subclass names its phases
    # as ordinary class attributes, and `phase()` is typed as the `Phase` each one becomes, so
    # with these hidden a type checker reads the declaration and reports a phase name that was
    # never declared. At runtime the attributes are deleted from the class and answered here.
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

    def _all(self) -> dict[str, AnyPhase]:
        handles: dict[str, AnyPhase] = object.__getattribute__(self, "_handles")
        return handles

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


class TimeAspects(Container):
    """The initial and final time of one phase, and the guess for them.

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
        if not t0 < tf:
            msg = f"{self._label} guess: t0 {t0} is not less than tf {tf}"
            raise ValueError(msg)
        return (t0, tf)


class PhaseRegistry(Registry):
    """A phase's callbacks. Reached as ``ph.register``."""

    _registrations = ("continuous", "continuous_jacobian", "continuous_hessian")

    def __init__(self, phase: AnyPhase) -> None:
        self._phase = phase
        self._label = f"{phase._label} callbacks"

    def _register(self, which: str, function: Callable[..., Any] | None, *, replace: bool) -> Any:
        phase = self._phase
        attribute = f"_{which}"

        def register(callback: Callable[..., Any]) -> Callable[..., Any]:
            if not is_callable(callback):
                msg = f"{phase._label} {which} callback must be callable; got {callback!r}"
                raise TypeError(msg)
            current = getattr(phase, attribute)
            if current is not None and not replace:
                existing = getattr(current, "__qualname__", repr(current))
                msg = (
                    f"{phase._label} already has the {which} callback '{existing}'; pass "
                    f"replace=True to replace it"
                )
                raise ValueError(msg)
            object.__setattr__(phase, attribute, callback)
            return callback

        return register if function is None else register(function)

    def continuous(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the phase's continuous callback, as a decorator or as a call.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator, as in
            ``@ph.register.continuous(replace=True)``.
        replace : bool, default False
            Replace a callback already registered on this phase. Registering a second callback
            without it is refused, since it is nearly always a mistake.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("continuous", function, replace=replace)

    def continuous_jacobian(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the phase's continuous Jacobian, as a decorator or as a call.

        Required under ``derivatives.method = "user"``. The callback takes the same argument
        the continuous callback does and a `jacobian` to fill, whose entries are named for
        the things they relate: ``jacobian.dynamics.x.v`` is the derivative of the dynamics
        of ``x`` with respect to ``v``. What it writes is the sparsity structure, so a name
        it does not write is a derivative that is zero everywhere, a derivative that is zero
        only at this point is written as ``0.0``, and the same names must be written on every
        call.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator.
        replace : bool, default False
            Replace a callback already registered on this phase.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("continuous_jacobian", function, replace=replace)

    def continuous_hessian(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the phase's continuous Hessian, as a decorator or as a call.

        Required under ``derivatives.method = "user"`` at ``derivatives.order = "second"``,
        including when every entry of it is zero. ``hessian.dynamics.x.v.u`` chains one more
        name than the Jacobian does; the two orders name one derivative, so each unordered
        pair is written once and writing both is refused rather than summed.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator.
        replace : bool, default False
            Replace a callback already registered on this phase.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("continuous_hessian", function, replace=replace)


class Phase(HasRegistry, Generic[S_co, C_co, P_co, I_co]):
    """One phase of a problem: its aspects, its mesh, and its continuous callback.

    The four parameters are the classes the phase was declared with, so a type checker
    following ``problem.phases.<name>.state.bounds`` arrives at the state declaration itself
    and can say whether a field of that name was declared.
    """

    # `register` and the independent variable's name are added per instance, since the
    # latter is whatever the phase called it
    _held = ("state", "control", "path", "integral")
    _settable = ("mesh",)

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

        # A phase's independent variable is called whatever the phase called it, so it is the
        # one held name that is not known until the declaration is read. This reader is
        # deliberately left visible to type checkers: without it `ph.r` would be an error on a
        # phase that runs over a radius, and a false positive is worse than the misspelling it
        # would otherwise catch. `time` is declared above it so that the usual spelling still
        # resolves to something better than `Any`.
        time: TimeAspects

        def __getattr__(self, name: str) -> Any: ...

    def __init__(self, name: str, index: int, declaration: PhaseDeclaration) -> None:
        self._name = name
        self._index = index
        self._declaration = declaration
        self._continuous: Callable[..., Any] | None = None
        self._continuous_jacobian: Callable[..., Any] | None = None
        self._continuous_hessian: Callable[..., Any] | None = None
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
        self._hold(name, TimeAspects(f"{self._label} {name}"))
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
        """Return a short representation naming the phase and its vector classes."""
        declaration = self._declaration
        return (
            f"<Phase {self._name!r} index={self._index} "
            f"state={declaration.state.__name__} control={declaration.control.__name__}>"
        )
