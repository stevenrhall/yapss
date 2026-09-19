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
from typing import TYPE_CHECKING, Any, cast

from .containers import Container, HasRegistry, Registry, is_callable, is_subclass, suggest
from .kinds import Bounds, Guess, ScalarGuess, Scale
from .mesh import Mesh
from .vector import Empty, Field, Vector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["Phase", "Phases", "phase"]


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
    """The metadata of that variable: its units, its LaTeX, its one-line description."""


def _vector_class(value: object, argument: str) -> type[Vector]:
    if not is_subclass(value, Vector):
        msg = (
            f"phase({argument}=) takes a vector class declared with 'class X(yapss.Vector)'; "
            f"got {value!r}"
        )
        raise TypeError(msg)
    return cast("type[Vector]", value)


DEFAULT_INDEPENDENT = "time"
"""What a phase's independent variable is called when the phase does not name it."""

_ROLES = ("state", "control", "path", "integral")


def phase(
    *,
    state: type[Vector],
    control: type[Vector] = Empty,
    path: type[Vector] = Empty,
    integral: type[Vector] = Empty,
    **independent: Any,
) -> Any:
    """Declare one phase of a problem.

    Parameters
    ----------
    state : type[Vector]
        The class naming the phase's states.
    control : type[Vector], optional
        The class naming the phase's controls.
    path : type[Vector], optional
        The class naming the phase's path constraints.
    integral : type[Vector], optional
        The class naming the phase's integrals.
    **independent : Field
        One further keyword names the phase's independent variable, which is otherwise
        ``time``. The keyword is the name, as it is in a vector's class body, and its value is
        a `field`: ``yapss.phase(state=Nose, r=yapss.field(latex="r"))``.

    Returns
    -------
    Any
        A marker recording the declaration. YAPSS replaces it with a `Phase` when the problem
        is built, so it is never seen again.
    """
    name, marker = _independent(independent)
    _check_namespace(_vector_class(state, "state"), _vector_class(control, "control"), name)
    return PhaseDeclaration(
        state=_vector_class(state, "state"),
        control=_vector_class(control, "control"),
        path=_vector_class(path, "path"),
        integral=_vector_class(integral, "integral"),
        independent=name,
        independent_field=marker,
    )


def _check_namespace(state: type[Vector], control: type[Vector], independent: str) -> None:
    """Refuse a phase whose states, controls and independent variable share a name.

    Those three are one namespace, because that is what they are: the columns of the phase's
    Jacobian, which a derivative names without saying which vector it came from (spec 5.7). A
    name belonging to two of them would name two columns.

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
                f"'phase(..., <name>=yapss.field(...))'."
            )
            raise ValueError(msg)


def _independent(given: dict[str, Any]) -> tuple[str, Field]:
    """Return the name and the field of the phase's independent variable.

    A keyword `phase` does not know is the independent variable's name -- which is what makes
    the name arrive the way a field's name always does, from where it is bound. The value must
    be a `field`, and that is what keeps a misspelled role a misspelled role: `contrl=Thrust`
    passes a vector class, is not a field, and is refused with the suggestion.
    """
    if not given:
        return DEFAULT_INDEPENDENT, Field(units="", latex="", doc="", size=None)
    for name, value in given.items():
        if not isinstance(value, Field):
            msg = (
                # a vector class under an unknown keyword is a misspelled role, not an
                # independent variable, and that is the message it should get
                f"phase() got an unexpected keyword '{name}'.{suggest(name, _ROLES)}"
                if is_subclass(value, Vector)
                else (
                    f"phase({name}=) names the phase's independent variable, so it takes a "
                    f"field: '{name}=yapss.field(...)'; got {value!r}."
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
            f"phase({name}=) names the independent variable, which is one value, so its field "
            f"takes no size; got size={marker.size}"
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
        # problem like hs071 is written. See spec 1.1 and 3.
        cls._declared = declared

    def __init__(self) -> None:
        """Build one `Phase` per declared name. Called by `Problem`."""
        handles = {
            name: Phase(name, index, declaration)
            for index, (name, declaration) in enumerate(type(self)._declared.items())
        }
        object.__setattr__(self, "_handles", handles)

    def __getattr__(self, name: str) -> Phase:
        """Return the phase of that name."""
        if name.startswith("_"):
            raise AttributeError(name)
        handles: dict[str, Phase] = object.__getattribute__(self, "_handles")
        if name not in handles:
            msg = f"{type(self).__name__} has no phase '{name}'." f"{suggest(name, tuple(handles))}"
            raise AttributeError(msg)
        return handles[name]

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: phases are declared, not assigned."""
        del value
        msg = f"{type(self).__name__}.{name} cannot be assigned; phases are declared"
        raise AttributeError(msg)

    def _all(self) -> dict[str, Phase]:
        handles: dict[str, Phase] = object.__getattribute__(self, "_handles")
        return handles

    def __iter__(self) -> Iterator[Phase]:
        """Iterate over the phases, in declaration order."""
        return iter(self._all().values())

    def __len__(self) -> int:
        """Return the number of phases."""
        return len(self._all())

    def __getitem__(self, index: int) -> Phase:
        """Return the phase at `index`, in declaration order."""
        return list(self._all().values())[index]


# -- aspects ---------------------------------------------------------------------------------


class StateAspects(Container):
    """The state of one phase: its bounds, its endpoint bounds, its guess, and its scales.

    There are two scales. ``scale`` says how large the state itself typically is; while
    ``defect_scale`` says how large the collocation defect is -- the residual of the dynamics,
    which is a constraint rather than a variable and can be of a quite different size.
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
        if not isinstance(value, tuple) or len(value) != 2:  # noqa: PLR2004
            msg = f"{self._label} guess is a (t0, tf) pair; got {value!r}"
            raise TypeError(msg)
        t0, tf = (Bounds.check(v, label=self._label, name="guess", npoints=None)[0] for v in value)
        if not t0 < tf:
            msg = f"{self._label} guess: t0 {t0} is not less than tf {tf}"
            raise ValueError(msg)
        return (t0, tf)


class PhaseRegistry(Registry):
    """A phase's callbacks. Reached as ``ph.register``."""

    _registrations = ("continuous", "continuous_jacobian", "continuous_hessian")

    def __init__(self, phase: Phase) -> None:
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


class Phase(HasRegistry):
    """One phase of a problem: its aspects, its mesh, and its continuous callback."""

    # `register` and the independent variable's name are added per instance, since the
    # latter is whatever the phase called it
    _held = ("state", "control", "path", "integral")
    _settable = ("mesh",)

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
            state._hold(aspect, declaration.state._new(kind, f"{state._label} {aspect}"))
        self._hold("state", state)

        control = ControlAspects()
        control._label = f"{self._label} control"
        control._hold("bounds", declaration.control._new(Bounds, f"{control._label} bounds"))
        control._hold("guess", declaration.control._new(Guess, f"{control._label} guess"))
        control._hold("scale", declaration.control._new(Scale, f"{control._label} scale"))
        self._hold("control", control)

        path = PathAspects()
        path._label = f"{self._label} path"
        path._hold("bounds", declaration.path._new(Bounds, f"{path._label} bounds"))
        path._hold("scale", declaration.path._new(Scale, f"{path._label} scale"))
        self._hold("path", path)

        integral = IntegralAspects()
        integral._label = f"{self._label} integral"
        integral._hold("bounds", declaration.integral._new(Bounds, f"{integral._label} bounds"))
        integral._hold("guess", declaration.integral._new(ScalarGuess, f"{integral._label} guess"))
        integral._hold("scale", declaration.integral._new(Scale, f"{integral._label} scale"))
        self._hold("integral", integral)

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
