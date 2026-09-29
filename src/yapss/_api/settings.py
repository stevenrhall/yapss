"""

A problem's setup as it stood when it was solved: ``solution.settings``.

The settings are exactly the problem's settable names, spelled as the problem spells them, and
holding values only, so that a solution pickles. They are built by walking the problem's own
declared names -- each container's held and settable names, each vector's fields and their
settings -- rather than from a list, so a setting added to the problem appears here without
further work. The one renaming is ``register``, where the problem registers its callbacks: the
settings record what was registered, as ``callbacks``, and a callback by its identity only.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yapss._backend.ipopt_options import IpoptOptions

from .containers import Container, Registry, suggest
from .declare import Phases
from .fields import Fields, aspects_of
from .vector import BlockRows

if TYPE_CHECKING:
    from collections.abc import Callable

    from .problem import Problem

__all__ = ["Callback", "Settings", "SettingsGroup", "settings_of"]

_NAMES_FIXED = "a solution's names are fixed, and its arrays can be edited in place"


class Callback:
    """A registered callback, recorded by its identity: where it was defined, and what it says.

    Attributes
    ----------
    module : str or None
        The module the callback was defined in, such as ``"yapss.examples.brachistochrone"``.
    qualname : str
        Its qualified name, such as ``"setup.<locals>.continuous"``.
    doc : str or None
        Its whole docstring, or None if it has none.
    """

    __slots__ = ("doc", "module", "qualname")

    if TYPE_CHECKING:
        module: str | None
        qualname: str
        doc: str | None

    def __init__(self, module: str | None, qualname: str, doc: str | None) -> None:
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "qualname", qualname)
        object.__setattr__(self, "doc", doc)

    @classmethod
    def _of(cls, function: Callable[..., Any] | None) -> Callback | None:
        if function is None:
            return None
        qualname = getattr(function, "__qualname__", None) or repr(function)
        return cls(
            getattr(function, "__module__", None), qualname, getattr(function, "__doc__", None)
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the three strings."""
        return (Callback, (self.module, self.qualname, self.doc))

    def __eq__(self, other: object) -> bool:
        """Compare equal to a record of the same callback."""
        if not isinstance(other, Callback):
            return NotImplemented
        return (self.module, self.qualname, self.doc) == (other.module, other.qualname, other.doc)

    def __hash__(self) -> int:
        """Hash as the three strings."""
        return hash((self.module, self.qualname, self.doc))

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: the record is of what was registered."""
        del value
        msg = f"'{name}' cannot be assigned; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion."""
        msg = f"'{name}' cannot be deleted; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        """Return the callback's qualified name and module."""
        return f"<Callback {self.qualname} in {self.module}>"


class SettingsGroup:
    """One level of the settings tree: names, each holding a value or a group below it.

    The names are the problem's own at the same place, so ``solution.settings.phases.boost
    .state.h.bounds`` is what ``problem.phases.boost.state.h.bounds`` was when solved.
    """

    __slots__ = ("_label", "_values")

    def __init__(self, label: str, values: dict[str, Any]) -> None:
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_values", values)

    def _names(self) -> tuple[str, ...]:
        values: dict[str, Any] = object.__getattribute__(self, "_values")
        return tuple(values)

    # Hidden from type checkers, which then report a misspelled name on the typed root.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Return the value or group recorded under `name`."""
            if name.startswith("_"):
                raise AttributeError(name)
            values = object.__getattribute__(self, "_values")
            if name in values:
                return values[name]
            label = object.__getattribute__(self, "_label")
            msg = f"{label} has no '{name}'.{suggest(name, tuple(values))}"
            raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: each name is one setting as it was solved."""
        del value
        label = object.__getattribute__(self, "_label")
        msg = f"{label}.{name} cannot be assigned; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion."""
        label = object.__getattribute__(self, "_label")
        msg = f"{label}.{name} cannot be deleted; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        """Offer the recorded names, which is what completion should show."""
        return list(self._names())

    def __eq__(self, other: object) -> bool:
        """Compare equal to settings holding the same values, so two solves can be compared."""
        if not isinstance(other, SettingsGroup):
            return NotImplemented
        mine: dict[str, Any] = object.__getattribute__(self, "_values")
        theirs: dict[str, Any] = object.__getattribute__(other, "_values")
        return mine == theirs

    __hash__ = None  # type: ignore[assignment]  # equal by value, and not immutable all through

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the label and the values, which are data all the way down."""
        return (
            type(self),
            (object.__getattribute__(self, "_label"), object.__getattribute__(self, "_values")),
        )

    def __repr__(self) -> str:
        """Return the group's path and its names."""
        label = object.__getattribute__(self, "_label")
        return f"<{label}: {', '.join(self._names()) or 'nothing'}>"


class Settings(SettingsGroup):
    """The problem's setup as it stood when it was solved: ``solution.settings``.

    Every name the problem lets a user set, spelled as the problem spells it, holding values
    only: a bound never set is ``(-inf, inf)``, a scale left to YAPSS is None, a guess reads as
    the problem reads it, and a block field's settings are one per row. The callbacks are
    recorded by identity, as `Callback`, under ``callbacks`` where the problem has ``register``.

    Attributes
    ----------
    name : str
        The problem's name.
    comment : str
        The problem's comment.
    spectral_method : str
        The collocation method every phase was solved with.
    catch_keyboard_interrupt : bool
        Whether Ctrl-C stopped the solve cleanly.
    objective, derivatives, ipopt_options, callbacks, parameter, discrete, phases
        The groups of settings, each named as on the problem.
    """

    __slots__ = ()

    if TYPE_CHECKING:
        name: str
        comment: str
        spectral_method: str
        catch_keyboard_interrupt: bool
        objective: Any
        derivatives: Any
        ipopt_options: Any
        callbacks: Any
        parameter: Any
        discrete: Any
        phases: Any


def _value(value: Any) -> Any:
    """Return a setting's value as the settings hold it: a block field's rows as a tuple."""
    return tuple(value) if isinstance(value, BlockRows) else value


def _fields(fields: Fields, label: str) -> SettingsGroup:
    """Return a vector's settings, field first, as the problem reaches them."""
    aspects = aspects_of(fields)
    declaration = object.__getattribute__(fields, "_declaration")
    return SettingsGroup(
        label,
        {
            name: SettingsGroup(
                f"{label}.{name}",
                {
                    aspect: _value(getattr(getattr(fields, name), aspect))
                    for aspect in aspects._held
                },
            )
            for name in declaration._fields
        },
    )


def _walk(value: Any, label: str) -> Any:
    """Return the settings below `value`, one of the problem's held objects."""
    if isinstance(value, Fields):
        return _fields(value, label)
    if isinstance(value, Phases):
        return SettingsGroup(label, {ph.name: _walk(ph, f"{label}.{ph.name}") for ph in value})
    if isinstance(value, Registry):
        return SettingsGroup(
            label,
            {name: Callback._of(function) for name, function in value._registered().items()},
        )
    if isinstance(value, IpoptOptions):
        return SettingsGroup(label, dict(value.get_options()))
    if isinstance(value, Container):
        values: dict[str, Any] = {name: getattr(value, name) for name in value._settable}
        for name in value._held:
            key = "callbacks" if name == "register" else name
            values[key] = _walk(object.__getattribute__(value, name), f"{label}.{key}")
        return SettingsGroup(label, values)
    msg = f"no settings are known for {value!r}"  # pragma: no cover - a new kind of setting
    raise TypeError(msg)


def settings_of(problem: Problem) -> Settings:
    """Return `problem`'s settings as they stand, as values only.

    Parameters
    ----------
    problem : Problem
        The problem to record.

    Returns
    -------
    Settings
        The settings, which no later edit of `problem` can alter.
    """
    tree = _walk(problem, "solution.settings")
    return Settings("solution.settings", object.__getattribute__(tree, "_values"))


# Each class reports the public module it is exported from, as the solution's classes do.
for _public in (Callback, Settings, SettingsGroup):
    _public.__module__ = "yapss.solution"
del _public
