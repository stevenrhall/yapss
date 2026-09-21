"""

A vector's settings, reached field first: ``ph.state.x.bounds``.

The settings are stored as they always were -- one instance of the declaration per aspect,
``bounds``, ``guess``, ``scale`` and the rest -- because that is what validates each write
against its own grammar. What changes is the way in. A bound is a property *of a field*, so
the field is named first and the property last, which is also the only order a type checker
can follow: with the property last its name is fixed and its type known, where with the
field last one attribute had to answer for a bound, a guess and a scale at once, and so
answered `Any`.

`Fields` is what ``ph.state`` is: the fields of one vector, each a `FieldSettings`, which
forwards every read and write of a setting to the aspect that holds it, so every check and every
message is the one that aspect already made. Nothing here validates anything.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .containers import suggest

if TYPE_CHECKING:
    from .containers import Container
    from .vector import Vector

__all__ = ["FieldSettings", "Fields", "aspects_of"]


def aspects_of(fields: Any) -> Any:
    """Return the aspect container behind a `Fields`, for YAPSS's own reads.

    Parameters
    ----------
    fields : Fields
        What ``ph.state`` or ``problem.discrete`` is.

    Returns
    -------
    Container
        The container holding one instance of the declaration per aspect.
    """
    return object.__getattribute__(fields, "_aspects")


class Fields:
    """The fields of one vector, each reached by name and then by setting.

    Parameters
    ----------
    aspects : Container
        The container holding one instance of the declaration per aspect.
    declaration : type[Vector]
        The vector's declaration, whose fields these are.
    label : str
        How the vector is named in a message, such as ``"phase 'slide' state"``.
    """

    __slots__ = ("_aspects", "_declaration", "_label")

    def __init__(self, aspects: Container, declaration: type[Vector], label: str) -> None:
        object.__setattr__(self, "_aspects", aspects)
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_label", label)

    def _example(self) -> str:
        """Return a setting written the way this vector's are, for a message."""
        declaration: type[Vector] = object.__getattribute__(self, "_declaration")
        aspects: Container = object.__getattribute__(self, "_aspects")
        field = declaration._fields[0] if declaration._fields else "<field>"
        return f"{field}.{aspects._held[0]}"

    def __getattr__(self, name: str) -> FieldSettings:
        """Return the field `name`, or refuse the name with the form that works."""
        if name.startswith("_"):
            raise AttributeError(name)
        declaration: type[Vector] = object.__getattribute__(self, "_declaration")
        aspects: Container = object.__getattribute__(self, "_aspects")
        label: str = object.__getattribute__(self, "_label")
        if name in declaration._fields:
            return FieldSettings(aspects, name, label)
        if name in aspects._held:
            # the aspect-first spelling, which is what anyone who wrote 0.3.0 or the earlier
            # 0.4.0 will reach for first, so it gets the exact rewrite rather than a suggestion
            field = declaration._fields[0] if declaration._fields else "<field>"
            msg = (
                f"{label} has no field '{name}'. A setting belongs to a field, so the field is "
                f"named first: '{field}.{name}', not '{name}.{field}'."
            )
            raise AttributeError(msg)
        msg = f"{label} has no field '{name}'.{suggest(name, declaration._fields)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a field, which is set one setting at a time."""
        del value
        label: str = object.__getattribute__(self, "_label")
        msg = (
            f"{label} {name!r} cannot be assigned. A field is set one setting at a time, for "
            f"example '{self._example()} = ...'."
        )
        raise AttributeError(msg)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass a call to the aspects, which answer the slip it usually is.

        ``@problem.discrete`` written for ``@problem.register.discrete`` is the natural mistake,
        and the discrete aspects already answer it with the registration that was meant.
        """
        aspects: Container = object.__getattribute__(self, "_aspects")
        if callable(aspects):
            return aspects(*args, **kwargs)
        label: str = object.__getattribute__(self, "_label")
        msg = f"{label} holds settings and is not callable"
        raise TypeError(msg)

    def __getitem__(self, index: Any) -> Any:
        """Refuse a position: a vector's fields are reached by name, never by position."""
        raise TypeError(self._no_positions(index))

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse a position, as reading one is refused."""
        del value
        raise TypeError(self._no_positions(index))

    def _no_positions(self, index: Any) -> str:
        label: str = object.__getattribute__(self, "_label")
        return (
            f"{label}[{index!r}]: a vector has no positions. Its fields are reached by name, "
            f"and a setting after the field, for example '{self._example()}'."
        )

    def __dir__(self) -> list[str]:
        """Offer the fields, which is what completion should show."""
        declaration: type[Vector] = object.__getattribute__(self, "_declaration")
        return list(declaration._fields)

    def __repr__(self) -> str:
        """Return the label and the fields."""
        declaration: type[Vector] = object.__getattribute__(self, "_declaration")
        label: str = object.__getattribute__(self, "_label")
        return f"<{label}: {', '.join(declaration._fields) or 'no fields'}>"


class FieldSettings:
    """One field of a vector, whose settings are its attributes: ``ph.state.x.bounds``.

    Parameters
    ----------
    aspects : Container
        The container holding one instance of the declaration per aspect.
    name : str
        The field's name.
    label : str
        How the vector is named in a message.
    """

    __slots__ = ("_aspects", "_label", "_name")

    def __init__(self, aspects: Container, name: str, label: str) -> None:
        object.__setattr__(self, "_aspects", aspects)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_label", label)

    def _refuse(self, setting: str) -> AttributeError:
        aspects: Container = object.__getattribute__(self, "_aspects")
        label: str = object.__getattribute__(self, "_label")
        name: str = object.__getattribute__(self, "_name")
        msg = f"{label} {name!r} has no setting {setting!r}.{suggest(setting, aspects._held)}"
        return AttributeError(msg)

    def __getattr__(self, setting: str) -> Any:
        """Return the field's value for `setting`, read from the aspect that holds it."""
        if setting.startswith("_"):
            raise AttributeError(setting)
        aspects: Container = object.__getattribute__(self, "_aspects")
        if setting not in aspects._held:
            raise self._refuse(setting)
        return getattr(getattr(aspects, setting), object.__getattribute__(self, "_name"))

    def __setattr__(self, setting: str, value: Any) -> None:
        """Set the field's value for `setting`, through the aspect that validates it."""
        aspects: Container = object.__getattribute__(self, "_aspects")
        if setting not in aspects._held:
            raise self._refuse(setting)
        setattr(getattr(aspects, setting), object.__getattribute__(self, "_name"), value)

    def __getitem__(self, index: Any) -> Any:
        """Refuse an index on the field itself: rows belong to one of its settings."""
        raise TypeError(self._index_a_setting(index))

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse an index on the field itself, as reading one is refused."""
        del value
        raise TypeError(self._index_a_setting(index))

    def _index_a_setting(self, index: Any) -> str:
        aspects: Container = object.__getattribute__(self, "_aspects")
        label: str = object.__getattribute__(self, "_label")
        name: str = object.__getattribute__(self, "_name")
        return (
            f"{label} {name!r}[{index!r}]: a field has no rows of its own. Index one of its "
            f"settings, for example '{name}.{aspects._held[0]}[{index!r}]'."
        )

    def __dir__(self) -> list[str]:
        """Offer the settings this field has."""
        aspects: Container = object.__getattribute__(self, "_aspects")
        return list(aspects._held)

    def __repr__(self) -> str:
        """Return the field's label and name."""
        label: str = object.__getattribute__(self, "_label")
        return f"<{label} {object.__getattribute__(self, '_name')!r}>"
