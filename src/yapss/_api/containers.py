"""

The small setup objects: aspects, handles, and the problem itself.

Every one of them has a fixed set of public names. A name that holds a sub-object is read
only, and assigning to it explains the idiom instead of replacing the object; a name that
holds a value is set through a validator; anything else is refused with a suggestion. That is
the whole protection mechanism, and it is deliberately smaller than 0.3.0's, which had to
guard writable arrays as well.

"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

__all__ = ["Container", "HasRegistry", "Registry", "is_callable", "is_string", "is_subclass"]


def is_string(value: object) -> bool:
    """Report whether `value` is a string.

    The argument is typed as `object` so that the check still applies when the caller
    passed an annotated value; these checks exist for callers who write no annotations.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True if `value` is a string.
    """
    return isinstance(value, str)


def is_subclass(value: object, base: type) -> bool:
    """Report whether `value` is a class derived from `base`.

    Parameters
    ----------
    value : object
        The value to test.
    base : type
        The base class it must derive from.

    Returns
    -------
    bool
        True if `value` is such a class.
    """
    return isinstance(value, type) and issubclass(value, base)


def is_callable(value: object) -> bool:
    """Report whether `value` can be called.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True if `value` is callable.
    """
    return callable(value)


def suggest(name: str, candidates: tuple[str, ...]) -> str:
    """Return a " Did you mean 'x'?" fragment for `name`, or an empty string.

    Parameters
    ----------
    name : str
        The name the user wrote.
    candidates : tuple of str
        The names that exist.

    Returns
    -------
    str
        The suggestion, as part of the message so that notebooks show it.
    """
    close = difflib.get_close_matches(name, candidates, n=1)
    return f" Did you mean '{close[0]}'?" if close else ""


class Container:
    """Base of the setup objects. Subclasses list their public names.

    Attributes
    ----------
    _held : tuple of str
        Names holding a sub-object, which the user reads but never replaces.
    _settable : tuple of str
        Names holding a value the user sets.
    """

    _held: tuple[str, ...] = ()
    _settable: tuple[str, ...] = ()
    _label: str = "this object"

    def _names(self) -> tuple[str, ...]:
        return (*self._held, *self._settable)

    def _hold(self, name: str, value: Any) -> None:
        """Install a sub-object, bypassing the write rules. Used by YAPSS only."""
        object.__setattr__(self, name, value)

    def _advice(self, name: str) -> str:
        """Return the message for an attempt to replace the sub-object `name`."""
        held = object.__getattribute__(self, name)
        fields = getattr(held, "_fields", ())
        example = f"{name}.{fields[0]}" if fields else f"{name}.<field>"
        return (
            f"{self._label} {name} cannot be replaced; it is set one field at a time, "
            f"for example '{example} = ...'."
        )

    def _check(self, name: str, value: Any) -> Any:
        """Validate a settable value and return what to store. Overridden by subclasses."""
        del name
        return value

    # Hidden from type checkers, as `_backend.types_.Protected` hides its own: both of these
    # accept any name at runtime and answer for it there, and a type checker that can see
    # them stops reporting misspellings altogether. Every name a container really holds is
    # declared in its class body under `TYPE_CHECKING`, so static access is checked against
    # that list while the runtime messages stay the ones a user reads.
    if not TYPE_CHECKING:

        def __setattr__(self, name, value):
            """Set a public name, refusing anything the container does not declare."""
            if name.startswith("_"):
                object.__setattr__(self, name, value)
                return
            if name in self._settable:
                object.__setattr__(self, name, self._check(name, value))
                return
            if name in self._held:
                raise AttributeError(self._advice(name))
            msg = f"{self._label} has no setting '{name}'.{suggest(name, self._names())}"
            raise AttributeError(msg)

        def __getattr__(self, name):
            """Refuse an unknown name with a suggestion."""
            if name.startswith("_"):
                raise AttributeError(name)
            msg = f"{self._label} has no setting '{name}'.{suggest(name, self._names())}"
            raise AttributeError(msg)


class HasRegistry(Container):
    """A container whose callbacks live in a `Registry` held under ``register``.

    The registrations are offered as suggestions from the container itself, so that a callback
    reached for on its owner -- ``ph.continuous``, which a user who knows a phase has one may
    well try before looking -- is answered with where it lives rather than with silence.
    """

    def _names(self) -> tuple[str, ...]:
        names = super()._names()
        try:
            registry = object.__getattribute__(self, "register")
        except AttributeError:  # during __init__, before the registry is held
            return names
        return (*names, *(f"register.{name}" for name in registry._registrations))


class Registry(Container):
    """The callbacks of a problem or a phase, gathered under one name.

    Registration is a namespace of its own -- ``problem.register.objective`` rather than
    ``problem.objective`` -- for two reasons. The names it holds would otherwise sit beside the
    settings, where `objective` and `discrete` already name the aspects carrying `sense`,
    `scale` and `bounds`; and one namespace is one place to look for what can be registered.

    A registration is a method, so it is reached by normal lookup and this class only has to
    say what happens when a name is *assigned* instead of decorated, and what to suggest when
    one is misspelled.

    Attributes
    ----------
    _registrations : tuple of str
        The callbacks this registry accepts, in the order they are documented.
    """

    _registrations: tuple[str, ...] = ()

    def _names(self) -> tuple[str, ...]:
        return (*self._held, *self._settable, *self._registrations)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a registration, naming the idiom that works."""
        if not name.startswith("_") and name in self._registrations:
            msg = (
                f"{self._label} '{name}' is not assigned. Decorate the callback with "
                f"'register.{name}', or call 'register.{name}(callback)'."
            )
            raise AttributeError(msg)
        super().__setattr__(name, value)
