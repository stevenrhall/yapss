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
import inspect
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CallbackT",
    "Container",
    "FillerT",
    "HasRegistry",
    "Registry",
    "is_callable",
    "is_string",
    "is_subclass",
]

CallbackT = TypeVar("CallbackT", bound="Callable[..., Any]")
"""A callback being registered, which a registration returns unchanged -- so a decorated
function keeps its own type, annotations and all, rather than becoming `Any`."""

FillerT = TypeVar("FillerT", bound="Callable[..., None]")
"""A callback that fills its `out` and returns nothing: the continuous and discrete callbacks.

An annotated one that returns anything is reported, as the run-time check refuses it; an
unannotated one returns `Any` to a checker and passes, as it should."""


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


def check_arity(callback: Any, count: int, what: str) -> None:
    """Refuse a callback that cannot be called with `count` positional arguments.

    Checked where it is registered: the solve would otherwise fail inside YAPSS's call with
    Python's own "takes 1 positional argument but 2 were given". A callable whose signature
    cannot be read (some builtins) is let through, since nothing certain can be said of it.
    """
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(*(None,) * count)
    except TypeError:
        name = getattr(callback, "__qualname__", repr(callback))
        form = "(arg)" if count == 1 else "(arg, out)"
        msg = (
            f"the {what} is called as {name}{form}, and its signature {signature} does not allow it"
        )
        raise TypeError(msg) from None


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
        """Return the message for an attempt to replace the sub-object `name`.

        Every object a container holds answers `_example` with a line the user could write,
        relative to its own name: a setting, a field's setting, an option, or for a registry a
        registration.
        """
        held = object.__getattribute__(self, name)
        if isinstance(held, Registry):
            return (
                f"{self._label} {name} cannot be replaced; register a callback through it, "
                f"for example '@{name}.{held._example()}'."
            )
        return (
            f"{self._label} {name} cannot be replaced; it is changed one setting at a time, "
            f"for example '{name}.{held._example()} = ...'."
        )

    def _example(self) -> str:
        """Return a setting written the way a user sets one, for a message: the first."""
        return self._settable[0]

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

        def __delattr__(self, name):
            """Refuse deleting a public name, which would leave the container without it."""
            if name.startswith("_"):
                object.__delattr__(self, name)
                return
            advice = "; assign it a new value instead" if name in self._settable else ""
            msg = f"{self._label} '{name}' cannot be deleted{advice}"
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

    def _example(self) -> str:
        """Return a registration, for a message: the first."""
        return self._registrations[0]

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a registration, naming the idiom that works."""
        if not name.startswith("_") and name in self._registrations:
            msg = (
                f"{self._label} '{name}' is not assigned. Decorate the callback with "
                f"'register.{name}', or call 'register.{name}(callback)'."
            )
            raise AttributeError(msg)
        super().__setattr__(name, value)
