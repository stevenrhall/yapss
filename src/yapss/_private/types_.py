"""

Types used in YAPSS type hinting.

The `yapss._types` module defines types used in YAPSS type hinting, primarily
associated with the derivative structure of the user-defined objective, continuous, and
discrete functions, and with evaluating the derivatives using finite difference methods.

"""

# future imports
from __future__ import annotations

# standard imports
import difflib
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

# fmt: off
__all__ = [  # noqa: RUF022
    "CFIndex", "CFKey", "CFName", "CHFDS", "CHFDSPhase", "CHFDSTerm", "CHS", "CHSTerm", "CJFDS",
    "CJFDSPhase", "CJFDSTerm", "CJS", "CJSPhase", "CJSTerm", "CVIndex", "CVKey", "CVName",
    "DFIndex", "DHFDS", "DHS", "DHSTerm", "DJFDS", "DJS", "DJSTerm", "DVIndex", "DVKey",
    "DerivativeMethod", "DerivativeOrder", "OGS", "OHS", "OHSTerm", "PhaseIndex", "Sense",
    "SpectralMethod",
]
# fmt: on

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    ObjectArray = NDArray[np.object_]

SpectralMethod = Literal["lg", "lgr", "lgl"]
"""Spectral method: the allowed values of `Problem.spectral_method`."""

DerivativeMethod = Literal["auto", "central-difference", "central-difference-full", "user"]
"""Derivative method: the allowed values of `Derivatives.method`."""

DerivativeOrder = Literal["first", "second"]
"""Derivative order: the allowed values of `Derivatives.order`."""

Sense = Literal["minimize", "maximize"]
"""Optimization sense: the allowed values of `Problem.sense`."""

PhaseIndex = int  # NewType("PhaseIndex", int)
"""Phase index."""

CVName = str  # Literal["x", "u", "t", "s"]
"""Continuous variable name."""

CVIndex = int  # NewType("CVIndex", int)
"""Continuous variable index."""

CVKey = tuple[CVName, CVIndex]
"""Key type for the continuous function variables."""

CFName = str  # Literal["f", "g", "h"]
"""Continuous function name."""

CFIndex = int  # NewType("CFIndex", int)
"""Continuous function index."""

CFKey = tuple[CFName, CFIndex]
"""Continuous function key."""

CJSTerm = tuple[CFKey, CVKey]
"""Continuous Jacobian structure term."""

CJSPhase = tuple[CJSTerm, ...]
"""Continuous Jacobian structure phase."""

CJS = tuple[CJSPhase, ...]
"""Continuous Jacobian structure."""

CJFDSTerm = tuple[CVKey, tuple[CFKey, ...]]
"""Continuous Jacobian finite difference structure term."""

CJFDSPhase = tuple[CJFDSTerm, ...]
"""Continuous Jacobian finite difference structure phase."""

CJFDS = tuple[CJFDSPhase, ...]
"""Continuous Jacobian finite difference structure."""

CHSTerm = tuple[CFKey, CVKey, CVKey]
"""Continuous Hessian structure term."""

CHSPhase = tuple[CHSTerm, ...]
"""Continuous Hessian structure phase."""

CHS = tuple[CHSPhase, ...]
"""Continuous Hessian structure."""

CHFDSTerm = tuple[tuple[CVKey, CVKey], tuple[CFKey, ...]]
"""Continuous Hessian finite difference structure term."""

CHFDSPhase = tuple[CHFDSTerm, ...]
"""Continuous Hessian finite difference structure phase."""

CHFDS = tuple[CHFDSPhase, ...]
"""Continuous Hessian finite difference structure."""

DVName = str  # Literal["x0", "xf", "t0", "tf", "q", "s"]
"""Discrete variable name."""

DFIndex = int  # NewType("DFIndex", int)
"""Discrete function index."""

DVIndex = int  # NewType("DVIndex", int)
"""Discrete variable index."""

DVKey = tuple[PhaseIndex, DVName, DVIndex]
""" Discrete variable key."""

DJSTerm = tuple[DFIndex, DVKey]
"""Discrete Jacobian structure term."""

DJS = tuple[DJSTerm, ...]
"""Discrete Jacobian structure."""

DJFDS = tuple[tuple[DVKey, tuple[DFIndex, ...]], ...]
"""Discrete Jacobian finite difference structure."""

OGS = tuple[DVKey, ...]
"""Objective Gradient structure."""

OHSTerm = tuple[DVKey, DVKey]
"""Objective Hessian structure term."""

OHS = tuple[OHSTerm, ...]
"""Objective Hessian structure."""

DHSTerm = tuple[DFIndex, DVKey, DVKey]
"""Discrete Hessian structure term."""

DHS = tuple[DHSTerm, ...]
"""Discrete Hessian structure."""

DHFDS = tuple[tuple[DVKey, tuple[tuple[DVKey, tuple[DFIndex, ...]], ...]], ...]
"""Discrete Hessian finite difference structure."""

S = TypeVar("S")


class LimitOptions(Generic[S]):
    """Descriptor class for attributes that can only take on a limited set of values.

    Parameters
    ----------
    allowed_values : tuple[S, ...]
        The allowed values for the attribute.

    Raises
    ------
    ValueError
    """

    def __init__(self, allowed_values: tuple[S, ...]) -> None:
        """Initialize the descriptor."""
        self.allowed_values = allowed_values

    def __get__(self, instance: Any | None, owner: Any) -> S:
        """Get the value of the attribute."""
        return cast(S, getattr(instance, self.name))

    def __set__(self, instance: Any, value: S) -> None:
        """Set the value of the attribute."""
        if value not in self.allowed_values:
            msg = f"The value {value!r} is not allowed. Allowed values are in {self.allowed_values}"
            raise ValueError(msg)
        set_private(instance, self.name, value)

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Set the name of the attribute."""
        self.name = "_" + name


def set_private(instance: object, name: str, value: Any) -> None:
    """Set a backing field on a `Protected` instance, bypassing its public-attribute check.

    The one way internal code writes a private attribute once an instance is constructed:
    descriptors storing their values, and the few places that update internal state later.
    """
    object.__setattr__(instance, name, value)


class Field(Generic[S]):
    """A settable attribute of a `Protected` class, stored as given.

    For a public attribute that needs no conversion or check; declaring it as a descriptor
    is what makes it settable on a sealed instance.
    """

    private_name: str

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Record the backing field name."""
        self.private_name = "_" + name

    def __get__(self, instance: Any | None, owner: Any) -> S:
        """Return the stored value."""
        if instance is None:
            return self  # type: ignore[return-value]
        return cast(S, instance.__dict__[self.private_name])

    def __set__(self, instance: Any, value: S) -> None:
        """Store the value."""
        set_private(instance, self.private_name, value)


class _Sealing(type):
    """Metaclass that turns protection on once an instance's constructor has returned.

    Constructors assign freely, however deeply they call each other; protection starts
    when the outermost `__init__` returns, so no constructor has to remember a final step.
    """

    if not TYPE_CHECKING:  # a metaclass __call__ would hide constructor signatures from mypy

        def __call__(cls, *args, **kwargs):
            instance = super().__call__(*args, **kwargs)
            object.__setattr__(instance, "_sealed", True)
            return instance


class Protected(metaclass=_Sealing):
    """Base class whose instances accept assignment only to their public, settable attributes.

    Which names are settable is derived from the class, not listed: a public name whose
    class attribute is a data descriptor -- a property with a setter, or a descriptor with
    `__set__` -- is settable; every other name is refused once the instance is constructed.
    Refusals name the attribute and say why: a misspelling is offered the closest settable
    name, a public attribute without a setter is read-only, and a private backing name is
    treated as unknown, so validation cannot be bypassed through it. Internal code writes
    backing fields with `set_private`.

    `__setattr__` and `__delattr__` are hidden from type checkers: a class that defines
    `__setattr__` is taken to accept assignment to any name, which would stop a type
    checker from flagging a misspelled attribute.
    """

    _removed_attrs: ClassVar[dict[str, str]] = {}
    """Attributes removed from the public API, mapped to the message explaining what to do.

    Checked on assignment only, so a removed name appears nowhere a type checker or IDE
    would offer it. Reads get the ordinary AttributeError: handling them would take a
    `__getattr__`, which makes type checkers accept every attribute name on the class.
    """

    _settable: ClassVar[frozenset[str]] = frozenset()
    """Public names whose class attribute is a data descriptor; computed per class."""

    _class_public: ClassVar[frozenset[str]] = frozenset()
    """Public names defined on the class or its bases; computed per class."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Derive the settable and public names of a new subclass."""
        super().__init_subclass__(**kwargs)
        settable: set[str] = set()
        public: set[str] = set()
        for klass in reversed(cls.__mro__):
            for name, attribute in vars(klass).items():
                if name.startswith("_"):
                    continue
                public.add(name)
                if _is_settable(attribute):
                    settable.add(name)
                else:
                    settable.discard(name)
        cls._settable = frozenset(settable)
        cls._class_public = frozenset(public)

    if not TYPE_CHECKING:

        def __setattr__(self, name: str, value: Any) -> None:
            """Set a public, settable attribute; refuse anything else once constructed."""
            settable = name in self._settable or not self.__dict__.get("_sealed", False)
            if not settable or name in self._removed_attrs:
                raise AttributeError(self._refusal(name))
            object.__setattr__(self, name, value)

        def __delattr__(self, name: str) -> None:
            """Refuse deletion, unless a settable descriptor defines its own `__delete__`."""
            if name in self._settable:
                attribute = next(vars(k)[name] for k in type(self).__mro__ if name in vars(k))
                if _is_deletable(attribute):
                    object.__delattr__(self, name)
                    return
            msg = f"cannot delete '{type(self).__name__}' attribute '{name}'"
            raise AttributeError(msg)

    def _refusal(self, name: str) -> str:
        """Explain why assignment to `name` is refused."""
        if name in self._removed_attrs:
            return self._removed_attrs[name]
        head = f"cannot set '{type(self).__name__}' attribute '{name}'"
        public = self._class_public | {n for n in self.__dict__ if not n.startswith("_")}
        if name in public:
            return f"{head}: it is read-only"
        match = difflib.get_close_matches(name, sorted(self._settable), n=1)
        if match:
            return f"{head}: no such attribute; did you mean '{match[0]}'?"
        return f"{head}: no such attribute"


def _is_settable(attribute: object) -> bool:
    """Whether a class attribute makes its name assignable on instances."""
    if isinstance(attribute, property):
        return attribute.fset is not None
    return hasattr(type(attribute), "__set__")


def _is_deletable(attribute: object) -> bool:
    """Whether a settable class attribute handles deletion itself."""
    if isinstance(attribute, property):
        return attribute.fdel is not None
    return hasattr(type(attribute), "__delete__")
