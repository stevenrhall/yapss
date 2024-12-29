"""

Types used in YAPSS type hinting.

The `yapss._types` module defines types used in YAPSS type hinting, primarily
associated with the derivative structure of the user-defined objective, continuous, and
discrete functions, and with evaluating the derivatives using finite difference methods.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import Any, Generic, TypeVar, cast

# fmt: off
__all__ = [  # noqa: RUF022
    "CFIndex", "CFKey", "CFName", "CHFDS", "CHFDSPhase", "CHFDSTerm", "CHS", "CHSTerm", "CJFDS",
    "CJFDSPhase", "CJFDSTerm", "CJS", "CJSPhase", "CJSTerm", "CVIndex", "CVKey", "CVName",
    "DFIndex", "DHFDS", "DHS", "DHSTerm", "DJFDS", "DJS", "DJSTerm", "DVIndex", "DVKey", "OGS",
    "OHS", "OHSTerm", "PhaseIndex",
]
# fmt: on

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    ObjectArray = NDArray[np.object_]

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
        setattr(instance, self.name, value)

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Set the name of the attribute."""
        self.name = "_" + name


class Protected:
    """Class for which only select attributes can be set or deleted."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Set the attribute if it is allowed."""
        if hasattr(self, "_allowed_attrs") and name not in self._allowed_attrs:
            msg = f"cannot set '{self.__class__.__name__}' attribute '{name}'"
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete the attribute if it is allowed."""
        if hasattr(self, "_allowed_del_attrs") and name not in self._allowed_del_attrs:
            msg = f"cannot delete '{self.__class__.__name__}' attribute '{name}'"
            raise AttributeError(msg)
        super().__delattr__(name)
