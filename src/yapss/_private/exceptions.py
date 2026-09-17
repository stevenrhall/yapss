"""

The base classes of the warnings YAPSS issues and the errors it defines.

Every YAPSS warning points at the user's own code through ``stacklevel``, so a filter written
as ``module="yapss"`` matches none of them: the module recorded is the caller's. A common base
class is therefore the only reliable way to filter or escalate them in one line::

    import warnings
    import yapss

    warnings.simplefilter("error", yapss.YapssWarning)   # every YAPSS warning is now an error

`YapssError` is the base of the errors YAPSS defines. Each one also inherits the built-in
exception a caller would naturally catch, so ``except TypeError`` keeps working::

    class UnsupportedMathFunctionError(YapssError, TypeError): ...

Errors raised for ordinary bad input -- a bound that is not a number, a mesh with too few
collocation points -- stay plain `ValueError`, `TypeError`, and `IndexError`: they are what
Python itself would raise, and a base class adds nothing a caller would use.

The vendored `yapss._private.mseipopt` package does not use these classes. It is written to
stand on its own, with no assumptions about YAPSS, and keeps its own `IpoptVerificationWarning`
and error classes.
"""

from __future__ import annotations

__all__ = ["REMOVED_NAMES", "YapssDeprecationWarning", "YapssError", "YapssWarning"]

# Public names removed in 0.3.0, each with what replaced it. Accessing one through `yapss` or
# `yapss.math` raises AttributeError with this message, rather than Python's bare "has no
# attribute", so that a line written for an earlier version -- typically a warnings filter --
# says why it fails. Remove the notices in 0.4.0 or after 2027-09, whichever is later.
REMOVED_NAMES = {
    "MirroredHessianPairWarning": (
        "MirroredHessianPairWarning was removed in 0.3.0. Setting both orders of one Hessian "
        "variable pair, which it warned about, now raises ValueError. A warnings filter "
        "naming it can be deleted."
    ),
    "UnsupportedMathFunctionWarning": (
        "UnsupportedMathFunctionWarning was removed in 0.3.0. yapss.math.nextafter, signbit, "
        "and spacing, which it warned about on real arguments, now raise "
        "UnsupportedMathFunctionError on every argument. A warnings filter naming it can be "
        "deleted."
    ),
}


class YapssWarning(UserWarning):
    """Base class of every warning YAPSS issues.

    A `UserWarning`, so it is shown by default: these warnings report something about the
    user's own problem definition or solve, not about YAPSS's internals.
    """


class YapssDeprecationWarning(YapssWarning, FutureWarning):
    """A behavior that YAPSS will change or remove in a later release.

    Also a `FutureWarning` rather than a `DeprecationWarning`, because Python shows
    `DeprecationWarning` only in ``__main__``, and YAPSS problems are normally solved from a
    script or a notebook, where it would be hidden.
    """


class YapssError(Exception):
    """Base class of the errors YAPSS defines.

    Every subclass also inherits the built-in exception that describes the failure, so code
    that catches the built-in keeps working.
    """
