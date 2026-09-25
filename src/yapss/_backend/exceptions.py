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

The vendored `yapss._backend.mseipopt` package does not use these classes. It is written to
stand on its own, with no assumptions about YAPSS, and keeps its own `IpoptVerificationWarning`
and error classes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "REMOVED_NAMES",
    "YapssDeprecationWarning",
    "YapssError",
    "YapssWarning",
    "in_yapss",
    "user_stacklevel",
]

# The package directory, and the one directory inside it that holds user code rather than
# YAPSS's own: the examples are scripts a user runs, and a report they cause belongs there.
# `absolute`, not `resolve`: a frame records the path the module was imported by, symlinks
# and all, so resolving this one could stop it matching.
_PACKAGE = str(Path(__file__).absolute().parent.parent) + os.sep
_EXAMPLES = _PACKAGE + "examples" + os.sep


def in_yapss(filename: str) -> bool:
    """Return whether `filename` is YAPSS's own code: in the package, and not an example.

    The one line between YAPSS and its user, for everything that reports to the user: a
    warning points at the first frame outside it, and an exception from a callback is noted
    with the first function outside it. Stated as the whole package less the examples, so
    that code moved, renamed or added inside the package is on the right side without a list
    to keep up to date.
    """
    return filename.startswith(_PACKAGE) and not filename.startswith(_EXAMPLES)


def user_stacklevel() -> int:
    """Return the `stacklevel` that points a warning at the first frame outside YAPSS.

    Call it in the argument list of the `warnings.warn` it is for, so that the frame it starts
    from is the one issuing the warning::

        warnings.warn(msg, YapssWarning, stacklevel=user_stacklevel())

    Frames in the package are skipped, except the examples, and so are Python's frozen
    import frames, so a warning issued while `yapss` is being imported points at the user's
    ``import yapss``. The import frames are passed over without being counted, because
    `warnings.warn` skips them itself when it counts.
    """
    # When Python 3.11 support is retired, this can be replaced by the `skip_file_prefixes`
    # argument of `warnings.warn` (3.12+), given every directory of the package except
    # `examples` as the prefixes, so that the examples still count as user code.
    frame = sys._getframe(1)
    level = 1
    while (caller := frame.f_back) is not None:
        filename = frame.f_code.co_filename
        if "importlib" in filename and "_bootstrap" in filename:
            frame = caller
            continue
        if not in_yapss(filename):
            break
        frame = caller
        level += 1
    return level


# Public names removed in 0.3.0, each with what replaced it. Accessing one through `yapss` or
# `yapss.math` raises AttributeError with this message, rather than Python's bare "has no
# attribute", so that a line written for an earlier version -- typically a warnings filter --
# says why it fails. Remove the notices in 0.4.0 or after 2027-09, whichever is later.
REMOVED_NAMES = {
    "MirroredHessianPairWarning": (
        "MirroredHessianPairWarning was removed in 0.3.0. It warned about hand-written "
        "Hessian entries, which YAPSS 0.4 does not have, so there is nothing left for it to "
        "report. A warnings filter naming it can be deleted."
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


class LargeSegmentWarning(YapssWarning):
    """A mesh segment has more collocation points than it probably should.

    The collocation points of a segment are the roots of a polynomial of that degree, so a
    segment with many points is one high-order fit over the whole segment. On harder problems
    a large segment can slow Ipopt's convergence sharply or prevent it, and its quadrature
    takes longer to compute; published hp-adaptive methods raise the degree only to about 10
    to 16 per interval before splitting it. More, shorter segments are usually faster and
    more robust.

    This is advice, not a limit: the mesh is valid, and a deliberate single-segment (global)
    method is a legitimate thing to want. Silence it with
    ``warnings.simplefilter("ignore", yapss.LargeSegmentWarning)``.
    """
