"""

Recognize code written for YAPSS 0.3 or earlier, and say what happened.

Every release before 0.4 declared a problem as ``Problem(name=..., nx=...)``, with both keywords
keyword-only and ``nx`` required, so an ``nx`` keyword identifies such code exactly: no 0.4-style
call can pass it. The other thing old code reaches before it fails is one of the argument
classes it annotated callbacks with, which 0.4 does not export. Both are answered with one
message: what the line belongs to, how to keep the code running as it is, and where to start
porting it.

The message is kept indefinitely. It guards the one boundary old code crosses, not a single
name, and old notebooks outlive their software. It names the installed version, so it stays
true in every later release. Its link to the old API's documentation is versioned, so later
releases cannot break it; its link for porting is the documentation's front door, which always
shows the newest release and whatever guidance for upgrading it has.

"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["OLD_ROOT_NAMES", "old_api_message"]

OLD_API_DOCS = "https://yapss.readthedocs.io/en/v0.3.0/"
"""The documentation of the last release of the old API."""

CURRENT_DOCS = "https://yapss.readthedocs.io/"
"""The documentation of the newest release: the address redirects to Read the Docs' `stable`."""

OLD_ROOT_NAMES = frozenset(
    {
        "ObjectiveArg",
        "ObjectiveGradientArg",
        "ObjectiveHessianArg",
        "DiscreteArg",
        "DiscreteJacobianArg",
        "DiscreteHessianArg",
        "ContinuousJacobianArg",
        "ContinuousHessianArg",
    }
)
"""Names ``yapss`` exported in every release from 0.1.0 to 0.3.0 and does not export now."""


def old_api_message(what: str) -> str:
    """Return the message for `what`, a line or name belonging to the API before 0.4.

    Parameters
    ----------
    what : str
        What the code used, as the first line should name it: ``"Problem(name=..., nx=...)"``
        or ``"yapss.ObjectiveArg"``.

    Returns
    -------
    str
        The message.
    """
    # imported here: the message is built only on the way to raising
    from yapss._backend.config import get_conda_prefix  # noqa: PLC0415

    try:
        installed = version("yapss")
    except PackageNotFoundError:
        installed = "0.4 or later"
    installer = "conda" if get_conda_prefix() else "pip"
    return (
        f"{what} is the API of YAPSS 0.3 and earlier, and this is YAPSS {installed}.\n"
        f"\n"
        f"YAPSS 0.4 redesigned the API. A problem is declared with classes that name its "
        f"states, controls, constraints and phases, and callbacks read and write them by "
        f"those names. Code written for 0.3 or earlier does not run unchanged.\n"
        f"\n"
        f"To keep this code running as it is, install the last release of the old API:\n"
        f"\n"
        f'    {installer} install "yapss<0.4"\n'
        f"\n"
        f"Its documentation is at {OLD_API_DOCS}\n"
        f"\n"
        f"To port the code, see the documentation for the current release at {CURRENT_DOCS}"
    )
