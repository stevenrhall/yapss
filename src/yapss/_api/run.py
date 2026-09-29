"""

What was true of one solve beyond its problem: ``solution.run``.

The same setup can be solved differently on another machine or another day, with other versions
of the libraries, another Ipopt build, a different time taken, and different warnings. That is
recorded here. Nothing that names the machine or the person is: a solution is pickled and
shared, so there is no hostname, no user name, and no full library path.

The warnings are every warning the solve issued, whether or not the user's filters show it, so a
solution's record does not depend on what earlier solves warned about. They are caught for the
length of the solve with every filter set to "always", and then issued again, each at the file
and line it named, through the user's own filters: a warning is shown, ignored, shown once, or
raised as an error just as it would have been, at the user's line. The setup's warnings are
issued again just before Ipopt starts, so one filtered into an error stops the solve before
Ipopt runs; the rest, the convergence warning among them, when the solve ends.

"""

from __future__ import annotations

import contextlib
import platform
import sys
import time
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .solution import _Record

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["Recording", "Run", "Seconds", "run_record"]


class Seconds(_Record):
    """How long each part of a solve took, in seconds: ``solution.run.seconds``.

    Attributes
    ----------
    total : float
        The whole solve, measured, not summed.
    setup : float
        Everything before Ipopt started: validation, the derivative setup, the setup check.
    ipopt : float
        Ipopt's own run.
    solution : float
        Building the solution from what Ipopt returned.
    """

    __slots__ = ("ipopt", "setup", "solution", "total")
    _label = "solution.run.seconds"

    if TYPE_CHECKING:
        total: float
        setup: float
        ipopt: float
        solution: float


class Run(_Record):
    """What was true of this solve that the same setup could have produced otherwise.

    Attributes
    ----------
    yapss_version, python_version, numpy_version, casadi_version : str
        The versions in use.
    ipopt_version : str or None
        The Ipopt version its ``IpoptConfig.h`` states, or None where it does not say.
    ipopt_build : str
        Where the Ipopt library came from, ``"wheel"`` (CasADi's) or ``"conda-forge"``, with
        the library's file name.
    ipopt_options : dict
        The options Ipopt received and accepted: the problem's, and those YAPSS sets itself.
    platform : str
        The operating system and architecture.
    started : datetime.datetime
        When the solve started, in UTC.
    seconds : Seconds
        How long each part of the solve took.
    warnings : tuple of (str, str)
        Each warning issued during the solve, shown or not, as its category's name and its
        message, in order.
    """

    __slots__ = (
        "casadi_version",
        "ipopt_build",
        "ipopt_options",
        "ipopt_version",
        "numpy_version",
        "platform",
        "python_version",
        "seconds",
        "started",
        "warnings",
        "yapss_version",
    )
    _label = "solution.run"

    if TYPE_CHECKING:
        yapss_version: str
        python_version: str
        numpy_version: str
        casadi_version: str
        ipopt_version: str | None
        ipopt_build: str
        ipopt_options: dict[str, Any]
        platform: str
        started: datetime
        seconds: Seconds
        warnings: tuple[tuple[str, str], ...]


class Recording:
    """Record every warning a solve issues, and issue each again through the user's filters.

    Used as a context manager around the solve, with `checkpoint` called just before Ipopt
    starts. Inside, warnings are caught with every filter set to "always", so each is recorded
    in `issued` whatever the user's filters say. At the checkpoint and at the end, what was
    caught is issued again with `warnings.warn_explicit` at its own file and line, with the
    name and registry of the module that line belongs to, so the user's filters decide what is
    shown, including "once" and "default" deduplication. A warning they turn into an error
    raises there: at the checkpoint, before Ipopt runs, for everything warned during setup (a
    refused option, say), and at the end for the rest.

    Leaving `warnings.catch_warnings` resets Python's record of which warnings each location
    has already shown, for every warning in the process, so under the "default" filter a
    warning repeated from one line is shown once per solve.

    Parameters
    ----------
    issued : list
        Where each warning is appended, as ``(category name, message)``.
    """

    def __init__(self, issued: list[tuple[str, str]]) -> None:
        self.issued = issued
        self._catcher: Any = None
        self._caught: list[warnings.WarningMessage] = []

    def _start(self) -> None:
        self._catcher = warnings.catch_warnings(record=True)
        self._caught = self._catcher.__enter__()
        warnings.simplefilter("always")

    def _stop(self) -> list[warnings.WarningMessage]:
        self._catcher.__exit__(None, None, None)
        caught, self._caught = self._caught, []
        self.issued.extend((w.category.__name__, str(w.message)) for w in caught)
        return caught

    def checkpoint(self) -> None:
        """Issue what was caught so far through the user's filters, then catch again."""
        caught = self._stop()
        try:
            _issue_again(caught)
        finally:
            self._start()

    def __enter__(self) -> Recording:
        """Start catching."""
        self._start()
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        """Stop catching, and issue what was caught through the user's filters.

        If the solve raised, the warnings are still issued, but one the user's filters turn
        into an error does not replace the exception already on its way.
        """
        caught = self._stop()
        if kind is None:
            _issue_again(caught)
            return
        with contextlib.suppress(Warning):
            _issue_again(caught)


def _issue_again(caught: list[warnings.WarningMessage]) -> None:
    """Issue each warning again at its own file and line, through the user's filters."""
    for w in caught:
        # The module and its registry only when the line belongs to a loaded module: CPython's
        # warn_explicit shows nothing when given module=None, as for a line typed in a doctest.
        where: dict[str, Any] = {}
        module = _module_of(w.filename)
        if module is not None:
            where["module"] = module.__name__
            where["registry"] = module.__dict__.setdefault("__warningregistry__", {})
        warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, source=w.source, **where
        )


def _module_of(filename: str) -> ModuleType | None:
    """Return the loaded module whose file is `filename`, if there is one."""
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == filename:
            return module
    return None


def run_record(
    facts: dict[str, Any],
    started: datetime,
    clock: tuple[float, float],
    issued: list[tuple[str, str]],
    in_conda: bool,  # noqa: FBT001
) -> Run:
    """Return the record of one solve.

    Parameters
    ----------
    facts : dict
        What the back end's solver recorded: the Ipopt library and version, the options it
        accepted, and the clock just before and after Ipopt ran.
    started : datetime
        When the solve started, in UTC.
    clock : tuple of float
        `time.perf_counter` at the start of the solve and at its end.
    issued : list
        The warnings issued during the solve.
    in_conda : bool
        Whether Ipopt is conda-forge's rather than CasADi's.
    """
    import casadi  # noqa: PLC0415 -- the solve has already imported it

    from yapss import __version__  # noqa: PLC0415 -- the package imports this module

    begin, end = clock
    return Run(
        {
            "yapss_version": __version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "casadi_version": casadi.__version__,
            "ipopt_version": facts.get("ipopt_version"),
            "ipopt_build": (
                f"{'conda-forge' if in_conda else 'wheel'} ({facts.get('ipopt_library', '?')})"
            ),
            "ipopt_options": dict(facts.get("ipopt_options", {})),
            "platform": platform.platform(),
            "started": started,
            "seconds": Seconds(
                {
                    "total": end - begin,
                    "setup": facts["ipopt_started"] - begin,
                    "ipopt": facts["ipopt_finished"] - facts["ipopt_started"],
                    "solution": end - facts["ipopt_finished"],
                }
            ),
            "warnings": tuple(issued),
        }
    )


def now() -> tuple[datetime, float]:
    """Return the wall-clock time in UTC and the performance counter, read together."""
    return datetime.now(UTC), time.perf_counter()


# Each class reports the public module it is exported from, as the solution's classes do.
for _public in (Run, Seconds):
    _public.__module__ = "yapss.solution"
del _public
