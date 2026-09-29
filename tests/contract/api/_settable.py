"""Every name a user can set on a problem, found by walking the problem as a user reaches it.

A problem's setting is reached by attribute from the problem: through its containers, whose
public names are the ones they hold and the ones they let be set; through a vector, whose
fields are what `dir` offers on it; and through a field, whose settings are what `dir` offers
on that. The walk ends at a settable name. Two things the problem holds are values set whole
rather than containers walked: its Ipopt options, which are whatever the user set, and its
callbacks, which are registered. Both are leaves here, under the names the settings use.

`solution.settings` must have exactly these paths (tests/contract/api/test_contract_settings.py),
and the quick-reference tree the documentation shows is checked against them too.
"""

from __future__ import annotations

from typing import Any

import yapss
from yapss._api.containers import Container, Registry
from yapss._api.fields import Fields
from yapss._backend.ipopt_options import IpoptOptions

__all__ = ["settable_paths"]


def _walk(obj: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(obj, Fields):
        for field in dir(obj):
            settings = getattr(obj, field)
            for setting in dir(settings):
                out[f"{path}.{field}.{setting}"] = getattr(settings, setting)
        return
    if isinstance(obj, yapss.Phases):
        for phase in obj:
            _walk(phase, f"{path}.{phase.name}", out)
        return
    if isinstance(obj, Registry):
        for name in obj._registrations:
            out[f"{path.rsplit('.', 1)[0]}.callbacks.{name}"] = obj
        return
    if isinstance(obj, IpoptOptions):
        out[path] = obj
        return
    assert isinstance(obj, Container), path
    for name in obj._settable:
        out[f"{path}.{name}"] = getattr(obj, name)
    for name in obj._held:
        _walk(object.__getattribute__(obj, name), f"{path}.{name}", out)


def settable_paths(problem: yapss.Problem) -> dict[str, Any]:
    """Return every settable path of `problem`, rooted at ``problem``, with what it holds.

    A leaf that is a value maps to that value. The Ipopt options map to the options object,
    and a callback to the registry that holds it, since neither is a single value to compare.
    """
    out: dict[str, Any] = {}
    _walk(problem, "problem", out)
    return out
