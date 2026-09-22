"""What a contract clause is written with, shared by both front ends' suites.

`raises` and `warns` are the whole vocabulary: a clause says what a user did, and what YAPSS
told them about it. Both record what they saw when `YAPSS_ERROR_CATALOG` is set, which is how
the error catalogue is generated rather than written.

Each record carries the *raise site* -- the line of YAPSS that produced the message -- so that
the messages a suite states can be compared against the messages the package can produce. A
raise site absent from a run is a refusal no clause covers, which `tools/contract_gaps.py`
reports. Nothing else can answer that question: a message names itself, but only the site says
where it came from.

The front end a record belongs to is read from the clause file's own location, so a suite for
one front end cannot file its messages under the other by forgetting to say which it is.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import yapss

__all__ = ["not_yet", "proposed", "raises", "warns"]

# Set YAPSS_ERROR_CATALOG to a path and every message `raises` and `warns` see is appended
# there, one JSON object per line, for `tools/error_catalog.py` to render. Off by default, so
# an ordinary run is untouched; one line per record, appended, because xdist runs the suite in
# several processes at once.
_CATALOG = os.environ.get("YAPSS_ERROR_CATALOG")

_PACKAGE = Path(yapss.__file__).parent


def _site(error: BaseException) -> str:
    """Return the line of YAPSS that raised `error`, as ``module.py:line``.

    The deepest frame inside the package, so that a message raised by a helper is recorded
    where it is written rather than where it was called from. The path is written with forward
    slashes on every platform, so that a catalogue built on Windows names the same sites.
    """
    frames = [
        frame
        for frame in traceback.extract_tb(error.__traceback__)
        if Path(frame.filename).is_relative_to(_PACKAGE)
    ]
    if not frames:
        return ""
    frame = frames[-1]
    return f"{Path(frame.filename).relative_to(_PACKAGE).as_posix()}:{frame.lineno}"


def _front(path: str) -> str:
    """Return the front end whose clauses live in `path`."""
    return "api" if Path(path).parent.name == "api" else "legacy"


def _area(frame: Any) -> str:
    """Return the subject of the clause file the calling `frame` is in.

    Its file name, unless the file sets `AREA` -- which a file does when it is the second or
    third file about one subject and its own name would split that subject's messages into
    sections that do not exist.
    """
    area = frame.f_globals.get("AREA")
    if area:
        return str(area)
    return Path(frame.f_code.co_filename).stem.replace("test_contract_", "")


def _record(**entry: Any) -> None:
    """Append one record to the catalogue."""
    with Path(_CATALOG).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")


@contextmanager
def raises(exc: type[BaseException], *fragments: str, at: str | None = None) -> Iterator[Any]:
    """Assert that the block raises `exc` with every fragment in its message or its notes.

    Notes count as message: Python prints them directly under the message, which is where a
    user reads them (YAPSS adds one naming the callback an exception came from).

    With `at`, also assert that the exception was raised by the statement in the block
    whose source contains `at` -- that is, the traceback's last frame in the calling test
    file is that statement, the stand-in for the user's own line.
    """
    caller = sys._getframe(2)  # 0: here, 1: contextmanager, 2: test
    caller_file = caller.f_code.co_filename
    # A clause with no fragment states only that *something* went wrong, which is how a
    # message that falls through to Python's own wording passes unnoticed: the exception type
    # is right and nothing checks the words. Every clause names at least part of what it
    # expects to read.
    #
    # Required of the 0.4.0 suite only. Twelve clauses of the 0.3.0 suite predate the rule and
    # have the same hole; whether to close it there is a separate decision, and failing that
    # suite to make the point would be the wrong way to raise it.
    if _front(caller_file) == "api":
        assert fragments, "a clause states what the message says: give raises() a fragment of it"
    with pytest.raises(exc) as info:
        yield info
    message = "\n".join([str(info.value), *getattr(info.value, "__notes__", [])])
    missing = [fragment for fragment in fragments if fragment not in message]
    assert not missing, f"message {message!r} lacks {missing!r}"
    frames = [
        frame
        for frame in traceback.extract_tb(info.value.__traceback__)
        if frame.filename == caller_file
    ]
    if at is not None:
        assert frames, "the traceback does not pass through the calling test"
        line = frames[-1].line or ""
        assert at in line, f"raised by the statement {line!r}, not by the one containing {at!r}"
    if _CATALOG:
        _record(
            front=_front(caller_file),
            area=_area(caller),
            kind="error",
            type=type(info.value).__name__,
            message=message,
            statement=(frames[-1].line or "" if frames else "").strip(),
            test=caller.f_code.co_name,
            site=_site(info.value),
        )


@contextmanager
def warns(category: type[Warning], *fragments: str, at: str | None = None) -> Iterator[Any]:
    """Assert that the block warns `category` with every fragment in the message.

    The counterpart of `raises`, and recorded beside it: a warning is something YAPSS tells a
    user about input it accepted, and a user meets the two the same way.

    With `at`, also assert that the warning points at the statement in the block whose source
    contains it. A warning's `stacklevel` is the whole of its usefulness -- one that points
    into YAPSS tells the user nothing about their own code -- so it is checked here rather
    than trusted.
    """
    caller = sys._getframe(2)  # 0: here, 1: contextmanager, 2: test
    caller_file = caller.f_code.co_filename
    if _front(caller_file) == "api":
        assert fragments, "a clause states what the warning says: give warns() a fragment"
    with pytest.warns(category) as record:
        yield record
    messages = [str(item.message) for item in record]
    missing = [fragment for fragment in fragments if fragment not in "\n".join(messages)]
    assert not missing, f"warnings {messages!r} lack {missing!r}"
    first = record[0]
    if fragments:
        first = next(item for item in record if fragments[0] in str(item.message))
    if at is not None:
        source = Path(first.filename).read_text(encoding="utf-8").splitlines()
        line = source[first.lineno - 1] if 0 < first.lineno <= len(source) else ""
        assert first.filename == caller_file, (
            f"warned from {first.filename}, not from the calling test: a warning must point at "
            f"the user's own line"
        )
        assert at in line, f"warned by the statement {line.strip()!r}, not by one containing {at!r}"
    if _CATALOG:
        source = Path(first.filename).read_text(encoding="utf-8").splitlines()
        statement = source[first.lineno - 1] if 0 < first.lineno <= len(source) else ""
        _record(
            front=_front(caller_file),
            area=_area(caller),
            kind="warning",
            type=type(first.message).__name__,
            message=str(first.message),
            statement=statement.strip() if first.filename == caller_file else "",
            test=caller.f_code.co_name,
            site=(
                f"{Path(first.filename).name}:{first.lineno}"
                if Path(first.filename).is_relative_to(_PACKAGE)
                else ""
            ),
        )


def not_yet(item: str, clause: str) -> pytest.MarkDecorator:
    """Mark a clause the code does not meet yet, naming the item that will meet it."""
    return pytest.mark.xfail(strict=True, reason=f"{item}: {clause}")


def proposed(clause: str) -> pytest.MarkDecorator:
    """Mark a clause whose behavior is proposed but not yet decided."""
    return pytest.mark.xfail(strict=True, reason=f"PROPOSED, not decided: {clause}")
