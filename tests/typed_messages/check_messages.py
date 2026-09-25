"""Check that mypy prints, for each line of ``unannotated_setup.py``, the message its comment quotes.

Run by the tox mypy environments, which pin mypy's version, as

    python tests/typed_messages/check_messages.py --python-version 3.15

A comment is the message in full, or its beginning when it ends in "..."; ``# fine`` means no
error on that line. Both directions fail: a quoted message mypy no longer prints, and an error
on a line whose comment does not quote it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from mypy import api

TARGET = Path(__file__).with_name("unannotated_setup.py")
ERROR = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+): error: (?P<message>.*?)(?:  \[[a-z-]+\])?$")


def expected() -> dict[int, str | None]:
    """Return each commented line's expected message, None for ``# fine``."""
    found: dict[int, str | None] = {}
    lines = TARGET.read_text(encoding="utf-8").splitlines()
    inside = False
    for number, line in enumerate(lines, start=1):
        if line.startswith("# -- shown on the page"):
            inside = True
            continue
        if line.startswith("# -- end of what the page shows"):
            break
        if inside and "  # " in line:
            comment = line.split("  # ", 1)[1].strip()
            found[number] = None if comment == "fine" else comment
    return found


def matches(quoted: str, printed: str) -> bool:
    """Return whether mypy's message is the one quoted, in full or up to "..."."""
    if quoted.endswith("..."):
        return printed.startswith(quoted[:-3].rstrip())
    return printed == quoted


def main() -> int:
    """Run mypy on the target and compare its errors with the comments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", required=True)
    version = parser.parse_args().python_version
    stdout, _, _ = api.run(
        ["--strict", "--python-version", version, "--no-error-summary", str(TARGET)]
    )
    printed: dict[int, list[str]] = {}
    for line in stdout.splitlines():
        match = ERROR.match(line)
        if match:
            printed.setdefault(int(match["line"]), []).append(match["message"])

    problems = []
    quoted = expected()
    for number, message in quoted.items():
        got = printed.get(number, [])
        if message is None and got:
            problems.append(f"line {number}: expected no error, mypy printed {got}")
        elif message is not None and not any(matches(message, g) for g in got):
            problems.append(f"line {number}: expected {message!r}, mypy printed {got}")
    for number, got in printed.items():
        if number not in quoted:
            problems.append(f"line {number}: unexpected error {got}")

    if problems:
        print(f"{TARGET.name}: mypy's messages differ from the ones quoted:", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"{TARGET.name}: {len(quoted)} lines, every message as quoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
