"""

Generate the Ipopt option tables from the Ipopt documentation.

Ipopt's ``doc/options.dox`` is the authoritative list of its options, and it records more
than the name: the kind (Integer, Number, String), the valid range of a numeric option, and
the allowed settings of a string one. This script turns that into two generated artifacts:

- the annotations on ``IpoptOptions`` (``name: kind | None``), which give an IDE its
  completions and a type checker the names it reports a misspelling against;
- ``_backend/ipopt_option_specs.py``, the table of each option's kind, range and settings.

Neither decides whether Ipopt accepts an option. Which options exist, and which values they
take, depends on the build, which need not match the documentation this was scraped from, so
Ipopt stays the authority. YAPSS reads an option's kind, to check the type of a value where it
is assigned, and its name, to add a hint when Ipopt refuses it: a did-you-mean for a near miss,
or that this build might not provide a documented option (``linear_solver = "ma27"`` is
documented, and refused by any build without HSL). The ranges and settings are kept as the
reference Ipopt's documentation gives; nothing reads them.

Run it from the repository root::

    python utils/get_ipopt_options.py --specs   # writes the spec module; then run black on it
    python utils/get_ipopt_options.py           # prints the annotation lines

Pass ``--ref releases/x.y.z`` to scrape a given release rather than the newest. At the release
the committed table names, the spec module comes back unchanged but for its date, and the
annotation lines are exactly those on ``IpoptOptions``.

Neither output is generated at build time: this needs the network, and the result is
committed.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "https://api.github.com/repos/coin-or/ipopt"
SOURCE = f"{REPO}/contents/doc/options.dox"
SPECS = Path("src/yapss/_backend/ipopt_option_specs.py")

# Documented as read only from an ipopt.opt file, but accepted through the C interface YAPSS
# uses, so annotated like any other option; tests/modules/test_ipopt_defaults.py sets both and
# reads back the log Ipopt wrote.
SETTABLE_IN_CODE = frozenset({"output_file", "file_print_level"})

# "The valid range for this real option is 0 < tol and its default value is ..."
RANGE = re.compile(
    r"valid range for this (?:real|integer) option is\s*(.+?) and its default",
    re.S,
)
# "Possible values:\n - monotone: use the ..."  and  "Possible values: yes, no"
BULLETED = re.compile(r"^\s*-\s*([A-Za-z0-9_.+*-]+):", re.M)
INLINE = re.compile(r"Possible values:\s*([^\n<]+)\n")


def latest_release() -> str:
    """Return the newest ``releases/x.y.z`` tag in the Ipopt repository."""
    with urllib.request.urlopen(f"{REPO}/tags?per_page=100") as stream:  # noqa: S310
        tags = [tag["name"] for tag in json.load(stream)]
    releases = [tag for tag in tags if tag.startswith("releases/")]
    if not releases:
        msg = "no releases/x.y.z tag found in the Ipopt repository"
        raise RuntimeError(msg)
    return max(releases, key=lambda tag: tuple(int(p) for p in tag.split("/")[1].split(".")))


def fetch(ref: str) -> str:
    """Return the text of Ipopt's options.dox at ``ref``.

    Pinned to a release tag rather than the default branch, so that the table always
    describes one named Ipopt version and the scrape is reproducible.
    """
    with urllib.request.urlopen(f"{SOURCE}?ref={ref}") as stream:  # noqa: S310 -- fixed URL
        data = json.load(stream)
    if "content" not in data:
        msg = f"options.dox not found at {SOURCE}?ref={ref}"
        raise RuntimeError(msg)
    return base64.b64decode(data["content"]).decode("utf-8")


def parse_range(text: str, name: str) -> tuple[float | None, bool, float | None, bool]:
    """Return (low, low_inclusive, high, high_inclusive) for a numeric option.

    ``None`` on a side means unbounded. The documentation writes the range as an inequality
    around the option name, such as ``0 <= max_iter`` or ``-2 <= mumps_scaling <= 77``.
    """
    match = RANGE.search(text)
    if not match:
        return (None, False, None, False)
    expression = " ".join(match.group(1).split()).replace("&le;", "<=").replace("&lt;", "<")
    parts = re.split(rf"\s*(<=|<)\s*{re.escape(name)}\s*", expression, maxsplit=1)
    low, low_inclusive = None, False
    if len(parts) == 3 and parts[0].strip():
        low, low_inclusive = float(parts[0]), parts[1] == "<="
        rest = parts[2]
    else:
        rest = expression.replace(name, "", 1)
    high, high_inclusive = None, False
    tail = re.match(r"\s*(<=|<)\s*(\S+)", rest)
    if tail:
        high, high_inclusive = float(tail.group(2)), tail.group(1) == "<="
    return (low, low_inclusive, high, high_inclusive)


def parse_values(text: str) -> tuple[str, ...] | None:
    """Return the allowed settings of a string option, or None if it is free-form."""
    values = BULLETED.findall(text)
    if not values:
        inline = INLINE.search(text)
        if inline:
            values = [v.strip() for v in inline.group(1).split(",") if v.strip()]
    if not values or "*" in values:  # "*" documents "any acceptable filename"
        return None
    return tuple(values)


def parse(text: str) -> dict[str, dict[str, object]]:
    """Return one spec per documented option."""
    specs: dict[str, dict[str, object]] = {}
    for piece in text.split(r"\anchor")[1:]:
        name = piece.split("\n")[0][5:].strip()
        if "real option" in piece:
            kind = "float"
        elif "integer option" in piece:
            kind = "int"
        elif "string option" in piece:
            kind = "str"
        else:
            print(f"skipping {name}: no kind in the documentation", file=sys.stderr)
            continue
        spec: dict[str, object] = {"kind": kind}
        # Ipopt honors these only when it reads them from an ipopt.opt file, so setting
        # them through the API does nothing; YAPSS refuses them rather than pretend
        if "only works when read from the ipopt.opt" in piece:
            spec["file_only"] = True
        if kind == "str":
            values = parse_values(piece)
            if values is not None:
                spec["values"] = values
        else:
            low, low_inclusive, high, high_inclusive = parse_range(piece, name)
            if low is not None:
                spec["low"], spec["low_inclusive"] = low, low_inclusive
            if high is not None:
                spec["high"], spec["high_inclusive"] = high, high_inclusive
        specs[name] = spec
    # not in the documentation, but accepted: suppresses the Ipopt banner
    specs.setdefault("sb", {"kind": "str", "values": ("yes", "no")})
    return specs


def _literal(value: object) -> str:
    """Render a value with double quotes, as the project's formatting requires."""
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, tuple):
        return "(" + "".join(f'"{v}", ' for v in value) + ")"
    return repr(value)


def render(specs: dict[str, dict[str, object]], ref: str) -> str:
    """Return the source of the generated spec module."""
    version = ref.split("/")[-1]
    lines = [
        '"""',
        "",
        "What Ipopt documents about each of its options: kind, range, allowed settings.",
        "",
        "Generated by ``utils/get_ipopt_options.py`` from Ipopt's ``doc/options.dox``. Do not",
        "edit by hand.",
        "",
        "This table never decides whether Ipopt accepts an option: that depends on the build,",
        "which need not match the documentation this was scraped from, so Ipopt stays the",
        "authority. YAPSS reads an option's kind, to check the type of a value where it is",
        "assigned, and its name, to add a hint when Ipopt refuses it (see",
        "``ipopt_options.refusal_message``). The ranges and settings are kept as reference;",
        "nothing reads them.",
        "",
        f"Scraped from Ipopt {version} (``{ref}``) on "
        f"{datetime.now(tz=timezone.utc).date().isoformat()}.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        f'IPOPT_DOC_VERSION = "{version}"',
        '"""The Ipopt release whose documentation this table describes.',
        "",
        "The Ipopt library actually loaded may be a different version.",
        '"""',
        "",
        "IpoptOptionSpec = dict[str, str | float | bool | tuple[str, ...]]",
        '"""Kind, and either a numeric range or the allowed string values."""',
        "",
        "IPOPT_OPTION_SPECS: dict[str, IpoptOptionSpec] = {",
    ]
    for name in sorted(specs):
        entry = ", ".join(f'"{k}": {_literal(v)}' for k, v in specs[name].items())
        lines.append(f'    "{name}": {{{entry}}},')
    lines += ["}", '"""Documented kind, range, and allowed values of every Ipopt option."""', ""]
    return "\n".join(lines)


def main() -> int:
    """Fetch the documentation and write whichever artifact was asked for."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", action="store_true", help="write the spec module")
    parser.add_argument(
        "--ref",
        default=None,
        help="Ipopt git ref to scrape (default: the newest releases/x.y.z tag)",
    )
    args = parser.parse_args()
    ref = args.ref or latest_release()
    specs = parse(fetch(ref))
    if args.specs:
        SPECS.write_text(render(specs, ref), encoding="utf-8")
        print(f"scraped from {ref}")
        ranged = sum(1 for s in specs.values() if "low" in s or "high" in s)
        listed = sum(1 for s in specs.values() if "values" in s)
        print(f"{len(specs)} options -> {SPECS} ({ranged} with a range, {listed} with values)")
    else:
        # imported here: only the annotations need it, and it needs yapss installed
        from yapss._backend.ipopt_options import RESERVED_IPOPT_OPTIONS  # noqa: PLC0415

        # Left out: the options YAPSS sets itself, so that a type checker reports setting one,
        # and those read only from an ipopt.opt file, which the class's comment explains.
        # `| None` because assigning None removes an option.
        for name in sorted(specs):
            if name in RESERVED_IPOPT_OPTIONS:
                continue
            if specs[name].get("file_only") and name not in SETTABLE_IN_CODE:
                continue
            print(f"    {name}: {specs[name]['kind']} | None")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
