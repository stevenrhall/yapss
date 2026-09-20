"""Tests for the error-catalogue renderer.

The catalogue is only built by `make error-catalog`, so nothing else would notice if the
renderer broke or if `_harness.raises` stopped feeding it the shape it expects.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("error_catalog", ROOT / "tools" / "error_catalog.py")
assert SPEC and SPEC.loader
error_catalog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(error_catalog)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("expected 3 rows, got 2", "expected 7 rows, got 4"),
        (
            "scale 'state' must be positive, got array([1., 0.])",
            "scale 'state' must be positive, got array([2.])",
        ),
        ("no such attribute 'phses'", "no such attribute 'stat'"),
    ],
)
def test_messages_differing_only_in_a_value_collapse(first, second):
    assert error_catalog.shape(first) == error_catalog.shape(second)


def test_messages_differing_in_wording_do_not_collapse():
    assert error_catalog.shape("expected 3 rows") != error_catalog.shape("wanted 3 rows")


def test_render_groups_by_area_and_type():
    records = [
        {
            "area": "bounds",
            "type": "ValueError",
            "message": "bad bound",
            "statement": "ocp.bounds.phase[0].state.lower = 1",
            "test": "t",
        },
        {
            "area": "guess",
            "type": "TypeError",
            "message": "bad guess",
            "statement": "",
            "test": "u",
        },
    ]
    page = error_catalog.render(records)
    assert page.startswith("Error Catalogue")
    assert "Bounds" in page and "Guess" in page
    # shown as the user sees it: the type on the message line, inside a traceback, which is
    # also what makes Pygments colour the exception name
    assert "ValueError: bad bound" in page
    assert "TypeError: bad guess" in page
    assert page.count("Traceback (most recent call last):") == 2
    assert "ocp.bounds.phase[0].state.lower = 1" in page
    assert "Stated by ``t``" in page


@pytest.mark.parametrize(
    ("statement", "shown"),
    [
        ("ocp.scale.phase[0].state = [1.0, 0.0]", True),
        ("Problem(name='x', nx=[1])  # type: ignore[call-arg]", True),
        ("setattr(ocp, attribute, None)", False),
        ('exec(statement, {"arg": arg})  # noqa: S102', False),
        (").solve()", False),
        ("", False),
    ],
)
def test_only_statements_a_user_might_have_written_are_shown(statement, shown):
    """Parametrized scaffolding in the traceback would be noise, not an example."""
    assert bool(error_catalog.as_user_wrote_it(statement)) is shown


def test_a_type_ignore_comment_is_stripped_from_the_statement():
    assert error_catalog.as_user_wrote_it("Problem('n', [1])  # type: ignore[misc]") == (
        "Problem('n', [1])"
    )


def test_the_contract_helper_still_feeds_the_expected_fields(tmp_path, monkeypatch):
    """A field renamed in `_harness.raises` would silently empty the catalogue.

    Three of the fields are not read by this renderer at all. `front` separates the two
    suites, so one run cannot file one front end's messages under the other; `kind` tells a
    warning from an error; and `site` is the line of YAPSS that produced the message, which
    is what makes completeness measurable against the raises in the package. They are
    checked here because this is the test that notices when the record changes shape.
    """
    catalog = tmp_path / "errors.jsonl"
    monkeypatch.setenv("YAPSS_ERROR_CATALOG", str(catalog))
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:randomly",
            "-x",
            "tests/contract/test_contract_scale.py::test_bad_factor_in_a_whole_assignment_raises_at_the_assignment",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert catalog.exists(), result.stdout + result.stderr
    entry = json.loads(catalog.read_text().splitlines()[0])
    assert set(entry) == {
        "front",
        "area",
        "kind",
        "type",
        "message",
        "statement",
        "test",
        "site",
    }
    assert entry["area"] == "scale"
    assert entry["front"] == "legacy"
    assert entry["kind"] == "error"
    assert entry["site"].startswith("_legacy/")
