"""Tests for `yapss._private.config`, which runs when yapss is imported."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.isolation
def test_an_invalid_logging_level_warns_as_a_yapss_warning():
    """The notice is a `YapssWarning`, so the one-line strict mode escalates it too.

    Runs in a subprocess, since the level is read when yapss is first imported.
    """
    script = textwrap.dedent("""
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import yapss
        import logging
        for warning in caught:
            if "Invalid logging level" in str(warning.message):
                print("CATEGORY", issubclass(warning.category, yapss.YapssWarning))
        print("LEVEL", logging.getLogger("yapss").level)
        """)
    process = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env={**os.environ, "YAPSS_LOGGING": "LOUD"},
    )
    output = process.stdout + process.stderr
    assert process.returncode == 0, output
    assert "CATEGORY True" in process.stdout, output
    assert "LEVEL 30" in process.stdout, output  # fell back to WARNING
