"""A failure to load Ipopt names the page that explains why YAPSS checks what it loads.

The three errors are the vendored mseipopt's, whose messages know nothing of YAPSS's
documentation, so YAPSS adds the link as a note where it calls `initialize_ipopt`. A failed load
is retained and re-raised as the same object on every later solve, so the note is added once.
"""

import pytest

from yapss._backend import solver
from yapss._backend.config import docs_url
from yapss._backend.mseipopt.library import (
    DuplicateIpoptLibraryError,
    IpoptAbiError,
    IpoptLibraryNotFoundError,
)
from yapss.examples.brachistochrone_minimal import setup

PAGE = "reference/ipopt_backend.html"


@pytest.mark.parametrize(
    "error", [DuplicateIpoptLibraryError, IpoptAbiError, IpoptLibraryNotFoundError]
)
def test_a_load_failure_links_the_page_once_however_often_it_is_raised(monkeypatch, error):
    retained = error("the vendored package's own message")

    def fail():
        raise retained

    monkeypatch.setattr(solver, "initialize_ipopt", fail)
    problem = setup()
    for _ in range(2):
        with pytest.raises(error) as info:
            problem.solve()
    assert info.value is retained
    notes = [note for note in info.value.__notes__ if PAGE in note]
    assert notes == [f"Read more about how YAPSS connects to Ipopt: {docs_url(PAGE)}"]


@pytest.mark.parametrize(
    ("installed", "version"),
    [
        ("0.4.0", "v0.4.0"),
        ("1.10.2", "v1.10.2"),
        ("0.4.0rc1", "stable"),
        ("0.4.1.dev3+g1234abc", "stable"),
        ("0.4.0+local", "stable"),
        ("", "stable"),
    ],
)
def test_a_release_links_its_own_docs_and_anything_else_the_newest(installed, version):
    """A release has docs under its tag; a development build or a pre-release has none."""
    url = docs_url(PAGE, installed)
    assert url == f"https://yapss.readthedocs.io/en/{version}/{PAGE}"
