"""The atmospheres the Delta III example offers, which teach `yapss.math.external`.

Three of them are one model: an exponential atmosphere traced by ``"auto"``, and the same
exponential hidden behind the wrapper with its derivatives supplied or differenced. They must
agree. The fourth is a library atmosphere, from a package YAPSS does not install.
"""

import inspect
import sys

import pytest

from yapss.examples.delta_iii_ascent import ATMOSPHERES, setup

ICAO_FINAL_MASS = 7478.3726
"""The payload with the ICAO atmosphere, as first solved; a regression value, not a reference."""


def solve(atmosphere):
    problem = setup(atmosphere)
    problem.ipopt_options.print_level = 0
    return problem.solve()


@pytest.fixture(scope="module")
def traced():
    return solve("exponential")


@pytest.mark.parametrize("atmosphere", ["supplied", "differenced"])
def test_the_hidden_exponential_gives_the_traced_answer(atmosphere, traced):
    wrapped = solve(atmosphere)
    assert wrapped.converged
    assert wrapped.objective == pytest.approx(traced.objective, rel=1e-9)


def test_the_library_atmosphere_solves():
    pytest.importorskip("ambiance")
    solution = solve("icao")
    assert solution.converged
    assert solution.objective == pytest.approx(ICAO_FINAL_MASS, rel=1e-7)


def test_the_library_atmosphere_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "ambiance", None)  # as if it were not installed
    with pytest.raises(ImportError, match=r"ambiance package.*pip install ambiance"):
        setup("icao")


def test_an_unknown_atmosphere_is_refused():
    with pytest.raises(ValueError, match="atmosphere must be one of"):
        setup("msis")


def test_the_default_is_the_traced_exponential():
    assert inspect.signature(setup).parameters["atmosphere"].default == "exponential"
    assert ATMOSPHERES == ("exponential", "supplied", "differenced", "icao")
