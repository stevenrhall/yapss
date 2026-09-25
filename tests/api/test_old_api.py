"""Code written for YAPSS 0.3 or earlier is recognized, and told what happened.

Every release from 0.1.0 to 0.3.0 declared a problem as ``Problem(name=..., nx=...)``, with
``nx`` required, and exported eight callback-argument classes that 0.4 does not have. Old code
meets one of the two before anything else fails, so both answer with one message: that the
code belongs to the old API, how to keep it running, and where to start porting it.
"""

import importlib
import inspect

import pytest

import yapss
from yapss._api import old_api
from yapss._api.old_api import CURRENT_DOCS, OLD_API_DOCS, OLD_ROOT_NAMES


def assert_is_the_old_api_message(message: str, what: str) -> None:
    assert message.startswith(f"{what} is the API of YAPSS 0.3 and earlier")
    assert 'install "yapss<0.4"' in message
    assert OLD_API_DOCS in message
    # the current documentation's address is a prefix of the old one's, so match its sentence
    assert f"documentation for the current release at {CURRENT_DOCS}" in message


def test_an_old_problem_call_is_recognized():
    with pytest.raises(TypeError) as info:
        yapss.Problem(name="Brachistochrone", nx=[3], nu=[1])
    assert_is_the_old_api_message(str(info.value), "Problem(name=..., nx=...)")


def test_nx_alone_is_enough():
    """`nx` was required in every old release, so it identifies the call by itself."""
    with pytest.raises(TypeError, match="API of YAPSS 0.3 and earlier"):
        yapss.Problem(name="x", nx=[1])


def test_a_new_problem_call_is_untouched():
    class Parameter(yapss.Parameter):
        x = yapss.scalar()

    problem = yapss.Problem("hs", parameter=Parameter)
    assert problem.name == "hs"


def test_the_signature_is_the_constructors():
    """The hidden __new__ must not become the signature a notebook's help shows."""
    parameters = inspect.signature(yapss.Problem).parameters
    assert list(parameters) == ["name", "phases", "discrete", "parameter"]


@pytest.mark.parametrize("name", sorted(OLD_ROOT_NAMES))
def test_an_old_argument_class_is_recognized(name):
    with pytest.raises(ImportError) as info:
        getattr(yapss, name)
    assert_is_the_old_api_message(str(info.value), f"yapss.{name}")


def test_the_message_survives_a_from_import():
    """A from-import replaces an AttributeError's message; an ImportError keeps it."""
    with pytest.raises(ImportError, match="API of YAPSS 0.3 and earlier"):
        exec("from yapss import ObjectiveArg", {})


def test_the_message_names_conda_under_conda(monkeypatch):
    config = importlib.import_module("yapss._backend.config")
    monkeypatch.setattr(config, "get_conda_prefix", lambda: "/opt/conda")
    assert 'conda install "yapss<0.4"' in old_api.old_api_message("x")
    monkeypatch.setattr(config, "get_conda_prefix", lambda: None)
    assert 'pip install "yapss<0.4"' in old_api.old_api_message("x")


def test_an_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'Problm'"):
        _ = yapss.Problm  # type: ignore[attr-defined]
