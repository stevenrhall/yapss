"""Registering callbacks through `register`.

The callbacks of a problem and of a phase are gathered under one name rather than sitting
beside the settings, where `objective` and `discrete` already name the aspects carrying the
sense, the scale and the bounds. These tests cover the namespace and the ways of reaching for
it in the wrong place, which are the errors the gathering is meant to answer.
"""

import pytest

from yapss.examples.brachistochrone import setup


@pytest.fixture
def problem():
    return setup()


@pytest.fixture
def phase(problem):
    return problem.phases.slide


def test_the_registry_holds_exactly_the_callbacks(problem, phase):
    assert problem.register._registrations == ("objective", "discrete")
    assert phase.register._registrations == ("continuous",)


def test_a_callback_can_be_registered_by_calling(problem, phase):
    def replacement(arg, out):
        return out

    phase.register.continuous(replacement, replace=True)
    assert phase._continuous is replacement


def test_a_second_callback_is_refused_unless_replacing(phase):
    with pytest.raises(ValueError, match="already has the continuous callback"):
        phase.register.continuous(lambda arg, out: out)


def test_a_registration_is_not_assigned(problem, phase):
    # the released API assigns -- `ocp.functions.continuous = continuous` -- so this is the
    # form a user arriving from 0.3.0 reaches for first
    with pytest.raises(AttributeError, match=r"Decorate the callback with 'register.objective'"):
        problem.register.objective = lambda arg: 0.0
    with pytest.raises(AttributeError, match=r"Decorate the callback with 'register.continuous'"):
        phase.register.continuous = lambda arg, out: out


def test_a_misspelled_registration_is_answered(problem, phase):
    with pytest.raises(AttributeError, match="Did you mean 'objective'"):
        _ = problem.register.objectiv
    with pytest.raises(AttributeError, match="Did you mean 'continuous'"):
        _ = phase.register.continous


def test_reaching_for_a_callback_on_its_owner_says_where_it_lives(phase):
    # a user who knows a phase has a continuous callback may try this before looking
    with pytest.raises(AttributeError, match=r"Did you mean 'register.continuous'"):
        _ = phase.continuous


@pytest.mark.parametrize(
    ("which", "callback"),
    [("objective", lambda arg: 0.0), ("discrete", lambda arg, out: out)],
)
def test_decorating_the_aspect_instead_of_the_registry_is_answered(problem, which, callback):
    # the natural slip: problem.objective is the sense and the scale, not the callback
    aspect = getattr(problem, which)
    with pytest.raises(TypeError, match=rf"Register the callback with '@problem.register.{which}'"):
        aspect(callback)
