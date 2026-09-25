"""The note under an exception from a callback names the user's function, and only it.

An exception raised in a callback propagates unchanged, with a note naming the callback and
the line of its ``def``: the traceback alone does not say which of the user's functions YAPSS
was calling. YAPSS's own functions sit between the transcription and the user's -- the 0.4
front end's adapters always, central difference's wrappers under that method -- and the note
must skip every one of them, which a change anywhere in the call chain can undo without any
other test noticing. So these tests state the whole note, under every derivative method, as
`tests/modules/test_callback_errors.py` does for the 0.3.0 front end.
"""

import math
from pathlib import Path

import pytest

import yapss
from yapss._backend.exceptions import in_yapss

from ..contract.api._api import solvable

METHODS = ["auto", "central-difference", "central-difference-full"]


def _notes(exc: BaseException) -> list[str]:
    return list(getattr(exc, "__notes__", []))


def _location(function) -> str:
    code = function.__code__
    return f"{function.__qualname__} ({code.co_filename}, line {code.co_firstlineno})"


@pytest.mark.parametrize("method", METHODS)
def test_a_continuous_callback_is_named_with_its_phase(method):
    problem = solvable(method)

    def my_dynamics(arg, out):
        raise KeyError("user error")

    problem.phases.slide.register.continuous(my_dynamics)
    with pytest.raises(KeyError, match="user error") as info:
        problem.solve()
    assert _notes(info.value) == [
        f"Raised in the continuous callback for phase 'slide': {_location(my_dynamics)}."
    ]


@pytest.mark.parametrize("method", METHODS)
def test_the_objective_callback_is_named(method):
    problem = solvable(method)

    def my_objective(arg):
        raise KeyError("user error")

    problem.register.objective(my_objective)
    with pytest.raises(KeyError, match="user error") as info:
        problem.solve()
    assert _notes(info.value) == [f"Raised in the objective callback: {_location(my_objective)}."]


@pytest.mark.parametrize("method", METHODS)
def test_the_discrete_callback_is_named(method):
    problem = solvable(method)

    def my_discrete(arg, out):
        raise KeyError("user error")

    problem.register.discrete(my_discrete)
    with pytest.raises(KeyError, match="user error") as info:
        problem.solve()
    assert _notes(info.value) == [f"Raised in the discrete callback: {_location(my_discrete)}."]


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_a_failure_during_the_solve_is_noted_once():
    """Central difference wraps the adapter in YAPSS's own functions; only the user's is named."""
    problem = solvable("central-difference")
    calls = {"n": 0}

    def my_dynamics(arg, out):
        calls["n"] += 1
        if calls["n"] > 20:
            msg = "fails mid-solve"
            raise RuntimeError(msg)
        v, theta = arg.state.v, arg.control.theta
        out.dynamics.x = v * yapss.math.cos(theta)
        out.dynamics.y = v * yapss.math.sin(theta)
        out.dynamics.v = 32.174 * yapss.math.sin(theta)
        out.path.speed = v
        out.integrand.effort = theta**2

    problem.phases.slide.register.continuous(my_dynamics)
    with pytest.raises(RuntimeError, match="fails mid-solve") as info:
        problem.solve()
    assert calls["n"] > 20
    assert _notes(info.value) == [
        f"Raised in the continuous callback for phase 'slide': {_location(my_dynamics)}."
    ]


def test_a_float_only_function_on_the_symbolic_trace_is_pointed_to_yapss_math():
    problem = solvable("auto")

    def my_objective(arg):
        return math.sin(arg[problem.phases.slide].final.time)

    problem.register.objective(my_objective)
    with pytest.raises(TypeError) as info:
        problem.solve()
    first, hint = _notes(info.value)
    assert first == f"Raised in the objective callback: {_location(my_objective)}."
    assert "Use the functions of yapss.math instead." in hint


def test_no_yapss_math_hint_off_the_symbolic_trace():
    problem = solvable("central-difference")

    def my_objective(arg):
        raise TypeError("a type error of the user's own")

    problem.register.objective(my_objective)
    with pytest.raises(TypeError) as info:
        problem.solve()
    assert _notes(info.value) == [f"Raised in the objective callback: {_location(my_objective)}."]


@pytest.mark.parametrize("method", METHODS)
def test_yapss_own_refusal_carries_no_note(method):
    """The front end's own messages name the callback already; a note would point into YAPSS."""
    problem = solvable(method)

    def incomplete(arg, out):
        pass

    problem.register.discrete(incomplete)
    with pytest.raises(ValueError, match="returned without assigning") as info:
        problem.solve()
    assert _notes(info.value) == []


def test_the_line_between_yapss_and_its_user():
    """The package less its examples, which are user code that ships with it."""
    package = Path(yapss.__file__).absolute().parent
    assert in_yapss(str(package / "_api" / "compile.py"))
    assert in_yapss(str(package / "_backend" / "input_args.py"))
    assert in_yapss(str(package / "_legacy" / "problem.py"))
    assert not in_yapss(str(package / "examples" / "brachistochrone.py"))
    assert not in_yapss(str(Path(__file__).absolute()))
