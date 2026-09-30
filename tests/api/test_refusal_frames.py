"""A refused assignment shows one YAPSS frame below the user's line, not the checks' route.

The checks a setting goes through run several calls deep, and a traceback that listed them
would put the message four frames below the line that caused it. The setter the user's line
calls raises the refusal again with those frames cut.
"""

import traceback
from pathlib import Path

import numpy as np
import pytest

import yapss
from yapss.examples.brachistochrone import setup

PACKAGE = str(Path(yapss.__file__).resolve().parent)


class Rocket(yapss.State):
    r = yapss.vector(3)


class Phase(yapss.Phase):
    state: Rocket


class Phases(yapss.Phases):
    phase: Phase


class Blocks(yapss.Problem):
    phases: Phases


def below_the_user(error: BaseException) -> list[str]:
    """Return the YAPSS frames after the last frame of this file, by file and function."""
    frames = traceback.extract_tb(error.__traceback__)
    last = max(i for i, frame in enumerate(frames) if frame.filename == __file__)
    return [
        f"{Path(frame.filename).name}:{frame.name}"
        for frame in frames[last + 1 :]
        if str(Path(frame.filename).resolve()).startswith(PACKAGE)
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "ph.state.x.bounds = 5.0",
        "ph.state.x.guess = (0, 1, 2)",
        "ph.control.u.scale = -1.0",
        "ph.time.final = (2, 1)",
        "ph.duration.bounds = (-1, 1)",
        "problem.spectral_method = 'lq'",
    ],
)
def test_a_refused_setting_shows_one_frame(statement):
    problem = setup()
    ph = problem.phases.phase
    with pytest.raises((TypeError, ValueError)) as info:
        exec(statement)  # noqa: S102
    assert len(below_the_user(info.value)) == 1, below_the_user(info.value)


def test_a_refused_row_of_a_block_field_shows_one_frame():
    problem = Blocks("blocks")
    with pytest.raises(TypeError) as info:
        problem.phases.phase.state.r.bounds[0] = 5.0
    assert below_the_user(info.value) == ["vector.py:__setitem__"]


def test_a_refused_output_in_a_callback_shows_one_frame():
    problem = setup()
    problem.ipopt_options.print_level = 0

    def continuous(arg, out):
        out.dynamics.x = np.ones((2, 2))

    problem.phases.phase.register.continuous(continuous)
    with pytest.raises(ValueError, match="a row is a scalar") as info:
        problem.solve()
    assert below_the_user(info.value) == ["vector.py:__setattr__"]
