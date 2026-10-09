"""A `SolutionPhases` survives a copy and a pickle with its phases intact.

The tuple's constructor takes the phases as separate arguments, and a tuple is otherwise rebuilt
from a copy or a pickle by passing its items as one argument, which made ``phase`` a tuple
holding one tuple. Strings stand in for the phases: the container is what is tested.
"""

import copy
import pickle

import pytest

from yapss._backend.solution import SolutionPhases


@pytest.mark.parametrize(
    "rebuild",
    [copy.copy, copy.deepcopy, lambda phases: pickle.loads(pickle.dumps(phases))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_the_phases_of_a_copy_or_a_pickle_are_the_phases(rebuild):
    phases = SolutionPhases("first", "second")
    rebuilt = rebuild(phases)
    assert type(rebuilt) is SolutionPhases
    assert tuple(rebuilt) == ("first", "second")
