"""The dynamics scale YAPSS chooses: None becomes the state's own scale when the problem is compiled.

The transcription never sees None. Each defect row is scaled by the number the user set for it,
or, where the setting is None, by the scale of the state the row belongs to -- row by row, so a
block field can mix the two.
"""

import numpy as np

from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot

from ..contract.api._api import solvable


def test_none_takes_the_states_scale_and_a_number_is_kept():
    problem = solvable()
    ph = problem.phases.slide
    ph.state.x.scale = 1000.0
    ph.state.v.scale = 7.0
    ph.dynamics.v.scale = 3.0
    phase = to_transcription_spec(snapshot(problem)).phases[0]
    np.testing.assert_array_equal(phase.state_scale, [1000.0, 1.0, 7.0])
    np.testing.assert_array_equal(phase.dynamics_scale, [1000.0, 1.0, 3.0])
