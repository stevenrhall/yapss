"""

The initial guess the transcription starts from, as one flat vector.

The front end holds the user's guess in whatever shape it was given -- constants, ramps,
samples on a grid of their own -- and by the time it reaches here it is a `ProblemSpec`.
All that is left is to interpolate it onto the mesh and lay it out in the order the NLP's
decision variables are in.

This runs before the derivative setup, because the "user" and "central-difference" methods
need a point at which to probe sparsity structure.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d

from .layout import problem_layout
from .structure import DVStructure, get_nlp_dv_structure

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .mesh import Mesh
    from .spec import ProblemSpec

    # Float array
    Array = NDArray[np.float64]


def make_initial_guess_nlp(problem: ProblemSpec, computational_mesh: Mesh) -> Array:
    """Make initial guess for the NLP solution from the user-provided initial guess.

    This method takes the initial guess provided by the user and interpolates to produce an
    initial guess for the NLP solver.

    Returns
    -------
    NDArray
        Initial guess of the NLP decision variable array
    """
    mesh = computational_mesh
    nlp_dv_guess: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)

    # guess for each phase
    for p, phase in enumerate(nlp_dv_guess.phase):
        tau_x = mesh.tau_x[p]
        tau_u = mesh.tau_u[p]
        time = problem.phases[p].guess_time
        t0 = time[0]
        tf = time[-1]
        # tau is defined over the interval [-1, 1], so we need to scale and shift it to the
        # interval [t0, tf]
        t_x = (tf - t0) / 2 * tau_x + (t0 + tf) / 2
        t_u = (tf - t0) / 2 * tau_u + (t0 + tf) / 2

        phase.t0[0] = t0
        phase.tf[0] = tf

        # interpolate state and control variables
        state = problem.phases[p].guess_state
        # tau_x is in time order; the stored order differs under LG, which time_order maps
        time_order = problem_layout(problem)[p].time_order
        for i in range(problem.nx[p]):
            f = interp1d(time, state[i], fill_value="extrapolate")
            phase.x[i][time_order] = f(t_x)
            phase.xs[i][:] = 0.0  # zero modes (empty unless LGL)

        control = problem.phases[p].guess_control
        for i in range(problem.nu[p]):
            f = interp1d(time, control[i], fill_value="extrapolate")
            phase.u[i][:] = f(t_u)

        phase.q[:] = problem.phases[p].guess_integral

    # guess for parameter
    nlp_dv_guess.s[:] = problem.guess_parameter

    return nlp_dv_guess.z
