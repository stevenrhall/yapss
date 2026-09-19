"""

What the transcription is given for bounds: two pairs of flat arrays.

The front end holds the user's bounds in whatever shape suits the way they were written;
by the time they reach here they are a `ProblemSpec`, and all that is left is to lay them
out in the order the NLP's decision variables and constraint functions are in.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import float64

from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .spec import ProblemSpec

    FloatArray = NDArray[float64]


def get_nlp_decision_variable_bounds(problem: ProblemSpec) -> tuple[FloatArray, FloatArray]:
    """Determine the upper and lower bounds on the NLP decision variables.

    Function to determine the upper and lower bounds on the NLP decision variables based
    on the upper and  lower bounds in the optimal control problem statement.

    Parameters
    ----------
    problem : ProblemSpec
        The user-defined optimal control problem

    Returns
    -------
    tuple[NDArray, NDArray]
        The upper and lower bounds on the NLP decisionv variables. The length of each is
        the same as the number of decision variables.
    """
    # make structure that allows easy translation from problem statement bounds to NLP
    # bounds

    lb: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    ub: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)

    # do for each phase
    for p in range(problem.np):
        self_phase = problem.phases[p]

        # state bounds at every time point, and zero-mode bounds (empty unless LGL)
        for i in range(problem.nx[p]):
            lb.phase[p].x[i][:] = self_phase.state_lower[i]
            ub.phase[p].x[i][:] = self_phase.state_upper[i]
            lb.phase[p].xs[i][:] = self_phase.zero_mode_lower[i]
            ub.phase[p].xs[i][:] = self_phase.zero_mode_upper[i]

        # overwrite boundary value bounds
        lb.phase[p].x0[:] = np.maximum(self_phase.initial_state_lower, self_phase.state_lower)
        ub.phase[p].x0[:] = np.minimum(self_phase.initial_state_upper, self_phase.state_upper)
        lb.phase[p].xf[:] = np.maximum(self_phase.final_state_lower, self_phase.state_lower)
        ub.phase[p].xf[:] = np.minimum(self_phase.final_state_upper, self_phase.state_upper)

        # control bounds
        for i in range(problem.nu[p]):
            lb.phase[p].u[i][:] = self_phase.control_lower[i]
            ub.phase[p].u[i][:] = self_phase.control_upper[i]

        # integral bounds
        lb.phase[p].q[:] = self_phase.integral_lower
        ub.phase[p].q[:] = self_phase.integral_upper

        # boundary time bounds
        lb.phase[p].t0[:] = self_phase.initial_time_lower
        ub.phase[p].t0[:] = self_phase.initial_time_upper
        lb.phase[p].tf[:] = self_phase.final_time_lower
        ub.phase[p].tf[:] = self_phase.final_time_upper

    # parameter bounds
    lb.s[:] = problem.parameter_lower
    ub.s[:] = problem.parameter_upper

    return ub.z, lb.z


def get_nlp_constraint_function_bounds(
    problem: ProblemSpec,
) -> tuple[FloatArray, FloatArray]:
    """Determine the upper and lower bounds on the NLP decision variables.

    Parameters
    ----------
    problem : ProblemSpec
        The user-defined optimal control problem

    Returns
    -------
    tuple[NDArray, NDArray]
    """
    lb: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    ub: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)

    for p in range(problem.np):
        # state equation defect
        for i in range(problem.nx[p]):
            lb.phase[p].defect[i][:] = 0.0
            ub.phase[p].defect[i][:] = 0.0

        # integral equation defect
        lb.phase[p].integral[:] = 0.0
        ub.phase[p].integral[:] = 0.0

        # path
        for i in range(problem.nh[p]):
            lb.phase[p].path[i][:] = problem.phases[p].path_lower[i]
            ub.phase[p].path[i][:] = problem.phases[p].path_upper[i]

        # duration
        lb.phase[p].duration[:] = problem.phases[p].duration_lower
        ub.phase[p].duration[:] = problem.phases[p].duration_upper

    # discrete constraints
    lb.discrete[:] = problem.discrete_lower
    ub.discrete[:] = problem.discrete_upper

    return ub.c, lb.c
