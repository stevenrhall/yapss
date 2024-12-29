"""

Module to encapsulate the problem solution in a Solution object.

"""

# standard library imports
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from scipy.sparse import csr_matrix

# package imports
from .structure import CFStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

    import yapss

    from .nlp import NLP

__all__ = ["NLPInfo", "Solution", "SolutionPhase", "SolutionPhases", "make_solution_object"]

# ipopt status message
ipopt_status_messages = {
    0: "Optimal Solution Found",
    1: "Solved To Acceptable Level.",
    2: "Converged to a point of local infeasibility. Problem may be infeasible.",
    3: "Search Direction is becoming Too Small",
    4: "Iterates diverging; problem might be unbounded.",
    5: "Stopping optimization at current point as requested by user.",
    6: "Feasible point for square problem found.",
    -1: "Maximum Number of Iterations Exceeded.",
    -2: "Restoration Failed!",
    -3: "Error in step computation!",
    -4: "Maximum CPU time exceeded.",
    -5: "Maximum wallclock time exceeded.",
    -10: "Problem has too few degrees of freedom.",
    -11: "Problem has inconsistent variable bounds or constraint sides.",
    -12: "Invalid_Option (Details about the particular error will be output to the console.)",
    -13: "Invalid number in NLP function or derivative detected.",
    -100: (
        "Unrecoverable_Exception (Details about the particular error will be output to "
        "the console.)"
    ),
    -101: "Unknown Exception caught in Ipopt",
    -102: "Not enough memory.",
    -199: "INTERNAL ERROR: Unknown SolverReturn value - Notify IPOPT Authors.",
}

_dataclass_msg = "All attributes must be provided, and cannot be None"


def make_solution_object(
    problem: yapss.Problem,
    mesh: yapss._private.mesh.Mesh,
    nlp_temp: NLP,
    nlp_info: dict[str, NDArray[np.float64] | float | int | bytes],
) -> Solution:
    """Extract the optimal control solution from the NLP solver output.

    This function extracts the optimal control solution from the NLP solver output and
    assembles it into a Solution object.

    Parameters
    ----------
    problem : yapss.Problem
    mesh : Mesh
    nlp_temp : NLP
    nlp_info : dict

    Returns
    -------
    Solution
    """
    # extract data from nlp_info
    g_value_ = nlp_info["g"]
    mult_g_ = nlp_info["mult_g"]
    mult_x_l_ = nlp_info["mult_x_L"]
    mult_x_u_ = nlp_info["mult_x_U"]
    status_ = nlp_info["status"]
    x_ = nlp_info["x"]
    objective_object = nlp_info["obj_val"]

    # validate data types
    if isinstance(g_value_, np.ndarray):
        g_value = g_value_
    else:
        msg = f"Expected 'g' to be np.ndarray, got type {type(g_value_)}"
        raise TypeError(msg)

    if isinstance(x_, np.ndarray):
        x = x_
    else:
        msg = f"Expected 'x' to be np.ndarray, got type {type(x_)}"
        raise TypeError(msg)

    if isinstance(mult_g_, np.ndarray):
        mult_g = mult_g_
    else:
        msg = f"Expected 'mult_g' to be np.ndarray, got type{type(mult_g_)}"
        raise TypeError(msg)

    if isinstance(mult_x_l_, np.ndarray):
        mult_x_l = mult_x_l_
    else:
        msg = f"Expected 'mult_x_L' to be np.ndarray, got type {type(mult_x_l_)}"
        raise TypeError(msg)

    if isinstance(mult_x_u_, np.ndarray):
        mult_x_u = mult_x_u_
    else:
        msg = f"Expected 'mult_x_U' to be np.ndarray, got type {type(mult_x_u_)}"
        raise TypeError(msg)

    if isinstance(status_, int):
        status = status_
    else:
        msg = "Expected int, got {type(status_)}"
        raise TypeError(msg)

    status_message: str = ipopt_status_messages.get(status, "Unknown status code")

    # initialize data views
    from .structure import DVStructure

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv.z[:] = x
    dv_multiplier: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv_multiplier.z[:] = mult_x_u - mult_x_l

    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    cf.c[:] = g_value
    cf_multiplier: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    cf_multiplier.c[:] = mult_g

    # Form the SolutionPhase objects for each phase
    solution_phases = []

    for p in range(problem.np):
        z_phase = dv.phase[p]

        # time vector
        initial_time = t0 = float(dv.phase[p].t0[0])
        final_time = tf = float(dv.phase[p].tf[0])
        time = (mesh.tau_x[p] + 1) * (tf - t0) * 0.5 + t0
        time_c = (mesh.tau_u[p] + 1) * (tf - t0) * 0.5 + t0

        # state
        nx = problem.nx[p]
        state_list = []
        for i in range(nx):
            if problem.spectral_method == "lg":
                state_list.append(z_phase.xa[i][mesh.lg_index[p]])
            else:
                state_list.append(z_phase.x[i])
        state = np.array(state_list, dtype=np.float64)

        # control
        control = np.array(dv.phase[p].u, dtype=np.float64)

        # costate
        c_phase = cf_multiplier.phase[p]
        if problem.spectral_method == "lgl":
            # for sparse matrix necessary to calculate costate from multipliers
            row = c_phase.defect_index
            n = len(row)
            col = list(range(n))
            mat = csr_matrix((n * [1], (row, col)))
        else:
            mat = 1.0
        nx = problem.nx[p]
        nh = problem.nh[p]

        # continuous multipliers

        control_multiplier = np.array(
            [z_phase.u[i] for i in range(problem.nu[p])],
            dtype=np.float64,
        )
        control_multiplier *= (tf - t0) / 2
        costate = np.array(
            [(mat * c_phase.defect[i]) / mesh.w[p] for i in range(nx)],
            dtype=np.float64,
        )
        path_multiplier = np.array(
            [c_phase.path[i] / mesh.w[p] for i in range(nh)],
            dtype=np.float64,
        )
        integral_multiplier = np.array(
            [c_phase.integral[i] for i in range(problem.nq[p])],
            dtype=np.float64,
        )

        # values calculated from the continuous function
        c_arg = nlp_temp.eval_continuous(dv.z, 0)
        dynamics = np.array(c_arg.phase[p].dynamics)
        integrand = np.array(c_arg.phase[p].integrand)
        path = np.array(c_arg.phase[p].path)

        # calculate the Hamiltonian
        hamiltonian = (integrand * integral_multiplier[:, np.newaxis]).sum(axis=0)
        if dynamics.shape[0] > 0:
            hamiltonian += (costate * dynamics).sum(axis=0)

        duration = final_time - initial_time
        duration_multiplier = float(cf_multiplier.phase[p].duration[0])

        solution_phase = SolutionPhase(
            index=p,
            initial_time=initial_time,
            initial_time_multiplier=float(dv_multiplier.phase[p].t0[0]),
            final_time=final_time,
            final_time_multiplier=float(dv_multiplier.phase[p].tf[0]),
            time=time,
            time_c=time_c,
            duration=duration,
            duration_multiplier=duration_multiplier,
            state=state,
            control=control,
            control_multiplier=control_multiplier,
            dynamics=dynamics,
            costate=costate,
            path=path,
            integrand=integrand,
            path_multiplier=path_multiplier,
            integral=dv.phase[p].q,
            integral_multiplier=integral_multiplier,
            hamiltonian=hamiltonian,
        )

        solution_phases.append(solution_phase)

    objective = float(objective_object)
    parameter = dv.s

    discrete = cf.discrete

    nlp = NLPInfo(
        g=g_value,
        obj_val=objective,
        mult_g=mult_g,
        mult_x_L=mult_x_l,
        mult_x_U=mult_x_u,
        ipopt_status=status,
        ipopt_status_message=status_message,
        x=x,
    )

    return Solution(
        name=problem.name,
        problem=deepcopy(problem),
        objective=objective,
        discrete=discrete,
        discrete_multiplier=cf_multiplier.discrete,
        parameter=parameter,
        parameter_multiplier=dv_multiplier.s,
        phase=SolutionPhases(*solution_phases),
        nlp_info=nlp,
    )


class SolutionPhases(tuple["SolutionPhase", ...]):
    """Container for a sequence of SolutionPhase objects."""

    __slots__ = ()

    def __new__(cls, *args: SolutionPhase) -> Self:
        return super().__new__(cls, args)

    def __repr__(self) -> str:
        n = len(self)
        return f"<{__name__}.SolutionPhases: {n} phase{'' if n == 1 else 's'}>"


@dataclass(frozen=True)
class SolutionPhase:
    """Container for the solution of a single phase of an optimal control problem."""

    index: int
    time: NDArray[np.float64]
    time_c: NDArray[np.float64]
    initial_time: float
    initial_time_multiplier: float
    final_time: float
    final_time_multiplier: float
    duration: float
    duration_multiplier: float
    state: NDArray[np.float64]
    control: NDArray[np.float64]
    control_multiplier: NDArray[np.float64]
    path: NDArray[np.float64]
    path_multiplier: NDArray[np.float64]
    dynamics: NDArray[np.float64]
    costate: NDArray[np.float64]
    integrand: NDArray[np.float64]
    integral: NDArray[np.float64]
    integral_multiplier: NDArray[np.float64]
    hamiltonian: NDArray[np.float64]

    def __post_init__(self) -> None:
        attributes = [
            self.time,
            self.time_c,
            self.state,
            self.control,
            self.control_multiplier,
            self.path,
            self.path_multiplier,
            self.dynamics,
            self.costate,
            self.integrand,
            self.integral,
            self.integral_multiplier,
            self.hamiltonian,
        ]
        if any(attr is None for attr in attributes):
            raise ValueError(_dataclass_msg)

    def __repr__(self) -> str:
        return f"<{__name__}.SolutionPhase: phase index p = {self.index}>"


@dataclass(frozen=True)
class Solution:
    """Represents the solution of an optimal control problem.

    A Solution object encapsulates the solution to an optimal control problem, including
    the optimal state, control, and parameter decision variables, the objective value,
    Lagrange multipliers, and related data.

    Attributes
    ----------
    name : str
        The name of the problem being solved.
    problem : Problem
        Deep copy of the problem definition object that includes all original user-defined
        settings, parameters, and configurations for the NLP problem.
    objective : float
        The optimal value of the objective function after solving the optimal control problem.
    discrete_multiplier : numpy.ndarray
        An array of Lagrange multipliers associated with the discrete constraints.
    parameter : numpy.ndarray
        An array of the optimal parameter values.
    parameter_multiplier : numpy.ndarray
        An array of Lagrange multipliers associated with the parameter bounds.
    phase : SolutionPhases
        A sequence of ``SolutionPhase`` objects, each corresponding to a phase of the problem.
        Each phase contains time, state, control, and additional detailed results:

        index : int
            The index of the phase.
        initial_time : float
            Initial time of the phase.
        initial_time_multiplier : float
            Lagrange multiplier for the initial time bound.
        final_time : float
            Final time of the phase.
        final_time_multiplier : float
            Lagrange multiplier for the final time bound.
        duration : float
            Duration of the phase.
        duration_multiplier : float
            Lagrange multiplier for the duration bound.
        time : numpy.ndarray
            Array of interpolation time points for the phase.
        time_c : numpy.ndarray
            Array of collocation time points for the phase.
        state : numpy.ndarray
            State variable values at the interpolation time points.
        control : numpy.ndarray
            Control variable values at the collocation time points.
        control_multiplier : numpy.ndarray
            Lagrange multipliers for control variable bounds.
        path : numpy.ndarray
            Path constraints values at the collocation time points.
        path_multiplier : numpy.ndarray
            Lagrange multipliers for path constraints.
        dynamics : numpy.ndarray
            Dynamics function evaluated at the collocation time points.
        costate : numpy.ndarray
            Costate values at the collocation time points.
        integrand : numpy.ndarray
            Integrand values at the collocation time points, used to evaluate the integrals
            used in the discrete constraints or objective function.
        integral : numpy.ndarray
            Array of integral values for the phase.
        integral_multiplier : numpy.ndarray
            Lagrange multipliers for the integral constraints.
        hamiltonian : numpy.ndarray
            Hamiltonian values at the collocation time points.

    nlp_info : NLPInfo
        Information about the NLP solver status and results, including the final values of
        decision variables, constraint multipliers, and other information. The attributes are:

        ipopt_status : int
            The status code returned by Ipopt, indicating the success or failure of the
            optimization.
        ipopt_status_message : str
            A human-readable message corresponding to the status code. In most cases, it's
            the same message as that printed in the console by Ipopt.
        objective : float
            The optimal value of the objective function after solving the optimal control
            problem.
        x : numpy.ndarray
            The optimal values of the NLP decision variables.
        mult_x_L : numpy.ndarray
            Lagrange multipliers associated with the lower bounds of the decision variables.
        mult_x_U : numpy.ndarray
            Lagrange multipliers associated with the upper bounds of the decision variables.
        mult_g : numpy.ndarray
            Lagrange multipliers associated with the constraints.
    """

    # TODO: Add integral to SolutionPhase

    name: str
    problem: yapss.Problem
    objective: float
    parameter: NDArray[np.float64]
    parameter_multiplier: NDArray[np.float64]
    discrete: NDArray[np.float64]
    discrete_multiplier: NDArray[np.float64]
    phase: SolutionPhases
    nlp_info: NLPInfo

    def __post_init__(self) -> None:
        attributes = [
            self.name,
            self.problem,
            self.objective,
            self.parameter,
            self.parameter_multiplier,
            self.discrete,
            self.discrete_multiplier,
            self.phase,
            self.nlp_info,
        ]
        if any(attr is None for attr in attributes):
            raise ValueError(_dataclass_msg)

    def __repr__(self) -> str:
        return f"<{__name__}.Solution: '{self.name}'>"

    def __str__(self) -> str:
        return (
            f"<{__name__}.Solution> object\n"
            f"    Name: {self.name}\n"
            f"    Ipopt Status Code: {self.nlp_info.ipopt_status}\n"
            f"    Status Message: {self.nlp_info.ipopt_status_message}\n"
            f"    Objective Value: {self.objective}"
        )


@dataclass(frozen=True)
class NLPInfo:
    """Container for NLP solver information."""

    ipopt_status: int
    ipopt_status_message: str
    g: NDArray[np.float64]
    obj_val: float
    x: NDArray[np.float64]
    mult_x_L: NDArray[np.float64]  # noqa: N815
    mult_x_U: NDArray[np.float64]  # noqa: N815
    mult_g: NDArray[np.float64]

    def __post_init__(self) -> None:
        attributes = [
            self.ipopt_status,
            self.ipopt_status_message,
            self.g,
            self.obj_val,
            self.x,
            self.mult_x_L,
            self.mult_x_U,
            self.mult_g,
        ]
        if any(attr is None for attr in attributes):
            raise ValueError(_dataclass_msg)

    def __repr__(self) -> str:
        return f"<{__name__}.NLPInfo: ipopt_status = {self.ipopt_status}>"
