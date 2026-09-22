"""

Module to encapsulate the problem solution in a Solution object.

"""

# standard library imports
from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from warnings import warn

# third party imports
import numpy as np
from scipy.sparse import csr_matrix

# package imports
from .exceptions import YapssWarning
from .ipopt_status import IpoptStatus
from .layout import problem_layout
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

    from .mesh import Mesh
    from .nlp import NLP
    from .spec import ProblemSpec

__all__ = [
    "IpoptConvergenceWarning",
    "NLPInfo",
    "Solution",
    "SolutionPhase",
    "SolutionPhases",
    "make_solution_object",
    "warn_if_not_converged",
]


class IpoptConvergenceWarning(YapssWarning):
    """Ipopt did not report a converged solution.

    A `Solution` is returned whatever Ipopt reports, so an unconverged run yields a
    plausible-looking trajectory that satisfies nothing in particular. This warning
    exists so that outcome is not silent.

    Public, so it can be filtered or escalated::

        warnings.filterwarnings("error", category=yapss.IpoptConvergenceWarning)

    or with every other YAPSS warning, as ``yapss.YapssWarning``.
    """


# The statuses that do not warn are the converged ones, 0, 1, and 6. 1 is deliberately
# included: it is the normal outcome when pushing tolerances hard -- the answer is routinely
# correct to many more digits than requested -- and warning on it would train users to ignore
# the warning, which destroys its value for the cases that matter. CasADi raises on 1, and it
# is a known annoyance.
QUIET_IPOPT_STATUSES = frozenset(status for status in IpoptStatus if status.converged)

_dataclass_msg = "All attributes must be provided, and cannot be None"


def warn_if_not_converged(solution: Solution, stacklevel: int = 2) -> None:
    """Emit `IpoptConvergenceWarning` unless Ipopt reported a converged solution.

    Called from `Problem.solve` rather than from `make_solution_object`, so that the
    default `stacklevel` points at the caller's own `solve()` rather than at YAPSS
    internals.

    Parameters
    ----------
    solution : Solution
        The solution just constructed.
    stacklevel : int, default=2
        Passed through to `warnings.warn`.

    Every unconverged solve warns, even when several are run from one line. Python's
    "default" and "once" filter actions would otherwise report only the first, and a solve
    that quietly returns a non-optimal trajectory is exactly what this warning exists to
    prevent: see `_forget_previous_warning`.
    """
    status = solution.nlp_info.ipopt_status
    if status in QUIET_IPOPT_STATUSES:
        return

    # The Ipopt message is quoted, and on its own line, for two reasons: it is Ipopt's
    # wording rather than YAPSS's, which is not otherwise apparent to a reader; and its
    # punctuation varies, so interpolating it mid-sentence produced run-ons. The
    # explanation is left as one paragraph rather than hard-wrapped, so that it wraps
    # to the reader's terminal instead of to a width guessed here.
    # not `status in IpoptStatus`: on Python 3.11 an int in an enum class raises TypeError
    try:
        message = IpoptStatus(status).message
    except ValueError:
        message = "Unknown status code."
    _forget_previous_warning(stacklevel)
    warn(
        f'Ipopt did not converge. Status {status}: "{message}"\n'
        f"The returned solution does not satisfy Ipopt's convergence criteria and "
        f"should not be treated as an optimal trajectory. Check "
        f"solution.status and the Ipopt output before using these "
        f"results.",
        category=IpoptConvergenceWarning,
        stacklevel=stacklevel,
    )


def _forget_previous_warning(stacklevel: int) -> None:
    """Let the warning at `stacklevel` be reported again, whatever it reported before.

    Python records each (message, category, lineno) it has reported in the calling module's
    ``__warningregistry__`` and, under the "default" and "once" actions, stays silent for the
    rest of the process. That is right for a deprecation notice and wrong here: every solve
    that does not converge must say so, including the tenth in a loop. Only this module's own
    bookkeeping for the calling frame is cleared, so the user's filters -- "ignore", "error",
    and any narrower rule -- still decide what happens to the warning itself.

    Until 0.3.0 this happened by accident: `solver.solve` wrapped the Ipopt call in
    `warnings.catch_warnings()`, whose exit invalidates the registry of *every* module, so no
    warning anywhere in the process deduplicated after a solve.
    """
    frame = sys._getframe(stacklevel)  # the caller owns the registry this warning lands in
    registry = frame.f_globals.get("__warningregistry__")
    if registry:
        for key in [key for key in registry if key != "version"]:
            if isinstance(key, tuple) and len(key) > 1 and key[1] is IpoptConvergenceWarning:
                del registry[key]


def _rows(rows: list[Any], n_points: int) -> NDArray[np.float64]:
    """Stack per-variable rows into a ``(n_rows, n_points)`` array, ``n_rows == 0`` included.

    ``np.array([])`` has shape ``(0,)``, which loses the point count and breaks every
    consumer that indexes axis 1, ``Guess.from_solution`` among them. A phase with no
    controls or no states is legitimate (a coast phase, a parameter-only phase).
    """
    if not rows:
        return np.empty((0, n_points), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def make_solution_object(
    problem: ProblemSpec,
    mesh: Mesh,
    nlp_temp: NLP,
    nlp_info: dict[str, Any],
    origin: Any = None,
) -> Solution:
    """Extract the optimal control solution from the NLP solver output.

    This function extracts the optimal control solution from the NLP solver output and
    assembles it into a Solution object.

    Parameters
    ----------
    problem : ProblemSpec
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
        msg = f"Expected 'mult_g' to be np.ndarray, got type {type(mult_g_)}"
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
        status = IpoptStatus(status_)
    else:
        msg = f"Expected int, got {type(status_)}"
        raise TypeError(msg)

    status_message: str = status.message

    # initialize data views
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
        time_order = problem_layout(problem)[p].time_order
        state = _rows([z_phase.x[i][time_order] for i in range(nx)], len(time))

        # control
        control = _rows(list(dv.phase[p].u), len(time_c))

        # costate
        c_phase = cf_multiplier.phase[p]
        # gather each defect multiplier onto the evaluation point its defect reads; under
        # LGL, segment-boundary points collect two
        layout = problem_layout(problem)[p]
        n_defect = layout.n_collocation
        mat = csr_matrix(
            (np.ones(n_defect), (layout.defect_index, np.arange(n_defect))),
            shape=(layout.n_eval, n_defect),
        )
        nx = problem.nx[p]
        nh = problem.nh[p]

        # continuous multipliers, in the time domain. A time integral is transcribed as
        # sum_k h w_k (.)_k with h = (tf - t0) / 2, so a multiplier on a per-point row or
        # bound that carries no h of its own (a control bound, a path row) is a density
        # in tau and must be divided by h w_k to be a density in t. The defect rows carry
        # h already (D x - h f = 0), so the costate needs only w_k. On a zero-duration
        # phase the continuous multipliers are undefined: the constraint holds on a set
        # of measure zero, and NaN is the honest value. Set explicitly, not left to 0/0:
        # the NLP multipliers there need not be zero, and x/0 is +/-inf.
        half_duration = (tf - t0) / 2
        n_points = len(mesh.w[p])
        control_multiplier: NDArray[np.float64]
        path_multiplier: NDArray[np.float64]
        if half_duration == 0:
            control_multiplier = np.full((problem.nu[p], n_points), np.nan)
            path_multiplier = np.full((nh, n_points), np.nan)
        else:
            control_multiplier = _rows(
                [
                    dv_multiplier.phase[p].u[i] / (half_duration * mesh.w[p])
                    for i in range(problem.nu[p])
                ],
                n_points,
            )
            path_multiplier = _rows(
                [c_phase.path[i] / (half_duration * mesh.w[p]) for i in range(nh)],
                n_points,
            )
        costate = _rows([(mat * c_phase.defect[i]) / mesh.w[p] for i in range(nx)], n_points)
        integral_multiplier = np.array(
            [c_phase.integral[i] for i in range(problem.nq[p])],
            dtype=np.float64,
        )

        # values calculated from the continuous function
        c_arg = nlp_temp.eval_continuous(dv.z, 0)
        dynamics = np.array(c_arg.phase[p].dynamics)
        integrand = np.array(c_arg.phase[p].integrand)
        path = np.array(c_arg.phase[p].path)

        # calculate the Hamiltonian, lambda.f + nu.g. The path constraints contribute no
        # term: the augmented Hamiltonian adds mu_h.(h - h_bound), which complementary
        # slackness makes zero on-shell, so the two agree in value along the solution --
        # though not in derivative. The class docstring says so where a user will read it.
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
        **{name: nlp_info[name] for name in _NLP_INPUTS},
    )

    return Solution(
        name=problem.name,
        problem=deepcopy(problem if origin is None else origin),
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

    @property
    def initial_state(self) -> NDArray[np.float64]:
        """Initial state of the phase, ``state[:, 0]``, as a copy."""
        return self.state[:, 0].copy()

    @property
    def final_state(self) -> NDArray[np.float64]:
        """Final state of the phase, ``state[:, -1]``, as a copy."""
        return self.state[:, -1].copy()

    def __repr__(self) -> str:
        return f"<{__name__}.SolutionPhase: phase index p = {self.index}>"


@dataclass(frozen=True)
class Solution:
    r"""Represents the solution of an optimal control problem.

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
    discrete : numpy.ndarray
        An array of the discrete constraint function values.
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
        initial_state, final_state : numpy.ndarray
            Initial and final state of the phase, ``state[:, 0]`` and ``state[:, -1]``,
            returned as copies.
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
            Hamiltonian values at the collocation time points,
            :math:`\mathcal{H} = \lambda^T f + \nu^T g`, where :math:`f` is the
            dynamics and :math:`g` the integrand. There is no path-constraint term, by
            construction rather than by omission: written in the standard form
            :math:`h - h_\text{bound} \le 0`, the augmented Hamiltonian's term
            :math:`\mu_h^T (h - h_\text{bound})` is zero on-shell by complementary
            slackness, so the value reported here is the augmented Hamiltonian's.
            Its *derivative* is not: stationarity in the control still carries the
            path multiplier.

    status : IpoptStatus
        The status Ipopt reported, an `IntEnum` that compares equal to Ipopt's integer code.
        A status with no iterate to report raises from `Problem.solve` instead.
    converged : bool
        Whether Ipopt reported a converged solution: status 0, 1, or 6.
    nlp_info : NLPInfo
        Information about the NLP solver status and results, including the final values of
        decision variables, constraint multipliers, and other information. The attributes are:

        ipopt_status : IpoptStatus
            The status code returned by Ipopt, the same object as ``status``.
        ipopt_status_message : str
            A human-readable message corresponding to the status code. In most cases, it's
            the same message as that printed in the console by Ipopt.
        obj_val : float
            The value of the NLP objective function at the returned point.
        x : numpy.ndarray
            The optimal values of the NLP decision variables.
        g : numpy.ndarray
            The values of the NLP constraint functions at the returned point.
        mult_x_L : numpy.ndarray
            Lagrange multipliers associated with the lower bounds of the decision variables.
        mult_x_U : numpy.ndarray
            Lagrange multipliers associated with the upper bounds of the decision variables.
        mult_g : numpy.ndarray
            Lagrange multipliers associated with the constraints.
    """

    name: str
    # Whatever the front end handed to solve() as its origin -- a yapss.Problem from the
    # released API -- or, when a front end passes none, the ProblemSpec the solve ran from.
    # The back end does not name either type here, which is why this is not narrower.
    problem: Any
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

    @property
    def status(self) -> IpoptStatus:
        """The status Ipopt reported, as an `IpoptStatus` (an `IntEnum`)."""
        return self.nlp_info.ipopt_status

    @property
    def converged(self) -> bool:
        """Whether Ipopt reported a converged solution: status 0, 1, or 6."""
        return self.nlp_info.ipopt_status.converged

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


_NLP_INPUTS = (
    "x_L",
    "x_U",
    "g_L",
    "g_U",
    "z0",
    "grad_f",
    "jac_g_row",
    "jac_g_col",
    "jac_g",
    "obj_scaling",
    "x_scaling",
    "g_scaling",
    "iterations",
    "inf_pr",
    "inf_du",
    "complementarity",
)
"""What `solver.solve` records beside Ipopt's outputs: its inputs, the first derivatives at the
returned point, and Ipopt's final measures of convergence."""


@dataclass(frozen=True)
class NLPInfo:
    """Container for NLP solver information.

    Beside Ipopt's outputs it holds what Ipopt was given -- the bounds ``x_L``, ``x_U``,
    ``g_L``, ``g_U``, the starting point ``z0``, and the scaling -- and, at the returned point,
    the objective's gradient ``grad_f`` and the constraints' Jacobian ``jac_g`` in the
    structure (``jac_g_row``, ``jac_g_col``) Ipopt was given, with Ipopt's final measures of
    convergence in its scaled terms.
    """

    ipopt_status: IpoptStatus
    ipopt_status_message: str
    g: NDArray[np.float64]
    obj_val: float
    x: NDArray[np.float64]
    mult_x_L: NDArray[np.float64]  # noqa: N815
    mult_x_U: NDArray[np.float64]  # noqa: N815
    mult_g: NDArray[np.float64]
    x_L: NDArray[np.float64]  # noqa: N815
    x_U: NDArray[np.float64]  # noqa: N815
    g_L: NDArray[np.float64]  # noqa: N815
    g_U: NDArray[np.float64]  # noqa: N815
    z0: NDArray[np.float64]
    grad_f: NDArray[np.float64]
    jac_g_row: NDArray[np.intp]
    jac_g_col: NDArray[np.intp]
    jac_g: NDArray[np.float64]
    obj_scaling: float
    x_scaling: NDArray[np.float64]
    g_scaling: NDArray[np.float64]
    iterations: int
    inf_pr: float
    inf_du: float
    complementarity: float

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
