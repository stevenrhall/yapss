"""

Construct the nonlinear program (NLP) from the user input and solve.

"""

# future imports
from __future__ import annotations

# standard imports
import contextlib
import functools
import signal
import sys
import threading
import warnings
from typing import TYPE_CHECKING, Any, assert_never

# third party imports
import numpy as np

# package imports
from .auto import make_auto_functions
from .bounds import get_nlp_constraint_function_bounds, get_nlp_decision_variable_bounds
from .central_difference import make_cd_functions
from .config import get_conda_prefix, warn_if_ipopt_source_env_set
from .guess import make_initial_guess_nlp
from .ipopt_options import IpoptOptionSettingWarning, explain_refusal
from .ipopt_status import status_or_raise
from .mesh import Mesh
from .mseipopt import bare_np, initialize_ipopt
from .nlp import NLP
from .setup_check import check_callbacks, check_derivatives
from .solution import Solution, make_solution_object
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .user import make_user_functions

# In a Conda environment CasADi, and so YAPSS, uses conda-forge's Ipopt package, which
# the user installed and which may be built against solvers YAPSS cannot detect (HSL,
# for instance). The solver defaults below exist for CasADi's own bundled build and are
# not applied there.
_IN_CONDA = bool(get_conda_prefix())

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    from numpy.typing import NDArray

    from .input_args import ProblemFunctions
    from .spec import ProblemSpec


def solve(problem: ProblemSpec, origin: Any = None) -> Solution:
    """Create the nonlinear program (NLP) from the user input and solve.

    This function does **not** warn when Ipopt fails to converge. The caller is
    responsible for that, via `solution.warn_if_not_converged()`; `Problem.solve` is
    currently the only caller and does so.

    The check belongs at the public boundary rather than here for two reasons. Only
    there can `stacklevel` point at user code. Second, an internal caller (adaptive mesh
    refinement, say, which solves repeatedly) may need to solve without warning each
    time.

    Every unconverged solve warns, even when repeated from the same line: the
    `warnings.catch_warnings()` block around the Ipopt call below invalidates Python's
    per-location warning registry on exit, so the "default" and "once" filter actions
    do not deduplicate across solves ("ignore" and "error" are unaffected). The same
    invalidation applies to every other warning in the process.

    Any new public entry point that returns a `Solution` should call
    `warn_if_not_converged` itself.

    Parameters
    ----------
    problem : ProblemSpec

    Returns
    -------
    Solution
    """
    # TODO: Move line below to nlpy.py
    mesh = Mesh(problem.phases)
    mesh.set_matrices(problem.spectral_method)

    # need initial guess to get the derivative structure when using methods "user" and
    # "central-difference"
    z0 = make_initial_guess_nlp(problem, mesh)

    method = problem.derivative_method
    functions: ProblemFunctions
    match method:
        case "user":
            functions = make_user_functions(problem, z0, mesh.tau_u)
        case "central-difference" | "central-difference-full":
            functions = make_cd_functions(problem, z0, mesh.tau_u)
        case "auto":
            functions = make_auto_functions(problem)
        case _:
            assert_never(method)
    nlp_temp = NLP(problem, functions, mesh)
    # Ctrl-C asks Ipopt to stop at the next iteration rather than interrupting Python inside
    # the solver. The flag is state of *this solve*, so it lives here rather than on whatever
    # object the problem came from.
    aborted = [False]

    def signal_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        aborted[0] = True

    def intermediate(*args: Any) -> bool:
        if aborted[0]:
            aborted[0] = False
            return False
        user_callback = problem.intermediate_callback
        return True if user_callback is None else bool(user_callback(*args))

    nlp_temp.intermediate = intermediate

    # Check the callbacks (unassigned rows, non-finite values, pointwise), then the NLP's
    # first derivatives, before Ipopt can pass a non-finite Jacobian to its linear solver.
    # After the derivative setup, so that its own errors come first (see the module).
    check_callbacks(problem, mesh, z0)
    check_derivatives(problem, nlp_temp, z0)

    # NLP variable and constraint bounds
    ub, lb = get_nlp_decision_variable_bounds(problem)
    gu, gl = get_nlp_constraint_function_bounds(problem)

    warn_if_ipopt_source_env_set()

    # Resolve, load, verify and configure once per process; idempotent.
    initialize_ipopt()

    jacobian_structure = nlp_temp.jacobianstructure()
    hessian_structure = nlp_temp.hessianstructure()
    ipopt_problem = MseipoptProblem(
        lb,
        ub,
        gl,
        gu,
        eval_f=_objective_callback(nlp_temp.objective),
        eval_g=_constraint_callback(nlp_temp.constraints),
        eval_grad_f=_gradient_callback(nlp_temp.gradient),
        jacobian_structure=jacobian_structure,
        eval_jac_g=_jacobian_callback(nlp_temp.jacobian),
        hessian_structure=hessian_structure,
        eval_h=_hessian_callback(nlp_temp.hessian),
    )
    ipopt_problem.set_intermediate_callback(nlp_temp.intermediate)

    # apply user ipopt options
    for name, value in problem.ipopt_options.items():
        try:
            ipopt_problem.add_option(name, value)
        except (ValueError, TypeError) as e:
            # Ipopt says only that it refused the option, so compare the value with what
            # Ipopt's own documentation records for it: a value outside the documented
            # range is the user's mistake, a value inside it means the build most likely
            # lacks the option. Neither verdict is stated as certain (E5).
            msg, is_error = explain_refusal(name, value, str(e))
            if is_error:
                raise ValueError(msg) from e
            # stacklevel 3: warn -> solver.solve -> Problem.solve -> the user's call,
            # as warn_if_not_converged does
            warnings.warn(msg, category=IpoptOptionSettingWarning, stacklevel=3)

    if "timing_statistics" not in problem.ipopt_options:
        with contextlib.suppress(ValueError, TypeError):
            ipopt_problem.add_option("timing_statistics", "yes")

    # CasADi's bundled Ipopt is built with SPRAL and selects it by default on
    # Windows and Linux. macOS has no SPRAL in that build, and Conda's Ipopt
    # defaults to MUMPS on every platform -- so the vendored path is the only
    # configuration YAPSS supports that does not use MUMPS, and it is that way
    # because CasADi's build enables a solver it also builds without OpenMP, not
    # because SPRAL suits these problems. Setting MUMPS makes every YAPSS
    # installation behave alike.
    #
    # Only when the user has not chosen a solver, and only for CasADi's own library
    # -- not in Conda, where Ipopt is the user's own package. This must precede the
    # `mumps_pivot_order` block below, which is meaningful only once MUMPS is the
    # solver in use.
    if not _IN_CONDA and "linear_solver" not in problem.ipopt_options:
        with contextlib.suppress(ValueError, TypeError):
            ipopt_problem.add_option("linear_solver", "mumps")

    # macOS crash workaround, not performance tuning. CasADi's bundled Ipopt
    # segfaults inside libcoinmetis (METIS 4.0, called by MUMPS for the
    # fill-reducing ordering) on macOS only -- at every problem size when METIS is
    # requested, and above roughly 5000 variables by default, where MUMPS selects
    # METIS on its own. Fixed upstream in CasADi 3.8.0, but the workaround stays
    # until the *floor* of the CasADi requirement can be raised past it; a user who
    # resolves to 3.7.2 crashes no matter what the upper bound allows. QAMD is
    # chosen because it is not a METIS alias -- PORD is one, and crashes at the
    # identical fault address.
    #
    # Scoped narrowly on purpose: macOS only, CasADi's own bundled library only, and
    # only when the user has not chosen an ordering. Conda is excluded because its
    # Ipopt belongs to the user and may be built against HSL.
    if (
        sys.platform == "darwin"
        and not _IN_CONDA
        and "mumps_pivot_order" not in problem.ipopt_options
    ):
        with contextlib.suppress(ValueError, TypeError):
            ipopt_problem.add_option("mumps_pivot_order", 6)

    if problem.derivative_order == "first":
        ipopt_problem.add_option("hessian_approximation", "limited-memory")

    # set NLP scaling
    obj_scale, z_scaling, c_scaling = get_nlp_scaling(problem)

    ipopt_problem.set_problem_scaling(
        obj_scaling=obj_scale,
        x_scaling=z_scaling,
        g_scaling=c_scaling,
    )

    # TODO: add near here the ability to scale as above or to use yapss scaling.
    ipopt_problem.add_option("nlp_scaling_method", "user-scaling")

    # solve NLP. If keyboard interrupt is raised, signal IPOPT to stop through the
    # intermediate callback
    #
    # No `catch_warnings` block around the solve: it used to silence one NumPy message
    # ("A builtin ctypes object gave a PEP3118 format string that does not match"), and
    # leaving such a block invalidates Python's per-location warning registry, so the
    # "default" and "once" filter actions stopped deduplicating for every warning in the
    # process, not only YAPSS's. The message does not appear on any supported NumPy
    # (checked on 2.4.6 and 2.5 across a full solve); if it returns, filter it where
    # `numpy.ctypeslib.as_array` is called, in mseipopt.
    try:
        # signal.signal is allowed only on the main thread. A worker thread never
        # receives the keyboard interrupt anyway, so there is nothing to catch there
        # and the solve simply runs without the handler.
        catch_interrupt = (
            problem.catch_keyboard_interrupt
            and threading.current_thread() is threading.main_thread()
        )
        if catch_interrupt:
            original_handler = signal.signal(signal.SIGINT, signal_handler)
            try:
                z, nlp_info = _solve_ipopt_problem(ipopt_problem, z0)
            finally:
                signal.signal(signal.SIGINT, original_handler)
                aborted[0] = False
        else:
            z, nlp_info = _solve_ipopt_problem(ipopt_problem, z0)
    finally:
        # Callback exceptions are re-raised only after Ipopt returns. Cleanup
        # must still release the native problem on that propagation path.
        ipopt_problem.close()

    # A status without an iterate raises here, in the internal solve, so that a loop of
    # solves (mesh refinement) raises too; the convergence warning is for the public boundary.
    nlp_info["status"] = status_or_raise(nlp_info["status"])
    nlp_info["x"] = z
    return make_solution_object(problem, mesh, nlp_temp, nlp_info, origin)


def get_nlp_scaling(
    problem: ProblemSpec,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Convert optimal control problem scaling to NLP scaling.

    Parameters
    ----------
    problem : ProblemSpec
        User-defined optimal control problem, including scaling factors for the
        problem variables and constraints.

    Returns
    -------
    float
        The objective scale factor
    NDArray
        The NLP decision variable scale factor array
    NDArray
        The NLP constraint function scale factor array
    """
    # objective
    sense_sign = -1.0 if problem.sense == "maximize" else 1.0
    obj_scale = sense_sign / problem.objective_scale

    # decision variables
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)
    dv.z[:] = 1.0

    for p in range(problem.np):
        phase = problem.phases[p]
        for i in range(problem.nx[p]):
            # every stored value of a state, zero modes (LGL) included, has its scale
            dv.phase[p].xa[i][:] = 1.0 / phase.state_scale[i]
        for i in range(problem.nu[p]):
            dv.phase[p].u[i][:] = 1.0 / phase.control_scale[i]
        for i in range(problem.nq[p]):
            dv.phase[p].q[i] = 1.0 / phase.integral_scale[i]
        dv.phase[p].t0[:] = 1.0 / phase.time_scale
        dv.phase[p].tf[:] = 1.0 / phase.time_scale

    dv.s[:] = 1.0 / problem.parameter_scale

    # constraints
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    cf.c[:] = 1.0

    for p in range(problem.np):
        phase = problem.phases[p]
        for i in range(problem.nx[p]):
            cf.phase[p].defect[i][:] = 1.0 / phase.dynamics_scale[i]
            cf.phase[p].lg_defect[i][:] = 1.0 / phase.state_scale[i]  # empty unless LG
        for i in range(problem.nq[p]):
            cf.phase[p].integral[i] = 1.0 / phase.integral_scale[i]
        for i in range(problem.nh[p]):
            cf.phase[p].path[i][:] = 1.0 / phase.path_scale[i]
        cf.phase[p].duration[0] = 1.0 / phase.time_scale

    cf.discrete[:] = 1.0 / problem.discrete_scale

    z_scaling = dv.z
    c_scaling = cf.c

    return obj_scale, z_scaling, c_scaling


def _solve_ipopt_problem(
    ipopt_problem: MseipoptProblem,
    z0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Solve and normalize the result for solution construction."""
    ipopt_problem.add_option("warm_start_init_point", "no")
    x = np.array(z0, dtype=np.float64, copy=True, order="C")
    result = ipopt_problem.solve(x)
    info: dict[str, Any] = {
        "g": result.g,
        "obj_val": result.obj_val,
        "mult_g": result.mult_g,
        "mult_x_L": result.mult_x_L,
        "mult_x_U": result.mult_x_U,
        "status": result.status,
    }
    return result.x, info


# The ``new_x`` argument Ipopt passes to each callback is deliberately unused. The NLP
# evaluates the continuous functions once per point through a value-keyed cache
# (``nlp.ContinuousEvaluator``), which cannot serve a stale value the way a misread
# flag could.
def _objective_callback(function: Callable[..., Any]) -> Callable[..., bool]:
    @functools.wraps(function)
    def callback(
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        output[()] = function(x)
        return True

    return callback


def _constraint_callback(function: Callable[..., Any]) -> Callable[..., bool]:
    @functools.wraps(function)
    def callback(
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        if output.size:
            output[()] = function(x)
        return True

    return callback


def _gradient_callback(function: Callable[..., Any]) -> Callable[..., bool]:
    @functools.wraps(function)
    def callback(
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        output[()] = function(x)
        return True

    return callback


def _jacobian_callback(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> bare_np.EvaluationCallback:
    @functools.wraps(function)
    def callback(
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        if output.size:
            output[...] = function(x)
        return True

    return callback


def _hessian_callback(
    function: Callable[
        [NDArray[np.float64], NDArray[np.float64], np.float64],
        NDArray[np.float64],
    ],
) -> bare_np.HessianCallback:
    @functools.wraps(function)
    def callback(  # noqa: PLR0913, PLR0917 -- signature dictated by mseipopt's C callback API
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        obj_factor: float,
        multipliers: NDArray[np.float64],
        new_multipliers: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        if output.size:
            output[...] = function(x, multipliers, np.float64(obj_factor))
        return True

    return callback


class MseipoptProblem(bare_np.Problem):
    """Add YAPSS's option/scaling vocabulary to the standalone NumPy layer."""

    def add_option(self, keyword: str, value: float | str) -> None:
        if isinstance(value, int):
            self.add_int_option(keyword, value)
        elif isinstance(value, float):
            self.add_num_option(keyword, value)
        elif isinstance(value, str):
            self.add_str_option(keyword, value)
        else:
            msg = (  # type: ignore[unreachable]
                f"'value' must be of type int, float or str, got '{type(value)}'"
            )
            raise TypeError(msg)

    def set_problem_scaling(
        self,
        obj_scaling: float,
        x_scaling: NDArray[np.float64] | None,
        g_scaling: NDArray[np.float64] | None,
    ) -> None:
        self.set_scaling(obj_scaling, x_scaling, g_scaling)
