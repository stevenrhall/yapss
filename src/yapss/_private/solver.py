"""

Construct the nonlinear program (NLP) from the user input and solve.

"""

# future imports
from __future__ import annotations

# standard imports
import contextlib
import signal
import textwrap
import warnings
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING

# third party imports
import numpy as np

# package imports
from .auto import make_auto_functions
from .bounds import get_nlp_constraint_function_bounds, get_nlp_decision_variable_bounds
from .central_difference import make_cd_functions
from .config import get_casadi_ipopt_library_path
from .guess import make_initial_guess_nlp
from .mesh import Mesh
from .nlp import NLP
from .solution import Solution, make_solution_object
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .user import make_user_functions

try:
    import cyipopt

    CYIPOPT = True
except ModuleNotFoundError:
    CYIPOPT = False

try:
    from mseipopt import bare, ez
    from mseipopt.bare import load_library

    MSEIPOPT = True
except ModuleNotFoundError:
    MSEIPOPT = False

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .input_args import ProblemFunctions

if not CYIPOPT and not MSEIPOPT:
    msg = "No module named 'cyipopt' or 'mseipopt'. At least one must be installed."
    raise ModuleNotFoundError(msg)


# Define a custom warning class
class IpoptOptionSettingWarning(Warning):
    """Custom warning for issues in setting Ipopt options."""


def solve(problem: yapss.Problem) -> Solution:
    """Create the nonlinear program (NLP) from the user input and solve.

    Parameters
    ----------
    problem : yapss.Problem

    Returns
    -------
    Solution
    """
    problem.validate()
    # TODO: Move line below to nlpy.py
    mesh = Mesh(problem.mesh.phase)
    mesh.set_matrices(problem.spectral_method)

    # need initial guess to get the derivative structure when using methods "user" and
    # "central-difference"
    z0 = make_initial_guess_nlp(problem, mesh)

    method = problem.derivatives.method
    functions: ProblemFunctions
    if method == "user":
        functions = make_user_functions(problem, z0)
    elif method in ("central-difference", "central-difference-full"):
        functions = make_cd_functions(problem, z0)
    elif method == "auto":
        functions = make_auto_functions(problem)
    else:
        raise RuntimeError
    nlp_temp = NLP(problem, functions, mesh)
    nlp_temp.intermediate = problem._intermediate_cb

    # NLP variable and constraint bounds
    ub, lb = get_nlp_decision_variable_bounds(problem)
    gu, gl = get_nlp_constraint_function_bounds(problem)

    ipopt_source: str = configure_ipopt_source(problem)

    if ipopt_source == "cyipopt":
        if not CYIPOPT:
            msg = (
                "The package 'cyipopt'  is not installed. To resolve:\n"
                "   1. Set 'ipopt_source' attribute to 'casadi' or an Ipopt library path as a "
                "string, or\n"
                "   2. Set 'ipopt_source' attribute to 'default', with the environment variable "
                "'YAPSS_IPOPT_SOURCE'\n"
                "      not set, or set to 'default', 'casadi' or an Ipopt library path, or\n"
                "   3. Install the 'cyipopt' package."
            )
            raise ModuleNotFoundError(msg)
        ipopt_problem = cyipopt.Problem(
            n=len(lb),
            m=len(gl),
            lb=lb,
            ub=ub,
            cl=gl,
            cu=gu,
            problem_obj=nlp_temp,
        )
    else:
        if not MSEIPOPT:
            msg = (
                "The package 'mseipopt'  is not installed. To resolve:\n"
                "   1. Set 'ipopt_source' attribute to 'cyipopt', or\n"
                "   2. Set 'ipopt_source' attribute to 'default', with the environment variable "
                "'YAPSS_IPOPT_SOURCE'\n"
                "      not set, or set to 'cyipopt' or 'default', or\n"
                "   3. Install the 'mseipopt' package."
            )
            raise ModuleNotFoundError(msg)
        ipopt_path = get_casadi_ipopt_library_path() if ipopt_source == "casadi" else ipopt_source

        # don't reload library if already loaded
        if bare._ipopt_lib is None or ipopt_path != bare._ipopt_lib._name:
            load_library(ipopt_path)

        jacobian_structure = nlp_temp.jacobianstructure()
        hessian_structure = nlp_temp.hessianstructure()
        hess = (
            hessian_structure,
            lambda x, obj_factor, _lambda: nlp_temp.hessian(x, _lambda, obj_factor),
        )

        ipopt_problem = EZProblem(
            x_bounds=(lb, ub),
            g_bounds=(gl, gu),
            f=nlp_temp.objective,
            g=nlp_temp.constraints,
            grad=nlp_temp.gradient,
            jac=(jacobian_structure, nlp_temp.jacobian),
            nele_jac=len(jacobian_structure[0]),
            hess=hess,
            nele_hess=len(hessian_structure[0]),
        )

    # apply user ipopt options
    for name, value in problem.ipopt_options.get_options().items():
        try:
            ipopt_problem.add_option(name, value)
        # try/except in a loop is unavoidable here, and not a performance issue
        except (ValueError, TypeError) as e:  # noqa: PERF203 (try-except-in-loop)
            msg = (
                f"Failed to set option '{name}' with value '{value}': {e}. "
                f"See Ipopt console output for more details."
            )
            warnings.warn(msg, category=IpoptOptionSettingWarning, stacklevel=2)

    if "timing_statistics" not in problem.ipopt_options.get_options():
        with contextlib.suppress(ValueError, TypeError):
            ipopt_problem.add_option("timing_statistics", "yes")

    if problem.derivatives.order == "first":
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

    # suppress expected warning message from numpy
    warning_message = "A builtin ctypes object gave a PEP3118 format string that does not match"

    # solve NLP. If keyboard interrupt is raised, signal IPOPT to stop through the
    # intermediate callback

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", warning_message)

        if problem.catch_keyboard_interrupt:
            original_handler = signal.signal(signal.SIGINT, problem._signal_handler)
            try:
                z, nlp_info = ipopt_problem.solve(z0)
            finally:
                signal.signal(signal.SIGINT, original_handler)
                problem._abort = False
        else:
            z, nlp_info = ipopt_problem.solve(z0)

    # close Ipopt problem to prevent memory leak
    ipopt_problem.close()
    del ipopt_problem

    nlp_info["x"] = z
    return make_solution_object(problem, mesh, nlp_temp, nlp_info)


def get_nlp_scaling(
    problem: yapss.Problem,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Convert optimal control problem scaling to NLP scaling.

    Parameters
    ----------
    problem : yapss.Problem
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
    obj_scale = 1.0 / problem.scale.objective

    # decision variables
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)
    dv.z[:] = 1.0

    for p in range(problem.np):
        phase = problem.scale.phase[p]
        for i in range(problem.nx[p]):
            if problem.spectral_method == "lg":
                dv.phase[p].xa[i][:] = 1.0 / phase.state[i]
            else:
                dv.phase[p].x[i][:] = 1.0 / phase.state[i]
            if problem.spectral_method == "lgl":
                dv.phase[p].xs[i][:] = 1.0 / phase.state[i]
        for i in range(problem.nu[p]):
            dv.phase[p].u[i][:] = 1.0 / phase.control[i]
        for i in range(problem.nq[p]):
            dv.phase[p].q[i] = 1.0 / phase.integral[i]
        dv.phase[p].t0[:] = 1.0 / phase.time
        dv.phase[p].tf[:] = 1.0 / phase.time

    dv.s[:] = 1.0 / problem.scale.parameter

    # constraints
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    cf.c[:] = 1.0

    for p in range(problem.np):
        phase = problem.scale.phase[p]
        for i in range(problem.nx[p]):
            cf.phase[p].defect[i][:] = 1.0 / phase.dynamics[i]
            if problem.spectral_method == "lg":
                lg_defect = cf.phase[p].lg_defect
                lg_defect[i][:] = 1.0 / phase.state[i]
        for i in range(problem.nq[p]):
            cf.phase[p].integral[i] = 1.0 / phase.integral[i]
        for i in range(problem.nh[p]):
            cf.phase[p].path[i][:] = 1.0 / phase.path[i]
        cf.phase[p].duration[0] = 1.0 / phase.time

    cf.discrete[:] = 1.0 / problem.scale.discrete

    z_scaling = dv.z
    c_scaling = cf.c

    return obj_scale, z_scaling, c_scaling


def configure_ipopt_source(problem: yapss.Problem) -> str:
    """Determine the source of the IPOPT library.

    Parameters
    ----------
    problem : yapss.Problem
    """
    ipopt_source = problem.ipopt_source
    if ipopt_source == "default":
        env_ipopt_source: str = getenv("YAPSS_IPOPT_SOURCE", "")
        if env_ipopt_source:
            ipopt_source = env_ipopt_source
        elif CYIPOPT:
            ipopt_source = "cyipopt"
        else:
            ipopt_source = "casadi"
    else:
        ipopt_source = problem.ipopt_source

    # if "cyipopt" is selected, check if the package is installed
    if ipopt_source == "cyipopt" and not CYIPOPT:
        msg = textwrap.dedent(
            """
            The 'cyipopt' option requires the 'cyipopt' package, which is not installed
                by default in the yapss distribution. To use this option, install the package
                using: 'pip install cyipopt' or 'conda install -c conda-forge cyipopt'.
            """,
        ).strip()
        raise ModuleNotFoundError(msg)

    # if a path is provided, check if it exists
    if ipopt_source not in ("cyipopt", "casadi") and not Path(ipopt_source).exists():
        msg = f"The provided path to the IPOPT library does not exist: {ipopt_source}"
        raise FileNotFoundError(msg)

    return ipopt_source


if MSEIPOPT:

    class EZProblem(ez.Problem):  # type: ignore[misc]

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

        def close(self) -> None:
            """Close the Ipopt problem."""
            self.free()
