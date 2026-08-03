"""

Construct the nonlinear program (NLP) from the user input and solve.

"""

# future imports
from __future__ import annotations

# standard imports
import contextlib
import signal
import warnings
from typing import TYPE_CHECKING

# third party imports
import numpy as np

# package imports
from .auto import make_auto_functions
from .bounds import get_nlp_constraint_function_bounds, get_nlp_decision_variable_bounds
from .central_difference import make_cd_functions
from .config import get_conda_prefix
from .guess import make_initial_guess_nlp
from .mesh import Mesh
from .nlp import NLP
from .solution import Solution, make_solution_object
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .user import make_user_functions

# Backend is fully determined by environment: a conda environment gets cyipopt
# (safe to coexist with CasADi's own bundled IPOPT there, since conda-forge
# builds share one OpenMP runtime); anything else gets the vendored mseipopt,
# which loads CasADi's own bundled IPOPT directly rather than a second binary.
# Running a second, independently-built IPOPT alongside CasADi's outside of
# conda risks an OpenMP runtime collision. No user override.
if get_conda_prefix():
    import cyipopt

    CYIPOPT = True
else:
    from .ipopt_library import load_ipopt
    from .mseipopt import bare, ez
    from .mseipopt.bare import use_library

    CYIPOPT = False

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .input_args import ProblemFunctions


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

    if CYIPOPT:
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
        # Resolve, load, and verify once per process. The resolver caches its
        # answer and the answer cannot change within a process, so unlike the
        # path-comparison this replaces, there is nothing to re-check.
        if bare._ipopt_lib is None:
            use_library(load_ipopt()[0])

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


if not CYIPOPT:

    class EZProblem(ez.Problem):

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
