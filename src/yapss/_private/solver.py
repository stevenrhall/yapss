"""

Construct the nonlinear program (NLP) from the user input and solve.

"""

# future imports
from __future__ import annotations

# standard imports
import contextlib
import ctypes
import functools
import importlib.util
import inspect
import os
import signal
import sys
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

# third party imports
import numpy as np
from numpy.typing import NDArray

# package imports
from .auto import make_auto_functions
from .bounds import get_nlp_constraint_function_bounds, get_nlp_decision_variable_bounds
from .central_difference import make_cd_functions
from .config import get_conda_prefix, warn_ipopt_source_deprecated
from .guess import make_initial_guess_nlp
from .mesh import Mesh

# The backend default is determined by environment: a conda environment gets
# cyipopt (safe to coexist with CasADi's own bundled IPOPT there, since
# conda-forge builds share one OpenMP runtime); anything else gets the vendored
# mseipopt, which loads CasADi's own bundled IPOPT directly rather than a second
# binary. Running a second, independently-built IPOPT alongside CasADi's outside
# of conda risks an OpenMP runtime collision.
#
# `ipopt_source` can still override the default, but is deprecated and is removed
# in 0.3.0, after which the environment decides and nothing overrides it. See
# IPOPT_BACKEND_POLICY.md.
#
# Importing the vendored mseipopt is free -- it is ctypes declarations, and no
# library is opened until `initialize_ipopt()` is called.
from .mseipopt import bare, bare_np, initialize_ipopt
from .mseipopt.library import smoke_test as library_smoke_test
from .nlp import NLP
from .solution import Solution, make_solution_object
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure
from .user import make_user_functions

_IN_CONDA = bool(get_conda_prefix())

# Deliberately `find_spec` rather than `import cyipopt`: importing cyipopt opens
# its own IPOPT binary, so probing by import would map a second copy into the
# process for anyone who merely has cyipopt installed -- tripping the duplicate
# guard on a configuration that is perfectly fine. `find_spec` answers "is it
# installed" without executing it.
if _IN_CONDA and importlib.util.find_spec("cyipopt") is None:
    msg = textwrap.dedent(
        """
        YAPSS is running in a Conda environment, where it connects to Ipopt through
            'cyipopt', but 'cyipopt' is not installed. Install it with:

                conda install -c conda-forge cyipopt

            YAPSS uses cyipopt in a Conda environment and its own bundled interface
            everywhere else; this is not configurable. See "Sharp Edges" in the user
            guide for why.
        """,
    ).strip()
    raise ModuleNotFoundError(msg)

if TYPE_CHECKING:
    # package imports
    import yapss

    from .input_args import ProblemFunctions


# Define a custom warning class
class IpoptOptionSettingWarning(Warning):
    """Custom warning for issues in setting Ipopt options."""


_env_deprecation_warned = False
"""Whether the `YAPSS_IPOPT_SOURCE` deprecation has already been reported.

The environment variable never passes through the `ipopt_source` setter, so it
needs its own warning -- but it would otherwise fire on every solve, which is
noise rather than information.
"""


def configure_ipopt_source(problem: yapss.Problem) -> str:
    """Resolve which Ipopt backend to use, honoring the deprecated override.

    Deprecated in favor of letting the environment decide; removed in 0.3.0,
    along with this function. See IPOPT_BACKEND_POLICY.md.

    Parameters
    ----------
    problem : yapss.Problem

    Returns
    -------
    str
        One of ``"cyipopt"``, ``"casadi"``, or a path to an Ipopt library.
    """
    global _env_deprecation_warned  # noqa: PLW0603

    ipopt_source = problem.ipopt_source

    if ipopt_source == "default":
        env_ipopt_source = os.getenv("YAPSS_IPOPT_SOURCE", "")
        if env_ipopt_source:
            if not _env_deprecation_warned:
                warn_ipopt_source_deprecated(env_ipopt_source, stacklevel=3)
                _env_deprecation_warned = True
            ipopt_source = env_ipopt_source
        else:
            ipopt_source = "cyipopt" if _IN_CONDA else "casadi"

    # An explicitly requested cyipopt must actually be installed. Probed without
    # importing, so that merely asking the question does not map a second IPOPT.
    if ipopt_source == "cyipopt" and importlib.util.find_spec("cyipopt") is None:
        msg = textwrap.dedent(
            """
            The 'cyipopt' option requires the 'cyipopt' package, which is not installed
                by default in the yapss distribution. To use this option, install the
                package using: 'pip install cyipopt' or
                'conda install -c conda-forge cyipopt'.

                Alternatively, delete the 'ipopt_source' setting: YAPSS then uses its
                own bundled Ipopt interface, which needs no additional packages. Note
                that 'ipopt_source' is deprecated and is removed in 0.3.0.
            """,
        ).strip()
        raise ModuleNotFoundError(msg)

    # A custom path must exist. Checked before loading, since the failure mode
    # after loading a wrong library is a crash rather than an exception.
    if ipopt_source not in ("cyipopt", "casadi") and not Path(ipopt_source).exists():
        msg = f"The provided path to the Ipopt library does not exist: {ipopt_source}"
        raise FileNotFoundError(msg)

    return ipopt_source


def _load_explicit_ipopt(path: str) -> None:
    """Load an Ipopt library chosen by the user, bypassing every verification.

    Quarantined here rather than placed in `mseipopt.library`, whose invariant is
    that it only ever opens the Ipopt that CasADi bundles. That property is
    currently absolute and directly testable, and an exception living inside it
    would weaken it for every caller. This function is the exception, it is
    deprecated, and it is deleted in 0.3.0 along with the option it serves.

    None of the checks that protect the default path can apply here:

    * The ABI header check cannot run. `read_ipopt_header()` reads
      ``IpoptConfig.h`` from the CasADi package, which describes a *different*
      library; verifying against it would report a match that means nothing.
      `bare` therefore retains its fixed Ipopt 3.14+ `c_bool` declaration, so
      a pre-3.14 Ipopt silently gets the wrong `Bool` width.
    * The duplicate-copy guard is skipped, because a second copy is precisely
      what the caller asked for.
    * The smoke test cannot substitute for either: negative controls showed an
      ABI mismatch crashes the process rather than returning a bad status.

    Parameters
    ----------
    path : str
        Path to the Ipopt shared library.
    """
    # Mirrors `library.load_ipopt()`: since Python 3.8, Windows no longer searches
    # PATH when resolving a DLL's own dependencies (MUMPS, OpenBLAS, libgfortran),
    # so without this the load fails with a bare "DLL load failed" naming only the
    # top-level library.
    cookie = None
    if os.name == "nt" and Path(path).is_absolute():
        directory = Path(path).parent
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            cookie = os.add_dll_directory(str(directory))

    try:
        lib = ctypes.CDLL(path)
    except OSError as exc:
        msg = (
            f"could not load the Ipopt shared library from {path!r}. This path came "
            f"from the deprecated 'ipopt_source' setting; deleting that setting lets "
            f"YAPSS use its own bundled interface instead."
        )
        raise OSError(msg) from exc
    finally:
        if cookie is not None:
            cookie.close()

    bare.use_library(lib)
    # The smoke test cannot detect an ABI mismatch here -- it crashes instead --
    # but it does confirm the library exports the interface we expect, and a crash
    # in a known three-variable problem beats one an hour into a user's solve.
    library_smoke_test()


def solve(problem: yapss.Problem) -> Solution:
    """Create the nonlinear program (NLP) from the user input and solve.

    This function does **not** warn when Ipopt fails to converge. The caller is
    responsible for that, via `solution.warn_if_not_converged()`; `Problem.solve` is
    currently the only caller and does so.

    The check belongs at the public boundary rather than here for two reasons. Only
    there can `stacklevel` point at user code -- and `stacklevel` also determines the
    location Python's default warning filter dedupes on, so warning from in here would
    collapse every unconverged solve in a program to a single registry entry and
    silence all but the first. Second, an internal caller (adaptive mesh refinement,
    say, which solves repeatedly) may need to solve without warning each time.

    Any new public entry point that returns a `Solution` should call
    `warn_if_not_converged` itself.

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
        functions = make_user_functions(problem, z0, mesh.tau_u)
    elif method in ("central-difference", "central-difference-full"):
        functions = make_cd_functions(problem, z0, mesh.tau_u)
    elif method == "auto":
        functions = make_auto_functions(problem)
    else:
        raise RuntimeError
    nlp_temp = NLP(problem, functions, mesh)
    nlp_temp.intermediate = problem._intermediate_cb

    # NLP variable and constraint bounds
    ub, lb = get_nlp_decision_variable_bounds(problem)
    gu, gl = get_nlp_constraint_function_bounds(problem)

    ipopt_source = configure_ipopt_source(problem)

    if ipopt_source == "cyipopt":
        # Imported here, not at module scope: importing cyipopt opens its own IPOPT
        # binary, so an eager import would map a second copy for anyone who merely
        # has cyipopt installed. Availability was already checked with `find_spec`.
        import cyipopt

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
        if ipopt_source == "casadi":
            # Resolve, load, verify and configure once per process. Idempotent, so
            # unlike the path comparison this replaces, there is nothing to re-check.
            initialize_ipopt()
        else:
            _load_explicit_ipopt(ipopt_source)

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
            # A custom library path is a deprecated, explicitly unsafe escape
            # hatch. Its visible FutureWarning explains that YAPSS cannot verify
            # the ABI and that incompatibility may crash the process. Keep the
            # bypass private so direct bare_np callers retain the hard guarantee.
            _unsafe_allow_unverified_library=ipopt_source != "casadi",
        )
        ipopt_problem.set_intermediate_callback(nlp_temp.intermediate)

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

    # CasADi's bundled Ipopt is built with SPRAL and selects it by default on
    # Windows and Linux. macOS has no SPRAL in that build, and Conda's Ipopt
    # defaults to MUMPS on every platform -- so the vendored path is the only
    # configuration YAPSS supports that does not use MUMPS, and it is that way
    # because CasADi's build enables a solver it also builds without OpenMP, not
    # because SPRAL suits these problems. Setting MUMPS makes every YAPSS
    # installation behave alike.
    #
    # Only when the user has not chosen a solver, and only for CasADi's own library
    # -- an explicit `ipopt_source` path is the user's own build. This must precede
    # the `mumps_pivot_order` block below, which is meaningful only once MUMPS is
    # the solver in use.
    if ipopt_source == "casadi" and "linear_solver" not in problem.ipopt_options.get_options():
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
    # Scoped narrowly on purpose: macOS only, CasADi's own bundled library only
    # (an explicit `ipopt_source` path is the user's own build, not the one with
    # the defect), and only when the user has not chosen an ordering. Conda is
    # excluded because its Ipopt belongs to the user and may be built against HSL.
    if (
        sys.platform == "darwin"
        and ipopt_source == "casadi"
        and "mumps_pivot_order" not in problem.ipopt_options.get_options()
    ):
        with contextlib.suppress(ValueError, TypeError):
            ipopt_problem.add_option("mumps_pivot_order", 6)

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

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", warning_message)

            if problem.catch_keyboard_interrupt:
                original_handler = signal.signal(signal.SIGINT, problem._signal_handler)
                try:
                    z, nlp_info = _solve_ipopt_problem(ipopt_problem, z0, ipopt_source)
                finally:
                    signal.signal(signal.SIGINT, original_handler)
                    problem._abort = False
            else:
                z, nlp_info = _solve_ipopt_problem(ipopt_problem, z0, ipopt_source)
    finally:
        # Callback exceptions are re-raised only after Ipopt returns. Cleanup
        # must still release the native problem on that propagation path.
        ipopt_problem.close()

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
    sense_sign = -1.0 if problem.sense == "maximize" else 1.0
    obj_scale = sense_sign / problem.scale.objective

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


def _solve_ipopt_problem(
    ipopt_problem: Any,
    z0: NDArray[np.float64],
    ipopt_source: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Solve through either backend and normalize its result for solution construction."""
    if ipopt_source == "cyipopt":
        result = ipopt_problem.solve(z0)
        return cast(tuple[NDArray[np.float64], dict[str, Any]], result)

    mseipopt_problem = ipopt_problem
    mseipopt_problem.add_option("warm_start_init_point", "no")
    x = np.array(z0, dtype=np.float64, copy=True, order="C")
    result = mseipopt_problem.solve(x)
    info: dict[str, Any] = {
        "g": result.g,
        "obj_val": result.obj_val,
        "mult_g": result.mult_g,
        "mult_x_L": result.mult_x_L,
        "mult_x_U": result.mult_x_U,
        "status": result.status,
    }
    return result.x, info


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


def _jacobian_callback(function: Callable[..., Any]) -> Callable[..., bool]:
    @functools.wraps(function)
    def callback(
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        if output.size:
            if _accepts_output(function):
                function(x, out=output)
            else:
                output[...] = function(x)
        return True

    return callback


def _hessian_callback(function: Callable[..., Any]) -> Callable[..., bool]:
    @functools.wraps(function)
    def callback(  # noqa: PLR0913
        x: NDArray[np.float64],
        new_x: bool,  # noqa: ARG001, FBT001
        obj_factor: float,
        multipliers: NDArray[np.float64],
        new_multipliers: bool,  # noqa: ARG001, FBT001
        output: NDArray[np.float64],
    ) -> bool:
        if output.size:
            if _accepts_output(function):
                function(x, multipliers, obj_factor, out=output)
            else:
                output[...] = function(x, multipliers, obj_factor)
        return True

    return callback


@functools.lru_cache
def _accepts_output(function: Callable[..., Any]) -> bool:
    parameters = inspect.signature(function).parameters
    output = parameters.get("out")
    if output is None or list(parameters).index("out") == 0:
        return False
    return output.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


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
