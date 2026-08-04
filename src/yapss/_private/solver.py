"""

Construct the nonlinear program (NLP) from the user input and solve.

"""

# future imports
from __future__ import annotations

# standard imports
import contextlib
import ctypes
import importlib.util
import os
import signal
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# third party imports
import numpy as np

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
# in 0.2.0, after which the environment decides and nothing overrides it. See
# IPOPT_BACKEND_POLICY.md.
#
# Importing the vendored mseipopt is free -- it is ctypes declarations, and no
# library is opened until `initialize_ipopt()` is called.
from .mseipopt import bare, ez, initialize_ipopt
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
    # third party imports
    from numpy.typing import NDArray

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

    Deprecated in favor of letting the environment decide; removed in 0.2.0,
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
                that 'ipopt_source' is deprecated and is removed in 0.2.0.
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
    deprecated, and it is deleted in 0.2.0 along with the option it serves.

    None of the checks that protect the default path can apply here:

    * The ABI header check cannot run. `read_ipopt_header()` reads
      ``IpoptConfig.h`` from the CasADi package, which describes a *different*
      library; verifying against it would report a match that means nothing.
      `bare.set_bool_type()` therefore keeps its import-time default of
      `c_bool`, so a pre-3.14 Ipopt silently gets the wrong `Bool` width.
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


class EZProblem(ez.Problem):
    """Adapt the vendored `ez.Problem` to the API `solve` uses for both backends.

    Defined unconditionally: with `ipopt_source` restored, the vendored path is
    reachable inside a Conda environment too, via ``ipopt_source="casadi"`` or an
    explicit library path.
    """

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
