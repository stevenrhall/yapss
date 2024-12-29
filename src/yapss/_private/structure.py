"""

Structures for converting between problem variables and NLP variables.

The _structure module provides two functions:

* :meth:`get_nlp_dv_structure` returns a structure that converts problem decision variables
   to NLP decision variables and vice versa.

* :meth:`get_nlp_cf_structure` returns a structure that converts problem constraint function
   values to NLP constraint function values, and vice versa.

The functions can also be used to generate structures that simplify converting indices
of the problem variables and functions to and from the indices of the NLP variables and
functions.

"""

# future imports
from __future__ import annotations

# standard imports
from types import SimpleNamespace
from typing import TYPE_CHECKING, Generic, TypeVar

# third party imports
import numpy as np

# Define a generic type variable
T = TypeVar("T", bound=np.generic)

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .types_ import CVIndex, CVName, DVKey, PhaseIndex

    # Generic array type
    Array = NDArray[T]
    """Generic array type."""

SPECTRAL_METHODS = ("lg", "lgr", "lgl")
"""Valid spectral methods."""


class DVPhase(Generic[T], SimpleNamespace):
    """Container for decision variables of a single phase.

    Attributes
    ----------
    t0 : Array
    tf : Array
    x : list[Array]
    xs : list[Array]
    xc : list[Array]
    x0 : Array
    xf : Array
    u : list[Array]
    q : Array

    x : list[Array]
        collocation and interpolation points
    xa : list[Array]
        all points (including zero mode)
    xs : list[Array]
        zero mode variables
    xc : list[Array]
        collocation points
    xi : list[Array]
        interpolation points
    """

    t0: Array[T]
    tf: Array[T]
    x: list[Array[T]]
    xs: list[Array[T]]
    xc: list[Array[T]]
    x0: Array[T]
    xf: Array[T]
    u: list[Array[T]]
    q: Array[T]


class DVStructure(Generic[T], SimpleNamespace):
    """Container for decision variables at the problem level.

    Attributes
    ----------
    phase: list[DVPhase]
    s: Array
    z: Array
    var_dict: dict[tuple[PhaseIndex, CVName, CVIndex] | DVKey, Array]
    """

    phase: list[DVPhase[T]]
    s: Array[T]
    z: Array[T]
    var_dict: dict[tuple[PhaseIndex, CVName, CVIndex] | DVKey, Array[T]]


class CFPhase(Generic[T], SimpleNamespace):
    """Container for constraint function values of a single phase.

    Attributes
    ----------
    defect: list[Array]
    defect_index: list[int]
    path: list[Array]
    integral: Array
    duration: Array
    zero_mode: list[Array]
    """

    defect: list[Array[T]]
    lg_defect: list[Array[T]]
    defect_index: list[int]
    path: list[Array[T]]
    integral: Array[T]
    duration: Array[T]
    zero_mode: list[Array[T]]


class CFStructure(Generic[T], SimpleNamespace):
    """Container for constraint function values at the problem level.

    Attributes
    ----------
    phase: list[CFPhase]
    c: Array
    discrete: Array
    """

    phase: list[CFPhase[T]]
    c: Array[T]
    discrete: Array[T]


def validate_spectral_method(spectral_method: str) -> None:
    if spectral_method not in SPECTRAL_METHODS:
        msg = f"'spectral_method' must be one of {SPECTRAL_METHODS}"
        raise RuntimeError(msg)


def calculate_nz(problem: yapss.Problem, spectral_method: str) -> int:
    nz = 0

    # Iterate through the phases
    for p in range(problem.np):
        nx = problem.nx[p]
        nu = problem.nu[p]
        nq = problem.nq[p]

        nc_points = problem.mesh.phase[p].collocation_points
        n_segments = len(nc_points)

        if spectral_method == "lgl":
            nupoints = sum(nc_points) - n_segments + 1
            nxpoints = nupoints
            nz += (nxpoints + n_segments) * nx + nupoints * nu
        elif spectral_method == "lgr":
            nupoints = sum(nc_points)
            nxpoints = nupoints + 1
            nz += nxpoints * nx + nupoints * nu
        elif spectral_method == "lg":
            nupoints = sum(nc_points)
            nxpoints = nupoints + n_segments + 1
            nz += nxpoints * nx + nupoints * nu
        else:
            raise RuntimeError

        # integrals, t0, and tf
        nz += nq + 2

    # add in number of parameters
    nz += problem.ns

    return nz


def get_nlp_dv_structure(problem: yapss.Problem, dtype: type) -> DVStructure[T]:
    """Create the vector of decision variables for the NLP.

    Parameters
    ----------
    problem : yapss.Problem
    dtype : {float, int}

    Returns
    -------
    DVStructure
    """
    spectral_method = problem.spectral_method
    validate_spectral_method(spectral_method)

    dv = DVStructure[T]()
    dv.phase = []

    nz = calculate_nz(problem, spectral_method)
    z: Array[T] = np.zeros([nz], dtype=dtype)
    dv.z = z

    # go back through list of variables, and assign subsets of z to them.
    iz = 0

    var_dict = dv.var_dict = {}
    ns = problem.ns

    for p in range(problem.np):
        dv_phase = DVPhase[T]()
        nx = problem.nx[p]
        nu = problem.nu[p]
        nq = problem.nq[p]

        nc_points = problem.mesh.phase[p].collocation_points
        n_segments = len(nc_points)

        if spectral_method == "lgl":
            nupoints = sum(nc_points) - n_segments + 1
            nxpoints = nupoints
            n = sum(nc_points) + 1
        elif spectral_method == "lgr":
            nupoints = sum(nc_points)
            nxpoints = nupoints + 1
            n = nxpoints
        elif spectral_method == "lg":
            nupoints = sum(nc_points)
            n = sum(nc_points) + n_segments + 1
            nxpoints = sum(nc_points)
        else:
            raise RuntimeError

        if spectral_method in ("lgl", "lgr"):
            dv_phase.x0 = z[iz : iz + nx * n : n]
            dv_phase.xf = z[iz + nxpoints - 1 : iz + nx * n : n]
        elif spectral_method == "lg":
            # TODO: get rid of sum(nc_points)
            dv_phase.x0 = z[iz + sum(nc_points) : iz + sum(nc_points) + nx * n : n]
            dv_phase.xf = z[iz + n - 1 : iz + n - 1 + nx * n : n]
        else:
            raise RuntimeError

        x = dv_phase.x = []
        xs = dv_phase.xs = []
        xc = dv_phase.xc = []
        xa = dv_phase.xa = []

        for _i in range(nx):
            if spectral_method == "lgl":
                x.append(z[iz : iz + nxpoints])
                xa.append(z[iz : iz + nxpoints + n_segments])
                xc.append(z[iz : iz + nupoints])
                iz += nxpoints
                xs.append(z[iz : iz + n_segments])
                iz += n_segments
            elif spectral_method == "lgr":
                x.append(z[iz : iz + nxpoints])
                xa.append(z[iz : iz + nxpoints])
                xc.append(z[iz : iz + nupoints])
                iz += nxpoints
            elif spectral_method == "lg":
                x.append(z[iz : iz + nxpoints])
                xc.append(z[iz : iz + nxpoints])
                xa.append(z[iz : iz + nxpoints + n_segments + 1])
                iz += nxpoints + n_segments + 1
            else:
                raise RuntimeError

        # control variables
        u = dv_phase.u = []
        for _i in range(nu):
            u.append(z[iz : iz + nupoints])
            iz += nupoints

        # integral variables
        dv_phase.q = z[iz : iz + nq]
        iz += nq

        # initial and final times
        dv_phase.t0 = z[iz : iz + 1]
        iz += 1
        dv_phase.tf = z[iz : iz + 1]
        iz += 1

        dv.phase.append(dv_phase)
        last_phase = dv.phase[-1]

        for i in range(nx):
            var_dict[p, "x0", i] = last_phase.x0[i : i + 1]
            var_dict[p, "xf", i] = last_phase.xf[i : i + 1]

        for i in range(nu):
            var_dict[p, "u", i] = last_phase.u[i]

        var_dict[p, "t0", 0] = last_phase.t0
        var_dict[p, "tf", 0] = last_phase.tf

        for i in range(nq):
            var_dict[p, "q", i] = last_phase.q[i : i + 1]

    # parameters
    s = z[iz : iz + problem.ns]

    dv.s = s
    iz += problem.ns

    for p in range(max(problem.np, 1)):
        for i in range(ns):
            key = p, "s", i
            var_dict[key] = s[i : i + 1]

    return dv


def calculate_ic(problem: yapss.Problem, spectral_method: str) -> int:
    ic = 0
    for p in range(problem.np):
        col_points = problem.mesh.phase[p].collocation_points
        nc = sum(col_points)
        nhpoints = nc
        if spectral_method == "lgl":
            nhpoints += -len(col_points) + 1
        ic += nc * problem.nx[p] + nhpoints * problem.nh[p] + problem.nq[p] + 1
        if spectral_method in ("lg", "lgr"):
            ic += len(col_points) * problem.nx[p]
        if spectral_method == "lg":
            ic += len(col_points) * problem.nx[p]
    ic += problem.nd
    return ic


def get_nlp_cf_structure(problem: yapss.Problem, dtype: type) -> CFStructure[T]:
    """Create the vector of constraint variables for the NLP.

    Also creates variables that are slices of the vector, so that the constraint vector
    can be manipulated more easily.

    Parameters
    ----------
    problem : yapss.Problem
    dtype : {float, int}

    Returns
    -------
    CFStructure
    """
    spectral_method = problem.spectral_method
    validate_spectral_method(spectral_method)

    cf: CFStructure[T] = CFStructure[T]()
    cf.phase = [CFPhase[T]() for _ in range(problem.np)]

    # define constraint array
    ic = calculate_ic(problem, spectral_method)
    c: Array[T] = np.zeros([ic], dtype=dtype)
    cf.c = c

    # now go back through and assign subsets of constraint function array to variables
    ic = 0
    for p in range(problem.np):
        # extract phase information. n is number of u collocation points. m is number
        # of x collocation points
        col_points = problem.mesh.phase[p].collocation_points
        nc = sum(col_points)

        # defect in state equations
        cf.phase[p].defect = []
        for _ in range(problem.nx[p]):
            cf.phase[p].defect.append(c[ic : ic + nc])
            ic += nc

        if spectral_method == "lg":
            lg_defect: list[Array[T]] = []
            cf.phase[p].lg_defect = lg_defect
            for _ in range(problem.nx[p]):
                lg_defect.append(c[ic : ic + len(col_points)])
                ic += len(col_points)

        if spectral_method == "lgl":
            defect_index: list[int] = []
            i = 0
            for m in col_points:
                defect_index += range(i, i + m)
                i += m - 1
            cf.phase[p].defect_index = defect_index

        # path constraints
        nhpoints = nc
        if spectral_method == "lgl":
            nhpoints += -len(col_points) + 1
        cf.phase[p].path = []
        for _ in range(problem.nh[p]):
            cf.phase[p].path.append(c[ic : ic + nhpoints])
            ic += nhpoints

        # integral equations
        nq = problem.nq[p]
        cf.phase[p].integral = c[ic : ic + nq]
        ic += nq

        # duration
        cf.phase[p].duration = c[ic : ic + 1]
        ic += 1

        # lgr zero mode
        if spectral_method in ("lg", "lgr"):
            cf.phase[p].zero_mode = []
            nz = len(col_points)
            for _ in range(problem.nx[p]):
                cf.phase[p].zero_mode.append(c[ic : ic + nz])
                ic += nz

    n = problem.nd
    cf.discrete = c[ic : ic + n]
    ic += n

    if ic != len(c):
        raise RuntimeError

    return cf
