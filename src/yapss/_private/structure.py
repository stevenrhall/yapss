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
from typing import TYPE_CHECKING, Generic, TypeVar, assert_never, get_args

# third party imports
import numpy as np

# package imports
from .layout import problem_layout
from .types_ import CFViewName, DVViewName

# Define a generic type variable
T = TypeVar("T", bound=np.generic)

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import NDArray

    from .spec import ProblemSpec
    from .types_ import CVIndex, CVName, DVKey, PhaseIndex

    # package imports

    # Generic array type
    Array = NDArray[T]
    """Generic array type."""


class DVPhase(Generic[T], SimpleNamespace):
    """Container for decision variables of a single phase.

    Every view is laid out by the phase's `layout.PhaseLayout`, identically for every
    spectral method; a view a method does not use is empty.

    Attributes
    ----------
    xa : list[Array]
        Everything stored for each state: its values at the time points, then its zero modes.
    x : list[Array]
        Each state's values at the time points, in storage order (see
        ``PhaseLayout.time_order`` for time order).
    xc : list[Array]
        Each state's values at the evaluation points, the first stored time points.
    xs : list[Array]
        Each state's zero modes (LGL only; empty otherwise).
    x0, xf : Array
        The initial and final value of every state, one entry per state.
    u : list[Array]
        Each control's values at the evaluation points.
    q : Array
        The integrals.
    t0, tf : Array
        The initial and final time, one entry each.
    """

    t0: Array[T]
    tf: Array[T]
    xa: list[Array[T]]
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

    Every view is laid out by the phase's `layout.PhaseLayout`, identically for every
    spectral method; a view a method does not use is empty.

    Attributes
    ----------
    defect : list[Array]
        Each state's defect rows, one per collocation point.
    lg_defect : list[Array]
        Each state's boundary defect rows, one per segment (LG only; empty otherwise).
    defect_index : NDArray[np.intp]
        For each defect row, the evaluation point whose dynamics it reads.
    path : list[Array]
        Each path constraint's rows, one per evaluation point.
    integral : Array
    duration : Array
    """

    defect: list[Array[T]]
    lg_defect: list[Array[T]]
    defect_index: NDArray[np.intp]
    path: list[Array[T]]
    integral: Array[T]
    duration: Array[T]


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


def get_nlp_dv_structure(problem: ProblemSpec, dtype: type) -> DVStructure[T]:
    """Create the vector of decision variables for the NLP, with views into it.

    Per phase, in order: every state's stored values (time points, then zero modes), every
    control's values, the integrals, t0, tf. The parameters follow the last phase.

    Parameters
    ----------
    problem : ProblemSpec
    dtype : {float, int}

    Returns
    -------
    DVStructure
    """
    layouts = problem_layout(problem)
    nz = problem.ns + sum(
        problem.nx[p] * layout.n_state_storage + problem.nu[p] * layout.n_eval + problem.nq[p] + 2
        for p, layout in enumerate(layouts)
    )

    dv = DVStructure[T]()
    dv.phase = []
    z: Array[T] = np.zeros([nz], dtype=dtype)
    dv.z = z
    var_dict = dv.var_dict = {}

    iz = 0
    for p, layout in enumerate(layouts):
        nx, nu, nq = problem.nx[p], problem.nu[p], problem.nq[p]
        n_storage, n_time, n_eval = layout.n_state_storage, layout.n_time, layout.n_eval
        dv_phase = DVPhase[T]()

        # states: each occupies n_storage consecutive entries, so the boundary values of
        # all states are strided views
        states_end = iz + nx * n_storage
        dv_phase.xa = [z[iz + i * n_storage : iz + (i + 1) * n_storage] for i in range(nx)]
        dv_phase.x = [xa[:n_time] for xa in dv_phase.xa]
        dv_phase.xc = [xa[:n_eval] for xa in dv_phase.xa]
        dv_phase.xs = [xa[n_time:] for xa in dv_phase.xa]
        dv_phase.x0 = z[iz + layout.x0_position : states_end : n_storage]
        dv_phase.xf = z[iz + layout.xf_position : states_end : n_storage]
        iz = states_end

        dv_phase.u = [z[iz + i * n_eval : iz + (i + 1) * n_eval] for i in range(nu)]
        iz += nu * n_eval

        dv_phase.q = z[iz : iz + nq]
        iz += nq
        dv_phase.t0 = z[iz : iz + 1]
        dv_phase.tf = z[iz + 1 : iz + 2]
        iz += 2

        dv.phase.append(dv_phase)
        for i in range(nx):
            var_dict[p, "x0", i] = dv_phase.x0[i : i + 1]
            var_dict[p, "xf", i] = dv_phase.xf[i : i + 1]
        for i in range(nu):
            var_dict[p, "u", i] = dv_phase.u[i]
        var_dict[p, "t0", 0] = dv_phase.t0
        var_dict[p, "tf", 0] = dv_phase.tf
        for i in range(nq):
            var_dict[p, "q", i] = dv_phase.q[i : i + 1]

    # parameters
    dv.s = z[iz : iz + problem.ns]
    for p in range(max(problem.np, 1)):
        for i in range(problem.ns):
            var_dict[p, "s", i] = dv.s[i : i + 1]

    return dv


def get_nlp_cf_structure(problem: ProblemSpec, dtype: type) -> CFStructure[T]:
    """Create the vector of constraint functions for the NLP, with views into it.

    Per phase, in order: every state's defect rows, every state's boundary defect rows,
    every path constraint's rows, the integral rows, the duration row. The discrete
    constraints follow the last phase.

    Parameters
    ----------
    problem : ProblemSpec
    dtype : {float, int}

    Returns
    -------
    CFStructure
    """
    layouts = problem_layout(problem)
    nc = problem.nd + sum(
        problem.nx[p] * (layout.n_collocation + layout.n_boundary_defect)
        + problem.nh[p] * layout.n_eval
        + problem.nq[p]
        + 1
        for p, layout in enumerate(layouts)
    )

    cf: CFStructure[T] = CFStructure[T]()
    cf.phase = []
    c: Array[T] = np.zeros([nc], dtype=dtype)
    cf.c = c

    ic = 0
    for p, layout in enumerate(layouts):
        nx, nh, nq = problem.nx[p], problem.nh[p], problem.nq[p]
        n_defect, n_boundary, n_eval = (
            layout.n_collocation,
            layout.n_boundary_defect,
            layout.n_eval,
        )
        cf_phase = CFPhase[T]()

        cf_phase.defect = [c[ic + i * n_defect : ic + (i + 1) * n_defect] for i in range(nx)]
        ic += nx * n_defect
        cf_phase.lg_defect = [c[ic + i * n_boundary : ic + (i + 1) * n_boundary] for i in range(nx)]
        ic += nx * n_boundary
        cf_phase.defect_index = layout.defect_index

        cf_phase.path = [c[ic + i * n_eval : ic + (i + 1) * n_eval] for i in range(nh)]
        ic += nh * n_eval

        cf_phase.integral = c[ic : ic + nq]
        ic += nq
        cf_phase.duration = c[ic : ic + 1]
        ic += 1
        cf.phase.append(cf_phase)

    cf.discrete = c[ic : ic + problem.nd]
    return cf


def nlp_variable_keys(problem: ProblemSpec) -> NDArray[np.object_]:
    """Return the key of every NLP decision variable, in NLP order.

    ``keys[k] == (p, view, i, j)`` says that ``z[k]`` is ``dv.phase[p].<view>[i][j]`` in the
    structure `get_nlp_dv_structure` returns (``dv.s[i]`` for a parameter, keyed with phase
    0). The position ``j`` is the storage position in the view, not a time index; see
    ``PhaseLayout.time_order``. The keys are written through the NLP's own views, into the
    views named by `DVViewName`, so the map cannot disagree with the layout.
    """
    dv: DVStructure[np.object_] = get_nlp_dv_structure(problem, object)
    for view in get_args(DVViewName):
        for p in (0,) if view == "s" else range(problem.np):
            _write_keys(_variable_components(dv, p, view), p, view)
    return dv.z


def nlp_constraint_keys(problem: ProblemSpec) -> NDArray[np.object_]:
    """Return the key of every NLP constraint, in NLP order.

    ``keys[k] == (p, view, i, j)`` says that ``c[k]`` is ``cf.phase[p].<view>[i][j]`` in the
    structure `get_nlp_cf_structure` returns (``cf.discrete[i]`` for a discrete constraint,
    keyed with phase 0). See `nlp_variable_keys`.
    """
    cf: CFStructure[np.object_] = get_nlp_cf_structure(problem, object)
    for view in get_args(CFViewName):
        for p in (0,) if view == "discrete" else range(problem.np):
            _write_keys(_constraint_components(cf, p, view), p, view)
    return cf.c


def _variable_components(
    dv: DVStructure[np.object_],
    p: int,
    view: DVViewName,
) -> list[Array[np.object_]]:
    """Return one array per component of a decision variable view."""
    match view:
        case "x":
            components = dv.phase[p].x
        case "xs":
            components = dv.phase[p].xs
        case "u":
            components = dv.phase[p].u
        case "q":
            components = _scalars(dv.phase[p].q)
        case "t0":
            components = [dv.phase[p].t0]
        case "tf":
            components = [dv.phase[p].tf]
        case "s":
            components = _scalars(dv.s)
        case _:
            assert_never(view)
    return components


def _constraint_components(
    cf: CFStructure[np.object_],
    p: int,
    view: CFViewName,
) -> list[Array[np.object_]]:
    """Return one array per component of a constraint view."""
    match view:
        case "defect":
            components = cf.phase[p].defect
        case "lg_defect":
            components = cf.phase[p].lg_defect
        case "path":
            components = cf.phase[p].path
        case "integral":
            components = _scalars(cf.phase[p].integral)
        case "duration":
            components = [cf.phase[p].duration]
        case "discrete":
            components = _scalars(cf.discrete)
        case _:
            assert_never(view)
    return components


def _scalars(array: Array[np.object_]) -> list[Array[np.object_]]:
    """Split a view whose components are single entries into one length-1 view each."""
    return [array[i : i + 1] for i in range(array.size)]


def _write_keys(components: list[Array[np.object_]], p: int, view: str) -> None:
    """Write ``(p, view, i, j)`` into entry ``j`` of component ``i``."""
    for i, array in enumerate(components):
        for j in range(array.size):
            array[j] = (p, view, i, j)
