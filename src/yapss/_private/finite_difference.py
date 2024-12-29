"""
Determine finite difference structures for finding numerical derivatives.

This module predetermines the structures required for calculating derivatives using central
differences. The first-order derivative structures (i.e., the nonzero elements of the objective
gradient, the discrete Jacobian, and the continuous Jacobian functions) are determined by the
functions:

* `get_objective_gradient_structure_nan`
* `get_continuous_jacobian_structure_nan`

The iteration patterns used by the first-order finite difference routines are determined by the
functions:

* `get_continuous_jacobian_fd_structure`
* `get_discrete_jacobian_fd_structure`

The iteration patterns used by the second-order finite difference routines are determined by the
functions:

* `get_objective_hessian_structure`
* `get_discrete_hessian_structure`
* `get_continuous_hessian_structure`
"""

# ruff: noqa: PERF401
# future imports
from __future__ import annotations

# standard imports
import math
from collections import defaultdict
from typing import TYPE_CHECKING, cast

# third party imports
import numpy  # noqa: ICN001
import numpy as np

# package imports
from .input_args import (
    ContinuousArg,
    ContinuousFunctionFloat,
    DiscreteArg,
    DiscreteFunctionFloat,
    ObjectiveArg,
    ObjectiveFunctionFloat,
    ProblemFunctions,
)
from .structure import DVStructure, get_nlp_dv_structure

if TYPE_CHECKING:
    # third party imports
    from numpy.typing import ArrayLike, NDArray

    # package imports
    import yapss

    from .types_ import (
        CHFDS,
        CHS,
        CJFDS,
        CJS,
        DHFDS,
        DHS,
        DJFDS,
        DJS,
        OGS,
        OHS,
        CFKey,
        CHSPhase,
        CHSTerm,
        CJSTerm,
        CVKey,
        CVName,
        DFIndex,
        DJSTerm,
        DVIndex,
        DVKey,
        DVName,
        PhaseIndex,
    )

    InnerDict = defaultdict[DVKey, list[DFIndex]]
    OuterDict = defaultdict[DVKey, InnerDict]


def get_continuous_jacobian_structure_nan(
    problem: yapss.Problem,
    z0: NDArray[numpy.float64],
) -> CJS:
    """Deduce the Jacobian structure of the continuous constraint function.

    Deduces the Jacobian structure of the continuous constraint function, using
    the NaN method.

    Returns
    -------
    cjs : tuple of tuple of jacobian keys

    Notes
    -----
    `cjs[p]` is a tuple of jacobian keys for the phase `p` for the continuous
    constraint jacobian. A typical key would be, say, `(("f", 2), ("u", 3))`,
    meaning the partial derivative of `f[2]` (the dynamics of the state `x[2]`)
    with respect to the control input `u[3]`. Because the time variable is a
    scalar, it is represented by `('t', 0)`.
    """
    nan = float("nan")

    # sort_key gives the sort order for jacobian keys
    item_dict = {"s": 0, "x": 1, "u": 2, "t": 3}

    # is this sort key correct?
    def sort_key(
        item: tuple[tuple[str, int], tuple[str, int]],
    ) -> tuple[str, int, int, int]:
        (a, b), (c, d) = item
        return a, b, item_dict[c], d

    from .structure import get_nlp_dv_structure

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)
    arg: ContinuousArg[np.float64] = ContinuousArg(problem, dv, dtype=numpy.float64)
    arg._dv.z[:] = z0

    cjs = []

    ns = problem.ns
    parameter = [arg.parameter[i : i + 1] for i in range(ns)]

    for p in range(problem.np):
        continuous = cast(ContinuousFunctionFloat, problem.functions.continuous)
        arg._phase_list = (p,)
        phase = arg.phase[p]

        cjs_phase: list[CJSTerm] = []

        nx = problem.nx[p]
        nu = problem.nu[p]
        nq = problem.nq[p]
        nh = problem.nh[p]

        var: CVName
        for var in ("s", "x", "u", "t"):
            v: ArrayLike
            if var == "x":
                n, v = nx, phase.state
            elif var == "u":
                n, v = nu, phase.control
            elif var == "t":
                n, v = 1, [phase.time]
            elif var == "s":
                n, v = ns, parameter
            else:
                raise RuntimeError

            for j in range(n):
                w = v[j].copy()
                v[j][:] = nan

                # pass to continuous and see which outputs are affected
                continuous(arg)

                for i in range(nx):
                    if numpy.any(numpy.isnan(phase.dynamics[i])):
                        cjs_phase.append((("f", i), (var, j)))
                for i in range(nq):
                    if numpy.any(numpy.isnan(phase.integrand[i])):
                        cjs_phase.append((("g", i), (var, j)))
                for i in range(nh):
                    if numpy.any(numpy.isnan(phase.path[i])):
                        cjs_phase.append((("h", i), (var, j)))

                v[j][:] = w

        cjs_phase.sort(key=sort_key)
        cjs.append(tuple(cjs_phase))

    return tuple(cjs)


def get_objective_gradient_structure_nan(
    problem: yapss.Problem,
    z0: NDArray[numpy.float64],
) -> tuple[OGS, DJS]:
    """Deduce the objective gradient structure of the discrete constraint function.

    Uses the NaN method.

    Returns
    -------
    The objective gradient structure, in the form of a structure tuple

    The discrete constraint function Jacobian structure, in the form of a
    structured tuple
    """
    v: DVName
    d: float

    objective_function = cast(ObjectiveFunctionFloat, problem.functions.objective)

    var_names = ("t0", "tf", "x0", "xf", "q")

    nan = float("nan")
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, numpy.float64)
    arg: ObjectiveArg[numpy.float64] = ObjectiveArg(problem, dv, dtype=numpy.float64)
    discrete_arg: DiscreteArg[numpy.float64] = DiscreteArg(problem, dv, numpy.float64)
    dv.z[:] = z0

    np = problem.np

    ogs: list[DVKey] = []
    djs: list[tuple[DFIndex, DVKey]] = []

    for p in range(np):
        nx = problem.nx[p]
        nq = problem.nq[p]

        for v in var_names:
            var: NDArray[numpy.float64]
            if v == "x0":
                var = dv.phase[p].x0
                n = nx
            elif v == "xf":
                var = dv.phase[p].xf
                n = nx
            elif v == "t0":
                var = dv.phase[p].t0
                n = 1
            elif v == "tf":
                var = dv.phase[p].tf
                n = 1
            elif v == "q":
                var = dv.phase[p].q
                n = nq
            else:
                raise RuntimeError

            for j in range(n):
                z = var[j]
                var[j] = nan
                objective_function(arg)
                objective = arg.objective

                if math.isnan(objective):
                    ogs.append((p, v, j))

                if problem.nd > 0:
                    discrete_function = cast(DiscreteFunctionFloat, problem.functions.discrete)
                    discrete_function(discrete_arg)
                    discrete = discrete_arg.discrete

                    for i, d in enumerate(discrete):
                        if math.isnan(d):
                            djs.append((i, (p, v, j)))

                var[j] = z

    ns = problem.ns

    for j in range(ns):
        # perturb variables
        z = arg.parameter[j]
        arg.parameter[j] = nan

        # determine if objective is affected
        objective_function(arg)
        objective = arg.objective
        if math.isnan(objective):
            ogs.append((0, "s", j))

        # determine which discrete constraints are affected
        if problem.nd > 0:
            discrete_function = cast(DiscreteFunctionFloat, problem.functions.discrete)
            discrete_function(discrete_arg)
            discrete = discrete_arg.discrete

            for i, d in enumerate(discrete):
                if math.isnan(d):
                    dv_key = (0, "s", j)
                    djs.append((i, dv_key))

        arg.parameter[j] = z

    djs.sort(key=djs_sort_key)
    return tuple(ogs), tuple(djs)


def get_discrete_hessian_structure(discrete_jacobian_structure: DJS) -> tuple[DHS, DHFDS]:
    """Find the discrete Hessian structure and also the finite difference structure."""
    # make structure for doing iteration
    discrete_hessian_dict: OuterDict = defaultdict(lambda: defaultdict(list))

    for i, (d1, key1) in enumerate(discrete_jacobian_structure):
        for d2, key2 in discrete_jacobian_structure[i:]:
            if d1 == d2:
                key = [key1, key2]
                key.sort()
                discrete_hessian_dict[key[0]][key[1]].append(d1)

    key1_list = list(discrete_hessian_dict.keys())
    key1_list.sort()
    outer_list = []
    dhs = []
    for key1 in key1_list:
        inner_dict = discrete_hessian_dict[key1]
        key2_list = list(inner_dict.keys())
        key2_list.sort()
        inner_list = []
        for key2 in key2_list:
            d_list = inner_dict[key2]
            d_list.sort()
            for d in d_list:
                dhs.append((d, key1, key2))
            inner_list.append((key2, tuple(d_list)))
        outer_list.append((key1, tuple(inner_list)))
    dh_iter_structure = tuple(outer_list)

    return tuple(dhs), dh_iter_structure


def get_objective_hessian_structure(ogs: OGS) -> OHS:
    """Find the objective hessian structure."""
    objective_hessian_structure = []

    for i, key1 in enumerate(ogs):
        objective_hessian_structure += [(key1, key2) for key2 in ogs[i:]]

    return tuple(objective_hessian_structure)


def get_continuous_hessian_structure(cjs: CJS) -> tuple[CHS, CHFDS]:
    """Infer the continuous function Hessian structure.

    It is assumed that the result from calling
    get_continuous_jacobian_structure_nan(), stored in
    problem._continuous_jacobian_structure,
    has not been modified by the user or another method.
    """
    jacobian_dict: defaultdict[CFKey, list[CVKey]]
    continuous_hessian_structure_cd = []
    continuous_hessian_structure: list[CHSPhase] = []

    # for each phase ...
    for phase_jacobian_structure in cjs:
        # ... find the variables that affect each continuous constraint
        jacobian_dict = defaultdict(list)
        for key_pair in phase_jacobian_structure:
            cf_key, cv_key = key_pair
            jacobian_dict[cf_key].append(cv_key)

        for value in jacobian_dict.values():
            value.sort()

        # find pairs of variables and which functions they affect
        pairs: defaultdict[tuple[CVKey, CVKey], list[CFKey]] = defaultdict(list)

        for cf_key, cv_keys in jacobian_dict.items():
            for i, cv_key1 in enumerate(cv_keys):
                for cv_key2 in cv_keys[i:]:
                    cv_pair = (cv_key1, cv_key2)
                    pairs[cv_pair].append(cf_key)

        # sort and construct phase structure
        phase_structure: list[CHSTerm] = []
        keys = list(pairs.keys())
        keys.sort()
        phase_structure_cd = []
        for key_pair2 in keys:
            phase_structure_cd.append((key_pair2, tuple(pairs[key_pair2])))
            for fcn in pairs[key_pair2]:
                phase_structure.append((fcn, key_pair2[0], key_pair2[1]))

        continuous_hessian_structure.append(tuple(phase_structure))
        continuous_hessian_structure_cd.append(tuple(phase_structure_cd))

    return tuple(continuous_hessian_structure), tuple(continuous_hessian_structure_cd)


def get_continuous_jacobian_fd_structure(cjs: CJS) -> CJFDS:
    """Derive the continuous Jacobian finite difference structure.

    Parameters
    ----------
    cjs: CJS
        Continuous Jacobian structure, that is, the Jacobian structure of the
        user-defined continuous constraint function

    Returns
    -------
    cjfds : CJFDS
        The continuous Jacobian finite difference structure
    """
    cjfds = []

    cv_cf_dict: defaultdict[CVKey, list[CFKey]]
    for cjs_phase in cjs:
        cv_cf_dict = defaultdict(list)

        for cf_key, cv_key in cjs_phase:
            cv_cf_dict[cv_key].append(cf_key)

        cv_keys = list(cv_cf_dict.keys())
        cv_keys.sort(key=continuous_sort_key)

        cjfds_phase = []
        for cv_key in cv_keys:
            cf_key_list = cv_cf_dict[cv_key]
            cf_key_list.sort()
            cjfds_phase.append((cv_key, tuple(cf_key_list)))

        cjfds.append(tuple(cjfds_phase))

    return tuple(cjfds)


def get_discrete_jacobian_fd_structure(djs: DJS) -> DJFDS:
    """Derive the discrete Jacobian finite difference structure.

    Parameters
    ----------
    djs : DJS
        Discrete Jacobian structure, that is, the Jacobian structure of the
        user-defined discrete constraint function.

    Returns
    -------
    djfds : DJFDS
        The discrete Jacobian finite difference structure
    """
    key_index_dict: defaultdict[DVKey, list[DFIndex]] = defaultdict(list)

    for index, var_key in djs:
        key_index_dict[var_key].append(index)

    var_keys = list(key_index_dict.keys())
    var_keys.sort(key=discrete_sort_key)

    discrete_jacobian_fd_structure = []
    for var_key in var_keys:
        key1_list = key_index_dict[var_key]
        key1_list.sort()
        discrete_jacobian_fd_structure.append((var_key, tuple(key1_list)))

    return tuple(discrete_jacobian_fd_structure)


def make_fd_structure(problem: yapss.Problem, z0: NDArray[numpy.float64]) -> ProblemFunctions:
    """Make finite difference structures for the optimal control problem functions.

    Parameters
    ----------
    problem : Problem
    z0 : NDArray

    Returns
    -------
    ProblemFunctions
    """
    fd_structure: ProblemFunctions = ProblemFunctions()
    order = problem.derivatives.order

    # copy functions from problem
    # TODO: is this necessary?
    if problem.functions.objective is None:
        msg = "Problem has no objective function"
        raise RuntimeError(msg)
    fd_structure.objective = problem.functions.objective
    if problem.np > 0:
        if problem.functions.continuous is None:
            msg = "Problem has dynamic phases but no continuous function"
            raise RuntimeError(msg)
        fd_structure.continuous = problem.functions.continuous
    if problem.nd > 0:
        if problem.functions.discrete is None:
            msg = "Problem has discrete variables but no discrete function"
            raise RuntimeError(msg)
        fd_structure.discrete = problem.functions.discrete

    # first derivatives
    method = problem.derivatives.method
    if method == "central-difference-full":
        cjs = get_continuous_jacobian_structure_full(problem)
        ogs, djs = get_objective_gradient_structure_full(problem)
    elif method == "central-difference":
        cjs = get_continuous_jacobian_structure_nan(problem, z0)
        ogs, djs = get_objective_gradient_structure_nan(problem, z0)
    else:
        msg = f"Unexpected derivatives.method = '{method}'"
        raise RuntimeError(msg)

    fd_structure.continuous_jacobian_structure = cjs
    fd_structure.objective_gradient_structure = ogs
    fd_structure.discrete_jacobian_structure = djs

    cjfds = get_continuous_jacobian_fd_structure(cjs)
    fd_structure.continuous_jacobian_structure_cd = cjfds
    djfds = get_discrete_jacobian_fd_structure(djs)
    fd_structure.discrete_jacobian_structure_cd = djfds

    if order == "second":
        # discrete hessian
        dhs, dhs_fd = get_discrete_hessian_structure(djs)
        fd_structure.discrete_hessian_structure = dhs
        fd_structure.discrete_hessian_structure_cd = dhs_fd

        # objective hessian
        fd_structure.objective_hessian_structure = get_objective_hessian_structure(ogs)

        # continuous hessian
        chs, chs_cd = get_continuous_hessian_structure(cjs=cjs)
        fd_structure.continuous_hessian_structure = chs
        fd_structure.continuous_hessian_structure_cd = chs_cd

    return fd_structure


# fmt: off
sort_dict = {"s": 0, "x": 1, "u": 2, "t": 3, "f": 11, "g": 12, "h": 13,
             "x0": 21, "xf": 22, "q": 23, "t0": 24, "tf": 25}  # fmt:on


def discrete_sort_key(dvkey: DVKey) -> tuple[PhaseIndex, int, DVIndex]:
    """Define sort key function for discrete variables.

    Parameters
    ----------
    dvkey : DVKey
        The discrete variable key (a tuple)

    Returns
    -------
    tuple[PhaseIndex, int, DVIndex]
        The dvkey tuple, with the variable name replaced  by an integer to give the
        desired lexical order
    """
    p, var, i = dvkey
    return p, sort_dict[var], i


def djs_sort_key(dvkey: DJSTerm) -> tuple[PhaseIndex, int, DVIndex, DFIndex]:
    """Define sort key function for discrete variables.

    Parameters
    ----------
    dvkey : DVKey
        The discrete variable key (a tuple)

    Returns
    -------
    tuple[PhaseIndex, int, DVIndex]
        The dvkey tuple, with the variable name replaced  by an integer to give the
        desired lexical order
    """
    p: int
    d, (p, var, i) = dvkey
    return p, sort_dict[var], i, d


def continuous_sort_key(item: tuple[str, int]) -> tuple[int, int]:
    """Define the sort key for the continuous variables keys of type CVKey.

    Parameters
    ----------
    item : CVKey

    Returns
    -------
    tuple[int, int]
    """
    var, index = item
    return sort_dict[var], index


def get_continuous_jacobian_structure_full(problem: yapss.Problem) -> CJS:
    """Deduce the Jacobian structure for full derivatives.

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    CJS
        The continuous Jacobian structure.
    """
    # sort_key gives the sort order for jacobian keys
    item_dict = {"s": 0, "x": 1, "u": 2, "t": 3}

    # is this sort key correct?
    def sort_key(
        item: tuple[tuple[str, int], tuple[str, int]],
    ) -> tuple[str, int, int, int]:
        (a, b), (c, d) = item
        return a, b, item_dict[c], d

    cjs = []

    ns = problem.ns

    for p in range(problem.np):
        cjs_phase: list[CJSTerm] = []

        nx = problem.nx[p]
        nu = problem.nu[p]
        nq = problem.nq[p]
        nh = problem.nh[p]

        for var, n in (("s", ns), ("x", nx), ("u", nu), ("t", 1)):
            for j in range(n):
                for label, count in (("f", nx), ("g", nq), ("h", nh)):
                    for i in range(count):
                        cjs_phase.append(((label, i), (var, j)))

        cjs_phase.sort(key=sort_key)
        cjs.append(tuple(cjs_phase))

    return tuple(cjs)


def get_objective_gradient_structure_full(problem: yapss.Problem) -> tuple[OGS, DJS]:
    """Deduce the objective gradient structure.

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    tuple[OGS, DJS]
        The objective gradient structure, in the form of a structured tuple
    """
    ogs: list[DVKey] = []
    djs: list[tuple[DFIndex, DVKey]] = []

    np = problem.np
    ns = problem.ns

    # Iterate over phases and add to ogs and djs based on structure only
    for p in range(np):
        nx = problem.nx[p]
        nq = problem.nq[p]

        # Append entries for each variable count without referencing any actual variables
        for var_name, n in [("x0", nx), ("xf", nx), ("t0", 1), ("tf", 1), ("q", nq)]:
            for j in range(n):
                ogs.append((p, var_name, j))
                for i in range(problem.nd):
                    djs.append((i, (p, var_name, j)))

    # Add entries for "s" parameter without evaluating functions
    for j in range(ns):
        ogs.append((0, "s", j))
        for i in range(problem.nd):
            djs.append((i, (0, "s", j)))

    djs.sort(key=djs_sort_key)
    return tuple(ogs), tuple(djs)
