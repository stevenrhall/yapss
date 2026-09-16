"""

Make optimal control problem derivative callback functions using finite difference methods.

This module makes the objective gradient, objective hessian, continuous Jacobian,
continuous Hessian, discrete Jacobian, and discrete Hessian callback function, using
either central differences.

"""

# future imports
from __future__ import annotations

__all__ = ["make_cd_functions"]
from collections.abc import Callable

# standard imports
from itertools import product
from typing import TYPE_CHECKING, cast

# third party imports
import numpy as np

# package imports
from .finite_difference import make_fd_structure
from .input_args import (
    ContinuousFunctionFloat,
    ContinuousHessianArg,
    ContinuousHessianFunction,
    ContinuousJacobianArg,
    ContinuousJacobianFunction,
    ContinuousStore,
    DiscreteArg,
    DiscreteFunctionFloat,
    DiscreteHessianArg,
    DiscreteJacobianArg,
    DiscreteJacobianFunction,
    ObjectiveArg,
    ObjectiveFunctionFloat,
    ObjectiveGradientArg,
    ObjectiveGradientFunction,
    ObjectiveHessianArg,
    ObjectiveHessianFunction,
    ProblemFunctions,
    call_callback,
)
from .structure import DVStructure, get_nlp_dv_structure
from .types_ import CFKey, PhaseIndex, set_private

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable, Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .problem import Problem, Scale
    from .types_ import CHFDS, CJFDS, DHFDS, DJFDS, OGS, DVKey

    Array = NDArray[np.float64]

# EPS should be 2 ** -53, but calculate to be sure
exponent: int = 1
while 1 - 2 ** float(-exponent) < 1:
    exponent += 1
exponent -= 1
EPS: float = 2 ** (-exponent)

# step sizes for first and second differences
DELTA1: np.float64 = (3 * EPS) ** (1 / 3)
DELTA2: np.float64 = (3 * EPS) ** (1 / 4)


def make_cd_functions(
    problem: yapss.Problem,
    z0: Array,
    tau_u: Sequence[NDArray[np.float64]],
) -> ProblemFunctions:
    """Make derivative callback functions and structures.

    Make callback functions for the gradients, Jacobians, and Hessians (first
    and second derivatives) of the user-defined objective, discrete, and
    continuous functions, using central difference methods.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    z0 : numpy.ndarray
        Initial decision variables.

    Returns
    -------
    ProblemFunctions
        The structure containing the callback functions.
    """
    cd_functions = make_fd_structure(problem, z0, tau_u)
    order = problem.derivatives.order

    # first derivatives
    ogs: OGS = cd_functions.objective_gradient_structure
    cjfds: CJFDS = cd_functions.continuous_jacobian_structure_cd
    cd_functions.objective_gradient = make_objective_gradient(problem, ogs)
    cd_functions.continuous_jacobian = make_continuous_jacobian(problem, cjfds, tau_u)
    djfds: DJFDS = cd_functions.discrete_jacobian_structure_cd
    cd_functions.discrete_jacobian = make_discrete_jacobian(problem, djfds)

    if order == "second":
        # discrete hessian
        dhfds: DHFDS | None = cd_functions.discrete_hessian_structure_cd

        # mypy hinting
        assert dhfds is not None
        cd_functions.discrete_hessian = make_discrete_hessian(problem, dhfds)

        # objective hessian
        cd_functions.objective_hessian = make_objective_hessian(problem, ogs)

        # continuous hessian
        chfds: CHFDS | None = cd_functions.continuous_hessian_structure_cd
        assert chfds is not None
        cd_functions.continuous_hessian = make_continuous_hessian(problem, chfds, tau_u)

    return cd_functions


def make_objective_gradient(
    problem: Problem,
    ogs: OGS,
) -> ObjectiveGradientFunction:
    """Generate objective gradient callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object
    ogs : OGS
        Finite difference structure for the objective gradient.

    Returns
    -------
    Callable[ObjectiveArg]
        The objective gradient callback function
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=np.float64)
    arg1: ObjectiveArg[np.float64] = ObjectiveArg(problem, dv, dtype=np.float64)
    scale: Scale = problem.scale

    objective_function = cast(ObjectiveFunctionFloat, problem.functions.objective)

    def objective_gradient(arg: ObjectiveGradientArg) -> None:
        """Evaluate the objective gradient via central difference."""
        dv.z[:] = arg._dv.z

        for dv_key in ogs:
            var = dv.var_dict[dv_key][:1]
            w = var[0]

            # central difference
            delta_objective = np.float64(0.0)
            d: np.float64 = scale[dv_key] * DELTA1
            for i in (-1, 1):
                var[0] = w + i * d
                call_callback(objective_function, arg1)
                delta_objective += i * arg1.objective
                arg.gradient[dv_key] = float(delta_objective / (2 * d))

            var[0] = w

        # end of objective_gradient callback function

    return objective_gradient


def make_continuous_jacobian(
    problem: yapss.Problem,
    cjfds: CJFDS,
    tau_u: Sequence[NDArray[np.float64]],
) -> ContinuousJacobianFunction:
    """Generate continuous Jacobian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    cjfds : CJFDS
        Finite difference structure for the continuous Jacobian.
    tau_u : Sequence[NDArray[np.float64]]
        Non-dimensional collocation time points.

    Returns
    -------
    Callable[[ContinuousJacobianArg], None]
        The continuous Jacobian callback function.
    """
    scale: Scale = problem.scale
    if problem.np > 0:
        continuous = cast(ContinuousFunctionFloat, problem.functions.continuous)
    # The stencil runs on a private argument, as the Hessian's does, so the caller's
    # function values are never perturbed. Through 0.2.2 it perturbed the caller's
    # argument and re-evaluated the user function once more at the end to restore
    # them -- one extra evaluation per Jacobian call.
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=np.float64)
    store2: ContinuousStore[np.float64] = ContinuousStore(
        problem, dv, dtype=np.float64, tau_u=tau_u
    )
    arg2 = store2.value_arg

    def continuous_jacobian(arg: ContinuousJacobianArg) -> None:
        """Calculate continuous Jacobian using finite differences.

        Parameters
        ----------
        arg : ContinuousJacobianArg
        """
        store2._sync(arg._dv.z)

        for p in [PhaseIndex(p) for p in arg.phase_list]:
            jacobian = arg.phase[p].jacobian
            set_private(store2, "_phase_list", (p,))
            ne = len(store2.phase[p].time)

            for cv_key, cf_keys in cjfds[p]:
                var2, i1 = cv_key
                var = store2[p, var2, i1]
                w = var.copy()
                for cf_key in cf_keys:
                    jacobian[cf_key, cv_key] = np.zeros(ne)

                # central difference
                d = scale[p, var2, i1] * DELTA1

                for j in (-1, 1):
                    var[:] = w + j * d
                    call_callback(continuous, arg2)

                    for cf_key in cf_keys:
                        var1, i = cf_key
                        jacobian[cf_key, cv_key] += j * store2[p, var1, i] / (2 * d)

                var[:] = w

        # end of continuous_jacobian callback function

    return continuous_jacobian


def make_discrete_jacobian(problem: yapss.Problem, djfds: DJFDS) -> DiscreteJacobianFunction:
    """Generate discrete Jacobian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    djfds : DJFDS
        Finite difference structure for the discrete Jacobian.

    Returns
    -------
    Callable[[DiscreteJacobianArg], None]
        The discrete Jacobian callback function.
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=float)
    discrete_arg: DiscreteArg[np.float64] = DiscreteArg(problem, dv, dtype=np.float64)
    scale: Scale = problem.scale

    if problem.nd > 0:
        discrete = cast(DiscreteFunctionFloat, problem.functions.discrete)

    def discrete_jacobian_cd(arg: DiscreteJacobianArg) -> None:
        """Calculate discrete Jacobian using finite differences.

        Parameters
        ----------
        arg : DiscreteJacobianArg
        """
        discrete_arg._dv.z[:] = arg._dv.z
        for dv_key, df_index in djfds:
            var = discrete_arg._dv.var_dict[dv_key][:1]
            w = var[0]

            # central difference
            d = scale[dv_key] * DELTA1

            for i in df_index:
                arg.jacobian[i, dv_key] = 0.0

            for j in (-1, 1):
                var[0] = w + j * d
                call_callback(discrete, discrete_arg)
                for i in df_index:
                    arg.jacobian[i, dv_key] += j * discrete_arg.discrete[i] / (2 * d)

            var[0] = w

        # end of discrete_jacobian_cd callback function

    return discrete_jacobian_cd


def make_objective_hessian(problem: yapss.Problem, ogs: OGS) -> ObjectiveHessianFunction:
    """Generate objective hessian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object
    ogs : OGS
        Objective hessian structure, which is the same as the objective
        hessian finite difference structure.

    Returns
    -------
    Callable[ObjectiveArg]
        The objective hessian callback function
    """
    scale: Scale = problem.scale
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=float)
    objective_arg: ObjectiveArg[np.float64] = ObjectiveArg(problem, dv, dtype=np.float64)

    objective_function = cast(ObjectiveFunctionFloat, problem.functions.objective)

    def objective_hessian(arg: ObjectiveHessianArg) -> None:
        """Evaluate the objective hessian using central differences."""
        dv.z[:] = arg._dv.z
        f0: float | None = None  # the objective at the unperturbed point, on demand

        for i, dv_key1 in enumerate(ogs):
            var1 = dv.var_dict[dv_key1][:1]
            w1 = var1[0]
            d1 = scale[dv_key1] * DELTA2

            for dv_key2 in ogs[i:]:
                if dv_key2 == dv_key1:
                    # Diagonal pair: the four-point stencil's (+,-) and (-,+) legs both
                    # land on the unperturbed point, so it is f(w+2d) - 2 f(w) + f(w-2d)
                    # over 4 d^2, with f(w) evaluated once for every diagonal pair.
                    if f0 is None:
                        call_callback(objective_function, objective_arg)
                        f0 = float(objective_arg.objective)
                    var1[0] = w1 + 2 * d1
                    call_callback(objective_function, objective_arg)
                    fp = float(objective_arg.objective)
                    var1[0] = w1 - 2 * d1
                    call_callback(objective_function, objective_arg)
                    fm = float(objective_arg.objective)
                    var1[0] = w1
                    arg.hessian[dv_key1, dv_key2] = float((fp - 2 * f0 + fm) / (4 * d1 * d1))
                    continue

                var2 = dv.var_dict[dv_key2][:1]
                w2 = var2[0]
                d2 = scale[dv_key2] * DELTA2

                h = np.float64(0.0)

                # central difference
                for i1 in (+1, -1):
                    for i2 in (+1, -1):
                        var1[0] += i1 * d1
                        var2[0] += i2 * d2
                        call_callback(objective_function, objective_arg)
                        h += i1 * i2 * objective_arg.objective
                        var1[0] = w1
                        var2[0] = w2

                arg.hessian[dv_key1, dv_key2] = float(h / (4 * d1 * d2))

        # end of objective_hessian callback function

    return objective_hessian


def make_continuous_hessian(
    problem: yapss.Problem,
    chfds: CHFDS,
    tau_u: Sequence[NDArray[np.float64]],
) -> ContinuousHessianFunction:
    """Generate continuous Hessian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    chfds : CHFDS
        Finite difference structure for the continuous Hessian.
    tau_u : Sequence[NDArray[np.float64]]
        Non-dimensional collocation time points.

    Returns
    -------
    Callable[[ContinuousArg], None]
        The continuous Hessian callback function.
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=np.float64)
    store2: ContinuousStore[np.float64] = ContinuousStore(
        problem,
        dv,
        dtype=np.float64,
        tau_u=tau_u,
    )
    arg2 = store2.value_arg
    scale: Scale = problem.scale

    def continuous_hessian(arg: ContinuousHessianArg) -> None:
        """Calculate Hessian of the continuous constraint functions using finite differences."""
        continuous = cast(ContinuousFunctionFloat, problem.functions.continuous)
        store2._sync(arg._dv.z)
        phase_list = arg.phase_list

        for p in [PhaseIndex(p) for p in phase_list]:
            hessian = arg.phase[p].hessian
            ne = len(store2.phase[p].time)
            set_private(store2, "_phase_list", (p,))
            # the functions at the unperturbed point, snapshotted on the first diagonal
            # pair and shared by all of them (see below)
            base: dict[CFKey, Array] | None = None

            for key in chfds[p]:
                # extract from the key the functions whose Hessian will be evaluated,
                # and the variables w.r.t. which the derivative is taken
                ((v1, i1), (v2, i2)), fcn_list = key

                # extract the variables and store their original values
                var1 = store2[p, v1, i1]
                w1 = var1.copy()

                # prepare the perturbation size
                d1: np.float64 = scale[p, v1, i1] * DELTA2

                if (v1, i1) == (v2, i2):
                    # Diagonal pair: the four-point stencil's (+,-) and (-,+) legs both
                    # land on the unperturbed point, so it is f(w+2d) - 2 f(w) + f(w-2d)
                    # over 4 d^2, with f(w) evaluated once per phase.
                    if base is None:
                        call_callback(continuous, arg2)
                        base = {
                            fcn: store2[p, *fcn].copy()
                            for diag_key in chfds[p]
                            if diag_key[0][0] == diag_key[0][1]
                            for fcn in diag_key[1]
                        }
                    var1[:] = w1 + 2 * d1
                    call_callback(continuous, arg2)
                    plus = {fcn: store2[p, *fcn].copy() for fcn in fcn_list}
                    var1[:] = w1 - 2 * d1
                    call_callback(continuous, arg2)
                    den = 4 * d1 * d1
                    for fcn in fcn_list:
                        hessian[fcn, (v1, i1), (v2, i2)] = (
                            plus[fcn] - 2 * base[fcn] + store2[p, *fcn]
                        ) / den
                    var1[:] = w1
                    continue

                var2 = store2[p, v2, i2]
                w2 = var2.copy()
                d2: np.float64 = scale[p, v2, i2] * DELTA2

                # initialize the Hessian to zero
                for fcn in fcn_list:
                    hessian[fcn, (v1, i1), (v2, i2)] = np.zeros([ne], dtype=float)

                # central difference
                for i in (+1, -1):
                    for j in (+1, -1):
                        var1[:] += i * d1
                        var2[:] += j * d2
                        call_callback(continuous, arg2)
                        den = 4 * d1 * d2
                        for f, k in fcn_list:
                            delta_hessian = i * j * store2[p, f, k] / den
                            hessian[(f, k), (v1, i1), (v2, i2)] += delta_hessian
                        var1[:] = w1
                        var2[:] = w2

        # end of continuous_hessian callback function

    return continuous_hessian


def make_discrete_hessian(
    problem: yapss.Problem,
    dhfds: DHFDS,
) -> Callable[[DiscreteHessianArg], None]:
    """Generate discrete Hessian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    dhfds : DHFDS
        Finite difference structure for the discrete Hessian.

    Returns
    -------
    Callable[[DiscreteHessianArg], None]
        The discrete Hessian callback function.
    """
    nd: int = problem.nd

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=np.float64)
    discrete_arg: DiscreteArg[np.float64] = DiscreteArg(problem, dv, dtype=np.float64)
    scale: Scale = problem.scale

    if nd > 0:
        discrete = cast(DiscreteFunctionFloat, problem.functions.discrete)

    def discrete_hessian(arg: DiscreteHessianArg) -> None:
        """Calculate the discrete Hessian using finite differences."""
        # dv_key1 and dv_key2 are the keys of the decision variables w.r.t. which the
        # derivatives are taken
        dv_key1: DVKey
        dv_key2: DVKey

        discrete_arg._dv.z[:] = arg._dv.z
        g0: Array | None = None  # the constraints at the unperturbed point, on demand

        for dv_key1, inner_list in dhfds:
            var1 = discrete_arg._dv.var_dict[dv_key1][:1]
            w1 = var1[0]
            d1: np.float64 = scale[dv_key1] * DELTA2

            for dv_key2, discrete_index_list in inner_list:
                h: Array
                if dv_key2 == dv_key1:
                    # Diagonal pair: see objective_hessian
                    if g0 is None:
                        call_callback(discrete, discrete_arg)
                        g0 = discrete_arg._discrete.copy()
                    var1[0] = w1 + 2 * d1
                    call_callback(discrete, discrete_arg)
                    gp = discrete_arg._discrete.copy()
                    var1[0] = w1 - 2 * d1
                    call_callback(discrete, discrete_arg)
                    h = (gp - 2 * g0 + discrete_arg._discrete) / (4 * d1 * d1)
                    var1[0] = w1
                    for d in discrete_index_list:
                        arg.hessian[d, dv_key1, dv_key2] = h[d]
                    continue

                var2 = discrete_arg._dv.var_dict[dv_key2][:1]
                w2 = var2[0]
                d2: np.float64 = scale[dv_key2] * DELTA2
                h = np.zeros([nd], dtype=float)

                # central difference
                for i1, i2 in product((+1, -1), (+1, -1)):
                    var1[0] += i1 * d1
                    var2[0] += i2 * d2
                    call_callback(discrete, discrete_arg)
                    h += i1 * i2 * discrete_arg._discrete / (4 * d1 * d2)
                    var1[0] = w1
                    var2[0] = w2

                for d in discrete_index_list:
                    arg.hessian[d, dv_key1, dv_key2] = h[d]

        # end of discrete_hessian callback function

    return discrete_hessian
