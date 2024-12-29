"""

Make optimal control problem derivative callback functions using finite difference methods.

This module makes the objective gradient, objective hessian, continuous Jacobian,
continuous Hessian, discrete Jacobian, and discrete Hessian callback function, using
either central differences.

"""

# future imports
from __future__ import annotations

__all__ = ["make_cd_functions"]
# standard imports
from itertools import product
from typing import TYPE_CHECKING, Callable, cast

# third party imports
import numpy as np

# package imports
from .finite_difference import make_fd_structure
from .input_args import (
    ContinuousArg,
    ContinuousFunctionFloat,
    ContinuousHessianFunction,
    ContinuousJacobianFunction,
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
)
from .structure import DVStructure, get_nlp_dv_structure
from .types_ import PhaseIndex

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .problem import Scale
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


def make_cd_functions(problem: yapss.Problem, z0: Array) -> ProblemFunctions:
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
    cd_functions = make_fd_structure(problem, z0)
    order = problem.derivatives.order

    # first derivatives
    ogs: OGS = cd_functions.objective_gradient_structure
    cjfds: CJFDS = cd_functions.continuous_jacobian_structure_cd
    cd_functions.objective_gradient = make_objective_gradient(problem, ogs)
    cd_functions.continuous_jacobian = make_continuous_jacobian(problem, cjfds)
    djfds: DJFDS = cd_functions.discrete_jacobian_structure_cd
    cd_functions.discrete_jacobian = make_discrete_jacobian(problem, djfds)

    if order == "second":
        # discrete hessian
        dhfds: DHFDS | None = cd_functions.discrete_hessian_structure_cd

        # mypy hinting
        assert dhfds is not None  # noqa: S101
        cd_functions.discrete_hessian = make_discrete_hessian(problem, dhfds)

        # objective hessian
        cd_functions.objective_hessian = make_objective_hessian(problem, ogs)

        # continuous hessian
        chfds: CHFDS | None = cd_functions.continuous_hessian_structure_cd
        assert chfds is not None  # noqa: S101
        cd_functions.continuous_hessian = make_continuous_hessian(problem, chfds)

    return cd_functions


def make_objective_gradient(
    problem: yapss._private.problem.Problem,
    ogs: OGS,
) -> ObjectiveGradientFunction:
    """Generate objective gradient callback function using finite differences.

    Parameters
    ----------
    problem : yapss._private.problem.Problem
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
        arg.gradient.clear()

        for dv_key in ogs:
            var = dv.var_dict[dv_key][:1]
            w = var[0]

            # central difference
            delta_objective = np.float64(0.0)
            d: np.float64 = scale[dv_key] * DELTA1
            for i in (-1, 1):
                var[0] = w + i * d
                objective_function(arg1)
                delta_objective += i * arg1.objective
                arg.gradient[dv_key] = float(delta_objective / (2 * d))

            var[0] = w

        # end of objective_gradient callback function

    return objective_gradient


def make_continuous_jacobian(problem: yapss.Problem, cjfds: CJFDS) -> ContinuousJacobianFunction:
    """Generate continuous Jacobian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    cjfds : CJFDS
        Finite difference structure for the continuous Jacobian.

    Returns
    -------
    Callable[[ContinuousJacobianArg], None]
        The continuous Jacobian callback function.
    """
    scale: Scale = problem.scale
    if problem.np > 0:
        continuous = cast(ContinuousFunctionFloat, problem.functions.continuous)

    def continuous_jacobian(arg: ContinuousArg[np.float64]) -> None:
        """Calculate continuous Jacobian using finite differences.

        Parameters
        ----------
        arg : ContinuousJacobianArg
        """
        phase_list = arg.phase_list

        for p in [PhaseIndex(p) for p in phase_list]:
            phase = arg.phase[p]
            jacobian = arg.phase[p].jacobian

            arg._phase_list = (p,)
            ne = len(phase.time)

            for cv_key, cf_keys in cjfds[p]:
                var2, i1 = cv_key
                var = arg[p, var2, i1]
                w = var.copy()
                for cf_key in cf_keys:
                    jacobian[cf_key, cv_key] = np.zeros(ne)

                # central difference
                d = scale[p, var2, i1] * DELTA1

                for j in (-1, 1):
                    var[:] = w + j * d
                    continuous(arg)

                    for cf_key in cf_keys:
                        var1, i = cf_key
                        jacobian[cf_key, cv_key] += j * arg[p, var1, i] / (2 * d)

                var[:] = w

        arg._phase_list = phase_list

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
                discrete(discrete_arg)
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
        arg.hessian.clear()

        for i, dv_key1 in enumerate(ogs):
            var1 = dv.var_dict[dv_key1][:1]
            w1 = var1[0]
            d1 = scale[dv_key1] * DELTA2

            for dv_key2 in ogs[i:]:
                var2 = dv.var_dict[dv_key2][:1]
                w2 = var2[0]
                d2 = scale[dv_key2] * DELTA2

                h = np.float64(0.0)

                # central difference
                for i1 in (+1, -1):
                    for i2 in (+1, -1):
                        var1[0] += i1 * d1
                        var2[0] += i2 * d2
                        objective_function(objective_arg)
                        h += i1 * i2 * objective_arg.objective
                        var1[0] = w1
                        var2[0] = w2

                arg.hessian[dv_key1, dv_key2] = float(h / (4 * d1 * d2))

        # end of objective_hessian callback function

    return objective_hessian


def make_continuous_hessian(problem: yapss.Problem, chfds: CHFDS) -> ContinuousHessianFunction:
    """Generate continuous Hessian callback function using finite differences.

    Parameters
    ----------
    problem : Problem
        The user-defined problem object.
    chfds : CHFDS
        Finite difference structure for the continuous Hessian.

    Returns
    -------
    Callable[[ContinuousArg], None]
        The continuous Hessian callback function.
    """
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, dtype=np.float64)
    arg2: ContinuousArg[np.float64] = ContinuousArg(problem, dv, dtype=np.float64)
    scale: Scale = problem.scale

    def continuous_hessian(arg: ContinuousArg[np.float64]) -> None:
        """Calculate Hessian of the continuous constraint functions using finite differences."""
        continuous = cast(ContinuousFunctionFloat, problem.functions.continuous)
        arg2._dv.z[:] = arg._dv.z
        phase_list = arg.phase_list

        for p in [PhaseIndex(p) for p in phase_list]:
            hessian = arg.phase[p].hessian
            arg2.phase[p].time[:] = arg.phase[p].time
            ne = len(arg2.phase[p].time)
            arg2._phase_list = (p,)

            for key in chfds[p]:
                # extract from the key the functions whose Hessian will be evaluated,
                # and the variables w.r.t. which the derivative is taken
                ((v1, i1), (v2, i2)), fcn_list = key

                # extract the variables and store their original values
                var1 = arg2[p, v1, i1]
                w1 = var1.copy()
                var2 = arg2[p, v2, i2]
                w2 = var2.copy()

                # prepare the perturbation size
                d1: np.float64 = scale[p, v1, i1] * DELTA2
                d2: np.float64 = scale[p, v2, i2] * DELTA2

                # initialize the Hessian to zero
                for fcn in fcn_list:
                    hessian[fcn, (v1, i1), (v2, i2)] = np.zeros([ne], dtype=float)

                # central difference
                for i in (+1, -1):
                    for j in (+1, -1):
                        var1[:] += i * d1
                        var2[:] += j * d2
                        continuous(arg2)
                        den = 4 * d1 * d2
                        for f, k in fcn_list:
                            delta_hessian = i * j * arg2[p, f, k] / den
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

        for dv_key1, inner_list in dhfds:
            var1 = discrete_arg._dv.var_dict[dv_key1][:1]
            w1 = var1[0]
            d1: np.float64 = scale[dv_key1] * DELTA2

            for dv_key2, discrete_index_list in inner_list:
                var2 = discrete_arg._dv.var_dict[dv_key2][:1]
                w2 = var2[0]
                d2: np.float64 = scale[dv_key2] * DELTA2
                h: Array = np.zeros([nd], dtype=float)

                # central difference
                for i1, i2 in product((+1, -1), (+1, -1)):
                    var1[0] += i1 * d1
                    var2[0] += i2 * d2
                    discrete(discrete_arg)
                    h += i1 * i2 * discrete_arg._discrete / (4 * d1 * d2)
                    var1[0] = w1
                    var2[0] = w2

                for d in discrete_index_list:
                    arg.hessian[d, dv_key1, dv_key2] = h[d]

        # end of discrete_hessian callback function

    return discrete_hessian
