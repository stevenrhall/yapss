"""

Callback functions for the "auto" derivative method.

This module automatically generates the derivative callback functions and derivative
structures needed to solve the optimal control problem, using the casadi module automatic
differentiation functions.

"""

from __future__ import annotations

# local package imports
# standard library imports
from typing import TYPE_CHECKING, cast

# third-party imports
import numpy as np
from casadi import SX, Function
from casadi import hessian as cd_hessian
from casadi import jacobian as cd_jacobian
from casadi import tril, vertcat

from yapss.math.wrapper import SXW

from .input_args import (
    ContinuousArg,
    ContinuousFunction,
    ContinuousFunctionObject,
    DiscreteArg,
    DiscreteFunction,
    DiscreteFunctionObject,
    DiscreteHessianArg,
    DiscreteHessianFunction,
    DiscreteJacobianArg,
    DiscreteJacobianFunction,
    ObjectiveArg,
    ObjectiveFunction,
    ObjectiveFunctionObject,
    ObjectiveGradientArg,
    ObjectiveGradientFunction,
    ObjectiveHessianArg,
    ObjectiveHessianFunction,
    ProblemFunctions,
)
from .structure import DVPhase, DVStructure

# type-checking imports
if TYPE_CHECKING:
    # third-party imports
    from numpy.typing import NDArray

    # local package imports
    import yapss

    from .types_ import CHS, CJS, DHS, DJS, OGS, OHS, CHSPhase, CJSPhase


def make_auto_functions(problem: yapss.Problem) -> ProblemFunctions:
    """Generate derivative callback functions using auto-differentiation.

    Use casadi auto-differentiation to make the callback functions and structures
    required by the NLP solver.

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    problem_functions : ProblemFunctions
    """
    # make the arguments for the objective, discrete, and continuous functions
    objective_arg, discrete_arg, continuous_arg = make_args(problem)

    # make the objective and discrete callback functions and derivative structures
    # noinspection PyTupleAssignmentBalance
    # because PyCharm sometimes can't count
    (
        objective,
        objective_gradient,
        objective_gradient_structure,
        objective_hessian,
        objective_hessian_structure,
        discrete,
        discrete_jacobian,
        discrete_jacobian_structure,
        discrete_hessian,
        discrete_hessian_structure,
    ) = make_discrete_derivatives(problem, objective_arg, discrete_arg)

    # make the continuous callback functions and derivative structures
    (
        continuous,
        continuous_jacobian,
        continuous_jacobian_structure,
        continuous_hessian,
        continuous_hessian_structure,
    ) = make_continuous_derivatives(problem, continuous_arg)

    return ProblemFunctions(
        continuous=continuous,
        continuous_jacobian=continuous_jacobian,
        continuous_jacobian_structure=continuous_jacobian_structure,
        continuous_hessian=continuous_hessian,
        continuous_hessian_structure=continuous_hessian_structure,
        objective=objective,
        objective_gradient=objective_gradient,
        objective_gradient_structure=objective_gradient_structure,
        objective_hessian=objective_hessian,
        objective_hessian_structure=objective_hessian_structure,
        discrete=discrete,
        discrete_jacobian=discrete_jacobian,
        discrete_jacobian_structure=discrete_jacobian_structure,
        discrete_hessian=discrete_hessian,
        discrete_hessian_structure=discrete_hessian_structure,
    )


def make_args(
    problem: yapss.Problem,
) -> tuple[ObjectiveArg[np.object_], DiscreteArg[np.object_], ContinuousArg[np.object_]]:
    """Make callback function arguments.

    Make callback function arguments of types `ObjectiveArg`, `DiscreteArg`, and
    `ContinuousArg` with `SX` symbolic variables.

    Parameters
    ----------
    problem : Problem

    Returns
    -------
    objective_arg : ObjectiveArg
    discrete_arg : DiscreteArg
    continuous_arg : ContinuousArg
    """
    dv: DVStructure[np.object_] = DVStructure()
    dv.phase = []
    dv.discrete = np.zeros([problem.nd], dtype=object)
    dv.auxdata = problem.auxdata
    dv.parameter = np.array([SXW(SX.sym(f"s_{i}")) for i in range(problem.ns)])
    dv.z = np.zeros([], dtype=object)

    for p in range(problem.np):
        phase: DVPhase[np.object_] = DVPhase()
        dv.phase.append(phase)

        phase.x0 = np.array(
            [SXW(SX.sym(f"x0_{p}_{i}")) for i in range(problem.nx[p])],
            dtype=object,
        )
        phase.xf = np.array(
            [SXW(SX.sym(f"xf_{p}_{i}")) for i in range(problem.nx[p])],
            dtype=object,
        )
        phase.q = np.array(
            [SXW(SX.sym(f"q_{p}_{i}")) for i in range(problem.nq[p])],
            dtype=object,
        )
        phase.t0 = np.array([SXW(SX.sym(f"t0_{p}"))], dtype=object)
        phase.tf = np.array([SXW(SX.sym(f"tf_{p}"))], dtype=object)
        phase.xc = [np.array([SXW(SX.sym(f"x_{i}"))]) for i in range(problem.nx[p])]
        phase.u = [np.array([SXW(SX.sym(f"u_{i}"))]) for i in range(problem.nu[p])]

    dv.s = np.array(dv.parameter, dtype=object)

    objective_arg = ObjectiveArg(problem, dv, dtype=np.object_)
    discrete_arg = DiscreteArg(problem, dv, dtype=np.object_)
    continuous_arg = ContinuousArg(problem, dv=dv, dtype=np.object_)

    for p in range(problem.np):
        continuous_arg.phase[p].time = np.array([SXW(SX.sym("t"))])
    return objective_arg, discrete_arg, continuous_arg


def make_discrete_derivatives(
    problem: yapss.Problem,
    objective_arg: ObjectiveArg[np.object_],
    discrete_arg: DiscreteArg[np.object_],
) -> tuple[
    ObjectiveFunction,
    ObjectiveGradientFunction,
    OGS,
    ObjectiveHessianFunction | None,
    OHS | None,
    DiscreteFunction | None,
    DiscreteJacobianFunction | None,
    DJS | None,
    DiscreteHessianFunction | None,
    DHS | None,
]:
    """Make the objective and discrete callback functions and derivative structures.

    Parameters
    ----------
    problem : Problem
    objective_arg : ObjectiveArg
    discrete_arg : DiscreteArg

    Returns
    -------
    objective : ObjectiveFunction
        The objective callback function
    objective_gradient : ObjectiveFunction
        The objective gradient callback function
    objective_gradient structure : OGS
    objective_hessian : Optional[ObjectiveFunction]
        The objective Hessian callback function
    objective_hessian_structure : Optional[OHS]
    discrete : Optional[DiscreteFunction]
        The discrete callback function
    discrete_jacobian : Optional[DiscreteFunction]
        The discrete Jacobian callback function
    discrete_jacobian_structure : Optional[DJS]
    discrete_hessian : Optional[DiscreteFunction]
        The discrete Hessian callback function
    discrete_hessian_structure : Optional[DHS]
    """
    objective_gradient_structure: OGS
    objective_hessian: ObjectiveHessianFunction | None = None
    objective_hessian_structure: OHS | None = None
    discrete: DiscreteFunction | None = None
    discrete_jacobian: DiscreteJacobianFunction | None = None
    discrete_jacobian_structure: DJS | None = None
    discrete_hessian: DiscreteHessianFunction | None = None
    discrete_hessian_structure: DHS | None = None

    sxqt = []
    sxqt += list(objective_arg.parameter)
    dv_keys = [(0, "s", i) for i in range(problem.ns)]

    for p in range(problem.np):
        phase = objective_arg.phase[p]

        # list of symbolic variables
        sxqt += list(phase.initial_state)
        sxqt += list(phase.final_state)
        sxqt += list(phase.integral)
        sxqt += [phase.initial_time]
        sxqt += [phase.final_time]

        # discrete variable keys
        dv_keys += [(p, "x0", i) for i in range(problem.nx[p])]
        dv_keys += [(p, "xf", i) for i in range(problem.nx[p])]
        dv_keys += [(p, "q", i) for i in range(problem.nq[p])]
        dv_keys += [(p, "t0", 0)]
        dv_keys += [(p, "tf", 0)]

    # objective function
    objective_function_ = cast(ObjectiveFunctionObject, problem.functions.objective)
    objective_function_(objective_arg)
    objective_out = SXW(objective_arg.objective)
    objective_function = Function(
        "objective",
        [vertcat(*[item._value for item in sxqt])],
        [objective_out._value],
    )
    problem.auxdata.objective_out = objective_out
    problem.auxdata.objective_function = objective_function

    # objective gradient
    gradient = cd_jacobian(objective_out._value, vertcat(*[item._value for item in sxqt]))
    col = gradient.sparsity().get_col()
    objective_gradient_structure = tuple(dv_keys[i] for i in col)
    gradient = vertcat(*[gradient[0, i] for i in col])
    gradient_function = Function("gradient", [vertcat(*[item._value for item in sxqt])], [gradient])

    if problem.derivatives.order == "second":
        # objective hessian
        casadi_objective_hessian = cd_hessian(
            objective_out._value,
            vertcat(*[item._value for item in sxqt]),
        )[0]
        casadi_objective_hessian = tril(casadi_objective_hessian, True)  # noqa: FBT003
        rc = tuple(zip(*casadi_objective_hessian.sparsity().get_triplet()))
        objective_hessian_structure = tuple((dv_keys[i], dv_keys[j]) for i, j in rc)
        casadi_objective_hessian = vertcat(*[casadi_objective_hessian[i, j] for i, j in rc])
        objective_hessian_function = Function(
            "objective_hessian",
            [vertcat(*[item._value for item in sxqt])],
            [casadi_objective_hessian],
        )

    if problem.nd > 0:
        discrete_function_ = cast(DiscreteFunctionObject, problem.functions.discrete)
        discrete_function_(discrete_arg)
        discrete_out = discrete_arg._discrete
    else:
        discrete_out = np.array([], dtype=object)

    for i, item in enumerate(discrete_out):
        discrete_out[i] = SXW(item)

    discrete_function = Function(
        "discrete",
        [vertcat(*[item._value for item in sxqt])],
        [vertcat(*[item._value for item in discrete_out])],
    )

    # discrete jacobian
    jac = cd_jacobian(
        vertcat(*[item._value for item in discrete_out]),
        vertcat(*[item._value for item in sxqt]),
    )
    rc = tuple(zip(*jac.sparsity().get_triplet()))
    discrete_jacobian_structure = tuple((i, dv_keys[j]) for i, j in rc)
    jac = vertcat(*[jac[i, j] for i, j in rc])
    jacobian_function = Function("jacobian", [vertcat(*[item._value for item in sxqt])], [jac])

    if problem.derivatives.order == "second":
        hessian = []
        dhs = []

        for i, f in enumerate([item._value for item in discrete_out]):
            hessian_i = cd_hessian(f, vertcat(*[item._value for item in sxqt]))[0]
            hessian_i = tril(hessian_i, True)  # noqa: FBT003
            rc = tuple(zip(*hessian_i.sparsity().get_triplet()))
            hessian_i = [hessian_i[j, k] for j, k in rc]
            hessian += hessian_i
            dhs += [(i, dv_keys[k], dv_keys[j]) for j, k in rc]

        discrete_hessian_structure = tuple(dhs)
        hessian = vertcat(*hessian)
        discrete_hessian_functions = Function(
            "discrete_hessian",
            [vertcat(*[item._value for item in sxqt])],
            [hessian],
        )

    discrete_jacobian_structure = tuple(discrete_jacobian_structure)

    # define callback functions

    def objective(arg: ObjectiveArg[np.float64]) -> None:
        """Objective callback function.

        Parameters
        ----------
        arg: ObjectiveArg

        """
        sxqt_ = make_sxqt(problem, arg)
        objective_ = objective_function(sxqt_).full()
        arg.objective = objective_[0, 0]

    def objective_gradient(arg: ObjectiveGradientArg) -> None:
        """Objective gradient callback function.

        Parameters
        ----------
        arg: ObjectiveArg
        """
        sxqt_ = make_sxqt(problem, arg)
        gradient_ = gradient_function(sxqt_).full()[:, 0]
        for i, key in enumerate(objective_gradient_structure):
            arg.gradient[key] = gradient_[i]

    if problem.nd > 0:

        def discrete(arg: DiscreteArg[np.float64]) -> None:
            """Discrete callback function.

            Parameters
            ----------
            arg: ObjectiveArg
            """
            sxqt_ = make_sxqt(problem, arg)
            discrete_ = discrete_function(sxqt_).full()
            arg.discrete = discrete_[:, 0]

        def discrete_jacobian(arg_dj: DiscreteJacobianArg) -> None:
            """Discrete Jacobian callback function.

            Parameters
            ----------
            arg_dj: DiscreteJacobianArg
            """
            sxqt_ = make_sxqt(problem, arg_dj)
            jacobian_ = jacobian_function(sxqt_).full()
            for i, key in enumerate(discrete_jacobian_structure):
                arg_dj.jacobian[key] = jacobian_[i, 0]

    if problem.derivatives.order == "second":

        def objective_hessian(arg: ObjectiveHessianArg) -> None:
            """Objective Hessian callback function.

            Parameters
            ----------
            arg: ObjectiveArg
            """
            sxqt_ = make_sxqt(problem, arg)
            hessian_ = objective_hessian_function(sxqt_).full().reshape(-1)
            assert objective_hessian_structure is not None  # noqa: S101
            for i, key in enumerate(objective_hessian_structure):
                arg.hessian[key] = hessian_[i]

        if problem.nd > 0:

            def discrete_hessian(arg: DiscreteHessianArg) -> None:
                """Discrete Hessian callback function.

                Parameters
                ----------
                arg: DiscreteArg
                """
                sxqt_ = make_sxqt(problem, arg)
                hessian_ = discrete_hessian_functions(sxqt_).full().reshape(-1)
                assert discrete_hessian_structure is not None  # noqa: S101
                for i, key in enumerate(discrete_hessian_structure):
                    arg.hessian[key] = hessian_[i]

    return (
        objective,
        objective_gradient,
        objective_gradient_structure,
        objective_hessian,
        objective_hessian_structure,
        discrete,
        discrete_jacobian,
        discrete_jacobian_structure,
        discrete_hessian,
        discrete_hessian_structure,
    )


def make_continuous_derivatives(
    problem: yapss.Problem,
    arg: ContinuousArg[np.object_],
) -> tuple[
    ContinuousFunction | None,
    ContinuousFunction | None,
    CJS | None,
    ContinuousFunction | None,
    CHS | None,
]:
    """Make the continuous callback functions and derivative structures.

    Parameters
    ----------
    problem : Problem
    arg : ContinuousArg

    Returns
    -------
    continuous : Optional[ContinuousFunction]
        The continuous callback function
    continuous_jacobian : Optional[ContinuousFunction]
        The continuous jacobian callback function
    continuous_jacobian_structure : Optional[CJS]
    continuous_hessian : Optional[ContinuousFunction]
        The continuous hessian callback function
    continuous_hessian : Optional[CHS]
    """
    # In the trivial case where there are no phases, all the returns values are None
    if problem.np == 0:
        return None, None, None, None, None

    continuous_hessian: ContinuousFunction | None = None
    cjs: list[CJSPhase] = []
    chs: list[CHSPhase] = []

    continuous_functions: list[Function] = []
    jacobian_functions: list[Function] = []
    hessian_functions: list[Function] = []

    # call the continuous function callback with symbolic args to get symbolic expressions
    # for the functions
    continuous_function_ = cast(ContinuousFunctionObject, problem.functions.continuous)
    continuous_function_(arg)

    for p in range(problem.np):
        phase = arg.phase[p]
        sxut = (
            list(arg.parameter)
            + [item[0] for item in phase.state]
            + [item[0] for item in phase.control]
            + list(phase.time)
        )
        sxut = [item._value for item in sxut]

        # corresponding keys
        cv_keys = [("s", i) for i in range(problem.ns)]
        cv_keys += [("x", i) for i in range(problem.nx[p])]
        cv_keys += [("u", i) for i in range(problem.nu[p])]
        cv_keys += [("t", 0)]

        # keys for the continuous functions
        cf_keys = []
        cf_keys += [("f", i) for i in range(problem.nx[p])]
        cf_keys += [("g", i) for i in range(problem.nq[p])]
        cf_keys += [("h", i) for i in range(problem.nh[p])]

        # convert to lists
        f = list(arg.phase[p].dynamics[:, 0])
        g = list(arg.phase[p].integrand[:, 0])
        h = list(arg.phase[p].path[:, 0])

        f = [item._value if isinstance(item, SXW) else item for item in f]
        g = [item._value if isinstance(item, SXW) else item for item in g]
        h = [item._value if isinstance(item, SXW) else item for item in h]

        fgh = f + g + h

        # create continuous function that operates on numeric arguments
        sxut = vertcat(*sxut)
        continuous_functions.append(
            Function("continuous", [sxut], [vertcat(*f), vertcat(*g), vertcat(*h)]),
        )

        # create jacobian function that operates on numeric arguments
        casadi_jacobian = cd_jacobian(vertcat(*fgh), sxut)

        rc = tuple(zip(*casadi_jacobian.sparsity().get_triplet()))
        cjs.append(tuple((cf_keys[i], cv_keys[j]) for i, j in rc))
        casadi_jacobian = vertcat(*[casadi_jacobian[i, j] for i, j in rc])
        jacobian_functions.append(Function("jacobian", [sxut], [casadi_jacobian]))

        # hessian
        if problem.derivatives.order == "second":
            casadi_hessian = []
            chs_i = []

            for i, f in enumerate(fgh):
                hes = cd_hessian(f, sxut)[0]
                hes = tril(hes, True)  # noqa: FBT003
                rc = tuple(zip(*hes.sparsity().get_triplet()))
                hes = [hes[j, k] for j, k in rc]
                casadi_hessian += hes
                chs_i += [(cf_keys[i], cv_keys[k], cv_keys[j]) for j, k in rc]

            chs.append(tuple(chs_i))
            casadi_hessian = vertcat(*casadi_hessian)
            hessian_functions.append(Function("hessian", [sxut], [casadi_hessian]))

    continuous_jacobian_structure = tuple(cjs)

    def continuous(continuous_arg: ContinuousArg[np.float64]) -> None:
        """Continuous callback function."""
        sxut_ = make_sxut(problem, continuous_arg)

        for q in continuous_arg.phase_list:
            continuous_ = continuous_functions[q](sxut_[q])
            if continuous_[0].shape[0]:
                continuous_arg.phase[q].dynamics[:] = continuous_[0]
            if continuous_[1].shape[0]:
                continuous_arg.phase[q].integrand[:] = continuous_[1]
            if continuous_[2].shape[0]:
                continuous_arg.phase[q].path[:] = continuous_[2]

    def continuous_jacobian(continuous_arg: ContinuousArg[np.float64]) -> None:
        """Continuous Jacobian callback function."""
        sxut_ = make_sxut(problem, continuous_arg)

        for q in continuous_arg.phase_list:
            jac = jacobian_functions[q](sxut_[q])
            for i, key in enumerate(cjs[q]):
                continuous_arg.phase[q].jacobian[key] = jac[i, :].full()[0]

    if problem.derivatives.order == "second":

        def continuous_hessian(continuous_arg: ContinuousArg[np.float64]) -> None:
            """Continuous Hessian callback function."""
            sxut_ = make_sxut(problem, continuous_arg)

            for q in continuous_arg.phase_list:
                hessian = hessian_functions[q](sxut_[q])
                for i, key in enumerate(chs[q]):
                    continuous_arg.phase[q].hessian[key] = hessian[i, :].full()[0]

    return (
        continuous,
        continuous_jacobian,
        continuous_jacobian_structure,
        continuous_hessian,
        tuple(chs) if chs is not None else None,
    )


def make_sxqt(
    problem: yapss.Problem,
    arg: (
        ObjectiveArg[np.float64]
        | ObjectiveGradientArg
        | ObjectiveHessianArg
        | DiscreteArg[np.float64]
        | DiscreteJacobianArg
        | DiscreteHessianArg
    ),
) -> NDArray[np.float64]:
    """Make numpy array argument for casadi discrete functions.

    The function eliminates the need to replicate this (modest) code fragment 6 times.

    Parameters
    ----------
    problem : Problem
    arg : Union[ObjectiveArg, DiscreteArg]

    Returns
    -------
    sxqt : NDArray
    """
    sxqt: NDArray[np.float64] = np.zeros(
        [problem.ns + 2 * sum(problem.nx) + sum(problem.nq) + 2 * problem.np],
        dtype=float,
    )
    j = 0
    sxqt[: problem.ns] = arg.parameter
    j += problem.ns

    for p in range(problem.np):
        nx = problem.nx[p]
        sxqt[j : j + nx] = arg.phase[p].initial_state
        j += nx
        sxqt[j : j + nx] = arg.phase[p].final_state
        j += nx
        nq = problem.nq[p]
        sxqt[j : j + nq] = arg.phase[p].integral
        j += nq
        sxqt[j] = arg.phase[p].initial_time
        j += 1
        sxqt[j] = arg.phase[p].final_time
        j += 1

    return sxqt


def make_sxut(
    problem: yapss.Problem,
    arg: ContinuousArg[np.float64],
) -> dict[int, NDArray[np.float64]]:
    """Make numpy array argument for casadi continuous functions.

    The function eliminates the need to replicate this (modest) code fragment 3 times.

    Parameters
    ----------
    problem : Problem
    arg : ContinuousArg

    Returns
    -------
    sxut : Dict[int, NDArray]
    """
    sxut: dict[int, NDArray[np.float64]] = {}

    for p in arg.phase_list:
        nt = len(arg.phase[p].time)
        sxut_p = np.zeros([problem.ns + problem.nx[p] + problem.nu[p] + 1, nt])
        j = 0
        for i in range(problem.ns):
            sxut_p[j, :] = arg.parameter[i]
            j += 1
        for i in range(problem.nx[p]):
            sxut_p[j, :] = arg.phase[p].state[i]
            j += 1
        for i in range(problem.nu[p]):
            sxut_p[j, :] = arg.phase[p].control[i]
            j += 1
        sxut_p[j, :] = arg.phase[p].time
        sxut[p] = sxut_p

    return sxut
