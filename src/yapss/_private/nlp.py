"""

Ipopt problem formulation.

This module defines a class `NLP` which the defines the nonlinear program to be solved by
IPOPT. In particular, the class NLP defines the methods

 :meth:`NLP.objective`
 :meth:`NLP.gradient`
 :meth:`NLP.constraints`
 :meth:`NLP.jacobian`
 :meth:`NLP.hessian`
 :meth:`NLP.jacobianstructure`
 :meth:`NLP.hessianstructure`

required by Ipopt. The Jacobian and Hessian callbacks and their structures are built
by the `jacobian` and `hessian` modules from single assembly plans; this module holds
the class, the value-level callbacks (objective, gradient, constraints), and the
evaluators for the user's continuous and discrete functions that the plan modules
consume.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, cast

# third party imports
import numpy as np

from .hessian import make_nlp_hessian

# package imports
from .input_args import (
    ContinuousArg,
    ContinuousFunctionFloat,
    ContinuousHessianArg,
    ContinuousJacobianArg,
    DiscreteArg,
    DiscreteFunctionFloat,
    DiscreteJacobianArg,
    ObjectiveArg,
    ObjectiveFunctionFloat,
    ObjectiveGradientArg,
    ProblemFunctions,
)
from .jacobian import make_nlp_jacobian
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable, Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .mesh import Mesh
    from .types_ import DVKey

    FloatArray = NDArray[np.float64]
    Intermediate = Callable[
        [int, int, float, float, float, float, float, float, float, float, int, int],
        bool,
    ]


class NLP:
    """Construct a nonlinear program (NLP) from the optimal control problem.

    Class to construct a nonlinear program (NLP) from the optimal control problem
    defined by the problem object, and the derivative functions derived from it.

    Parameters
    ----------
    problem : yapss.problem.Problem
        User-generated problem object that represent the optimal control problem.
    functions
        The objective, continuous, and discrete functions, their first and
        (sometimes) seconds derivatives, and data that provides the structure of the
        derivatives.
    """

    def __init__(
        self,
        problem: yapss.Problem,
        functions: ProblemFunctions,
        mesh: Mesh,
    ) -> None:
        # store arguments
        self.problem: yapss.Problem = problem
        self.functions: ProblemFunctions = functions
        self.mesh = mesh

        nlp = self
        self._objective = make_nlp_objective(nlp)
        self._constraints = make_nlp_constraints(nlp)
        self._gradient = make_nlp_objective_gradient(nlp)

        # constraint jacobian: structure and evaluator come from one assembly plan,
        # so their entries correspond by construction; see the jacobian module
        (self.irow, self.jcol), self._jacobian = make_nlp_jacobian(
            nlp,
            make_eval_continuous(nlp),
            make_eval_discrete_jacobian(nlp),
        )

        self.nlp_hessian_structure: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

        self.intermediate: Intermediate | None = None
        self._hessian: Callable[[FloatArray, FloatArray, np.float64], FloatArray]
        if problem.derivatives.order == "second":
            # the structure and the evaluator come from one assembly plan, so their
            # entries correspond by construction; see the hessian module
            self.nlp_hessian_structure, self._hessian = make_nlp_hessian(
                nlp,
                make_eval_continuous(nlp),
            )
        self.eval_continuous = make_eval_continuous(self)

    def objective(self, z: FloatArray) -> float:
        """Evaluate NLP objective function.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array

        Returns
        -------
        float
            The objective function value at z.
        """
        return self._objective(z)

    def gradient(self, z: FloatArray) -> FloatArray:
        """Evaluate NLP gradient function.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array

        Returns
        -------
        NDArray
            The gradient.
        """
        return self._gradient(z)

    def constraints(self, z: FloatArray) -> FloatArray:
        """Evaluate NLP constraint function.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array

        Returns
        -------
        NDArray
            The NLP constraint function
        """
        return self._constraints(z)

    def jacobian(self, z: FloatArray) -> FloatArray:
        """Evaluate NLP Jacobian function.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array

        Returns
        -------
        NDArray
            The NLP Jacobian function
        """
        return self._jacobian(z)

    def hessian(
        self,
        z: FloatArray,
        lambda_: FloatArray,
        objective_factor: np.float64,
    ) -> FloatArray:
        """Evaluate the Hessian of the NLP Lagrangian.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array
        lambda_ : NDArray
            NLP constraint Lagrange multiplier
        objective_factor : float
            objective scaling factor

        Returns
        -------
        NDArray
            The Hessian of the NLP Lagrangian
        """
        return self._hessian(z, lambda_, objective_factor)

    def jacobianstructure(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return Jacobian structure of nonlinear program.

        Returns
        -------
            tuple[tuple[int, ...], tuple[int, ...]]
        """
        return self.irow, self.jcol

    def hessianstructure(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return the NLP Hessian structure.

        Returns
        -------
        tuple[tuple[int, ...], tuple[int, ...]]
        """
        return self.nlp_hessian_structure


def make_nlp_objective(nlp: NLP) -> Callable[[FloatArray], float]:
    """Construct the NLP objective function from the optimal control problem definition.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    Callable[[FloatArray], float]
        The objective callback function
    """
    problem = nlp.problem
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    arg = ObjectiveArg(problem, dv, dtype=np.float64)

    objective_function = cast(ObjectiveFunctionFloat, nlp.functions.objective)

    # begin callback function

    def eval_nlp_objective(z: FloatArray) -> float:
        dv.z[:] = z
        objective_function(arg)

        return float(arg.objective)

    # end callback function

    return eval_nlp_objective


def make_nlp_constraints(nlp: NLP) -> Callable[[FloatArray], FloatArray]:
    """Construct the NLP constraint function from the optimal control problem definition.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    Callable[[FloatArray], FloatArray]
        NLP constraint function
    """
    problem = nlp.problem
    mesh: Mesh = nlp.mesh
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    ci: ContinuousArg[np.float64] = ContinuousArg(
        problem,
        dv,
        dtype=np.float64,
        tau_u=mesh.tau_u,
    )
    di: DiscreteArg[np.float64] = DiscreteArg(problem, dv, np.float64)
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)

    if problem.np > 0:
        continuous_function = cast(ContinuousFunctionFloat, nlp.functions.continuous)
    if problem.nd > 0:
        discrete_function = cast(DiscreteFunctionFloat, nlp.functions.discrete)

    # begin callback function

    def eval_constraints(z: FloatArray) -> FloatArray:
        """Evaluate the NLP constraint function.

        This callback function is autogenerated by make_nlp_constraints.

        Parameters
        ----------
        z : FloatArray
            The NLP decision variable array

        Returns
        -------
        FloatArray
            NLP constraint functions values
        """
        ci._sync(z)

        # call the user-defined continuous function
        if problem.np > 0:
            continuous_function(ci)

        cf.c[:] = 0.0

        for p in range(problem.np):
            nlp_cf_phase = cf.phase[p]
            t0 = dv.phase[p].t0[0]
            tf = dv.phase[p].tf[0]
            dt2 = (tf - t0) / 2

            # state equation defect
            for i in range(problem.nx[p]):
                if problem.spectral_method == "lgl":
                    defect = ci.phase[p].dynamics[i][cf.phase[p].defect_index] * dt2
                    defect -= mesh.d[p] @ dv.phase[p].xa[i]
                elif problem.spectral_method in ("lgr", "lg"):
                    defect = ci.phase[p].dynamics[i] * dt2 - mesh.d[p] @ dv.phase[p].xa[i]
                else:
                    raise RuntimeError
                nlp_cf_phase.defect[i] += defect

                # lg endpoint defect
                if problem.spectral_method == "lg":
                    lg_defect = nlp_cf_phase.lg_defect
                    lg_defect[i][:] = mesh.b_lg[p] @ dv.phase[p].xa[i]

            # integral evaluation defect
            for i in range(problem.nq[p]):
                nlp_cf_phase.integral[i] += (mesh.w[p] * ci.phase[p].integrand[i]).sum() * dt2
                nlp_cf_phase.integral[i] -= dv.phase[p].q[i]

            # path
            for i in range(problem.nh[p]):
                nlp_cf_phase.path[i] += ci.phase[p].path[i]

            # duration
            nlp_cf_phase.duration[:] = tf - t0

        # call the user-defined discrete function
        if problem.nd > 0:
            discrete_function(di)
            cf.discrete[:] = di.discrete

        result: FloatArray = cf.c.copy()
        return result

    # end callback function

    return eval_constraints


def make_nlp_objective_gradient(nlp: NLP) -> Callable[[FloatArray], FloatArray]:
    """Construct NLP objective gradient function from optimal control problem definition.

    Returns
    -------
    Callable[[FloatArray], FloatArray]
        The gradient callback function
    """
    problem = nlp.problem
    gradient: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    arg = ObjectiveGradientArg(problem, dv)

    # begin callback function

    def eval_nlp_objective_gradient(
        z: FloatArray,
    ) -> FloatArray:
        """NLP Objective callback function."""
        dv_key: DVKey

        dv.z[:] = z
        arg.gradient.clear()
        nlp.functions.objective_gradient(arg)

        gradient.z[:] = 0
        for dv_key in nlp.functions.objective_gradient_structure:
            gradient.var_dict[dv_key][0] = arg.gradient[dv_key]

        return gradient.z

    # end callback function

    return eval_nlp_objective_gradient


def make_eval_continuous(nlp: NLP) -> Callable[[FloatArray, int], ContinuousArg[np.float64]]:
    """Make callback function to evaluate the problem continuous functions.

    Make callback function that returns the results of calling the continuous function,
    and optionally the continuous_jacobian and continuous_hessian functions. Consumed
    by the Jacobian and Hessian assembly plans.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    Callable[[NDArray, int], ContinuousArg]
        Callback function
    """
    problem = nlp.problem
    mesh = nlp.mesh

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    ci: ContinuousArg[np.float64] = ContinuousArg(
        problem,
        dv,
        dtype=np.float64,
        tau_u=mesh.tau_u,
    )
    if problem.np > 0:
        continuous_function = cast(ContinuousFunctionFloat, nlp.functions.continuous)

    # begin callback function

    def eval_continuous(z: FloatArray, order: int = 0) -> ContinuousArg[np.float64]:
        # distribute nlp decision variables passed from pyipopt to x0, xf, q, t0, tf
        # (for each phase) and s
        ci._sync(z)

        # call the user-defined continuous constraint function
        ci._phase_list = tuple(range(problem.np))
        continuous_function(ci)

        if order == 0:
            return ci

        ci._phase_list = tuple(range(problem.np))
        for p in range(problem.np):
            ci.phase[p].jacobian.clear()
        nlp.functions.continuous_jacobian(cast(ContinuousJacobianArg, ci))

        if order == 1:
            return ci

        for p in range(problem.np):
            ci.phase[p].hessian.clear()
        nlp.functions.continuous_hessian(cast(ContinuousHessianArg, ci))

        return ci

    # end callback function

    return eval_continuous


def make_eval_discrete_jacobian(nlp: NLP) -> Callable[[FloatArray], Sequence[np.float64 | float]]:
    """Evaluate the contribution of the discrete constraints to the NLP Jacobian.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    Callable[[NDArray], list]
        Callback function to evaluate the contribution of the discrete constraints to the
        NLP Jacobian.
    """
    problem = nlp.problem
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    arg = DiscreteJacobianArg(problem, dv)

    # begin callback function

    def eval_discrete_jacobian(z: FloatArray) -> Sequence[np.float64 | float]:
        """Evaluate optimal control problem discrete Jacobian.

        This callback function is autogenerated by `make_eval_discrete_jacobian`

        Parameters
        ----------
        z : NDArray
            The NLP decision variable arrays

        Returns
        -------
        List[np.float64]
            The Jacobian evaluated for each value in the discrete Jacobian structure.
        """
        # distribute nlp decision variables passed from pyipopt to x0, xf, q, t0, tf
        # (for each phase) and s
        dv.z[:] = z

        # call and return the user-defined discrete jacobian
        if problem.nd == 0:
            return []
        nlp.functions.discrete_jacobian(arg)
        structure = nlp.functions.discrete_jacobian_structure
        return [arg.jacobian[djs_term] for djs_term in structure]

    # end callback function

    return eval_discrete_jacobian
