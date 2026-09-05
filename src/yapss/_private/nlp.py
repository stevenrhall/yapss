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
consume. `ContinuousEvaluator` is shared by the constraint, Jacobian, and Hessian
callbacks so that the user's continuous functions are evaluated once per point.

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

        # One evaluator of the user's continuous functions, shared by the constraint,
        # Jacobian, and Hessian callbacks. Ipopt asks for all three at each iterate,
        # and the Hessian needs the function and Jacobian values too (the chain-rule
        # terms through t0 and tf); sharing means each is computed once per point.
        self.eval_continuous = ContinuousEvaluator(nlp)

        self._objective = make_nlp_objective(nlp)
        self._constraints = make_nlp_constraints(nlp, self.eval_continuous)
        self._gradient = make_nlp_objective_gradient(nlp)

        # constraint jacobian: structure and evaluator come from one assembly plan,
        # so their entries correspond by construction; see the jacobian module
        (self.irow, self.jcol), self._jacobian = make_nlp_jacobian(
            nlp,
            self.eval_continuous,
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
                self.eval_continuous,
            )

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


def make_nlp_constraints(
    nlp: NLP,
    eval_continuous: ContinuousEvaluator,
) -> Callable[[FloatArray], FloatArray]:
    """Construct the NLP constraint function from the optimal control problem definition.

    Parameters
    ----------
    nlp : NLP
    eval_continuous : ContinuousEvaluator
        The shared evaluator of the continuous functions.

    Returns
    -------
    Callable[[FloatArray], FloatArray]
        NLP constraint function
    """
    problem = nlp.problem
    mesh: Mesh = nlp.mesh
    # the evaluator owns the decision-variable structure that its argument reads
    # from; the defect terms below read the same one, so they see the same z
    dv: DVStructure[np.float64] = eval_continuous.dv
    di: DiscreteArg[np.float64] = DiscreteArg(problem, dv, np.float64)
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)

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
        # evaluates the user-defined continuous function, or returns the values
        # already computed at this z by an earlier callback
        ci = eval_continuous(z, 0)

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


class ContinuousEvaluator:
    """Evaluate the continuous functions at a point, once per point.

    Returns the continuous function values and, on request, the continuous Jacobian
    and Hessian, in one `ContinuousArg` shared by the constraint, Jacobian, and Hessian
    callbacks. Ipopt asks for all three at each iterate, and the Hessian's chain-rule
    terms through ``t0`` and ``tf`` need the function and Jacobian values as well, so
    without sharing the user's function ran three times per iterate and its Jacobian
    twice -- under central differences, two full perturbation stencils.

    The cache is keyed on the value of ``z``. Ipopt's ``new_x`` flag is deliberately
    not used: cyipopt does not pass it through, it is set by vector identity rather
    than value, and the compare it would save costs a few microseconds against user
    functions that cost far more. A value compare can only ever cause a needless
    re-evaluation; a trusted flag that was wrong would serve derivatives from the
    wrong point silently.

    Orders are cumulative: order 1 adds the Jacobian to the function values, order 2
    adds the Hessian. A request for a higher order at the cached point evaluates only
    what is missing, and each order is recorded as done only after its evaluation
    returns, so an exception in a user callback leaves nothing marked as computed.
    The central-difference derivatives restore the function values after their
    stencils (pinned by ``test_central_difference_derivatives_restore_continuous_values``),
    which is what lets a lower order be served after a higher one.

    Parameters
    ----------
    nlp : NLP
    """

    def __init__(self, nlp: NLP) -> None:
        problem = nlp.problem
        self._functions = nlp.functions
        self._np = problem.np
        self.dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
        self.arg: ContinuousArg[np.float64] = ContinuousArg(
            problem,
            self.dv,
            dtype=np.float64,
            tau_u=nlp.mesh.tau_u,
        )
        self._z: FloatArray | None = None
        self._order_done = -1

    def __call__(self, z: FloatArray, order: int = 0) -> ContinuousArg[np.float64]:
        """Return the continuous argument evaluated at ``z`` through ``order``.

        Parameters
        ----------
        z : NDArray
            The NLP decision variable array
        order : int
            0 for the function values, 1 to add the Jacobian, 2 to add the Hessian.

        Returns
        -------
        ContinuousArg
            The shared argument. Valid until the next call at a different point.
        """
        arg = self.arg
        if self._z is None or not np.array_equal(z, self._z):
            # a new point: nothing computed here is valid until order 0 completes
            self._z = None
            self._order_done = -1
            arg._sync(z)
            self._z = np.array(z, dtype=np.float64, copy=True)

        if self._order_done < 0:
            arg._phase_list = tuple(range(self._np))
            if self._np > 0:
                cast(ContinuousFunctionFloat, self._functions.continuous)(arg)
            self._order_done = 0

        if order >= 1 and self._order_done < 1:
            arg._phase_list = tuple(range(self._np))
            for p in range(self._np):
                arg.phase[p].jacobian.clear()
            self._functions.continuous_jacobian(cast(ContinuousJacobianArg, arg))
            self._order_done = 1

        if order >= 2 and self._order_done < 2:  # noqa: PLR2004
            for p in range(self._np):
                arg.phase[p].hessian.clear()
            self._functions.continuous_hessian(cast(ContinuousHessianArg, arg))
            self._order_done = 2

        return arg


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
