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

required by Ipopt. In addition, it includes as an attribute `ipopt_kwargs`, which is a
dictionary of the keyword arguments needed to instantiate a `cyipopt.Problem` instance.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, cast

# third party imports
import numpy as np
from scipy.sparse import csr_matrix

# package imports
from .input_args import (
    ContinuousArg,
    ContinuousFunctionFloat,
    ContinuousHessianArg,
    ContinuousJacobianArg,
    DiscreteArg,
    DiscreteFunctionFloat,
    DiscreteHessianArg,
    DiscreteJacobianArg,
    ObjectiveArg,
    ObjectiveFunctionFloat,
    ObjectiveGradientArg,
    ObjectiveHessianArg,
    ProblemFunctions,
)
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable, Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .mesh import Mesh
    from .types_ import DHS, OHS, CHSTerm, CJSTerm, DHSTerm, DJSTerm, DVKey, OHSTerm

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
        mesh: yapss._private.mesh.Mesh,
    ) -> None:
        # store arguments
        self.problem: yapss.Problem = problem
        self.functions: ProblemFunctions = functions
        self.mesh = mesh

        # TODO: figure out where this should go, if at all:
        self.nj: int
        nlp = self
        self._objective = make_nlp_objective(nlp)
        self._constraints = make_nlp_constraints(nlp)
        self._gradient = make_nlp_objective_gradient(nlp)

        # constraint jacobian
        self.irow: tuple[int, ...] = ()
        self.jcol: tuple[int, ...] = ()
        self.jconst: FloatArray
        get_nlp_jacobian_structure(nlp)
        self._jacobian = make_nlp_constraint_jacobian(nlp)

        simplify_jacobian(nlp)

        self.nlp_hessian_structure: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

        self.intermediate: Intermediate | None = None
        self._hessian: Callable[[FloatArray, FloatArray, np.float64], FloatArray]
        if problem.derivatives.order == "second":
            self.nlp_hessian_structure = get_nlp_hessian_structure(nlp)
            self._hessian = make_nlp_hessian(nlp)
            simplify_hessian(nlp)
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


def simplify_hessian(nlp: NLP) -> None:
    """Eliminate redundant indices in the NLP hessian structure."""
    hs = nlp.nlp_hessian_structure

    # get sorted, unique row column pairs
    row, col = hs
    n = len(row)
    rc = [(row[i], col[i]) for i in range(n)]
    rc = list(set(rc))
    rc.sort()
    irow, jcol = tuple(item[0] for item in rc), tuple(item[1] for item in rc)

    # make hessian structure lower triangular
    if irow:
        temp = [
            [irow[i], jcol[i]] if irow[i] >= jcol[i] else [jcol[i], irow[i]]
            for i in range(len(irow))
        ]
        irow, jcol = tuple(zip(*temp))

    # make dictionary that will have values that are the row index of sparse matrix
    rc_dict = {item: k for k, item in enumerate(rc)}

    a_value: FloatArray
    a_value = np.ones(n)
    a_row = [rc_dict[row[i], col[i]] for i in range(n)]
    a_col = list(range(n))

    if row:
        a = csr_matrix((a_value, (a_row, a_col)))
        nlp.nlp_hessian_structure = tuple(irow), tuple(jcol)
        hessian_long = nlp._hessian

        def hessian_short(
            z: FloatArray,
            lam: FloatArray,
            objective_factor: np.float64,
        ) -> FloatArray:
            return np.array(
                a * hessian_long(z, lam, np.float64(objective_factor)),
                dtype=np.float64,
            )

        nlp._hessian = hessian_short


def simplify_jacobian(nlp: NLP) -> None:
    """Eliminate redundant indices in the NLP hessian structure."""
    # TODO: Move to the jacobian routine
    js = nlp.irow, nlp.jcol

    # get sorted, unique row column pairs
    row: tuple[int, ...] | list[int]
    col: tuple[int, ...] | list[int]

    row, col = js
    row = [int(r) for r in row]
    col = [int(c) for c in col]
    n = len(row)
    rc = [(row[i], col[i]) for i in range(n)]  # TODO: use zip?
    rc = list(set(rc))
    rc.sort()
    irow, jcol = [item[0] for item in rc], [item[1] for item in rc]

    # make dictionary have ??? values that are the row index of sparse matrix
    rc_dict = {item: k for k, item in enumerate(rc)}

    a_value: FloatArray
    a_value = np.ones(n)
    a_row = [rc_dict[row[i], col[i]] for i in range(n)]
    a_col = list(range(n))

    # n == 0 is edge case in which there are no constraints.
    if n > 0:
        a = csr_matrix((a_value, (a_row, a_col)))
        nlp.irow, nlp.jcol = tuple(irow), tuple(jcol)
        jacobian_long = nlp._jacobian

        def jacobian_short(z: FloatArray) -> FloatArray:
            return np.array(a * jacobian_long(z), dtype=float)

        nlp._jacobian = jacobian_short


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
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, np.float64)
    ci: ContinuousArg[np.float64] = ContinuousArg(problem, dv, dtype=np.float64)
    di: DiscreteArg[np.float64] = DiscreteArg(problem, dv, np.float64)
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    mesh: Mesh
    mesh = nlp.mesh

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
        # mesh
        dv.z[:] = z

        for p in range(problem.np):
            dv_phase = dv.phase[p]

            # form time vector
            t0 = dv_phase.t0[0]
            tf = dv_phase.tf[0]
            ci.phase[p].time[:] = mesh.tau_u[p] * (tf - t0) / 2 + (t0 + tf) / 2

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

            # zero_mode for lgr
            if problem.spectral_method == "lgr":
                col_points = problem.mesh.phase[p].collocation_points
                j = 0
                for i, nc in enumerate(col_points):
                    for ix in range(problem.nx[p]):
                        nlp_cf_phase.zero_mode[ix][i] = (
                            mesh.b_lgr[p][i] @ dv.phase[p].x[ix][j : j + nc + 1]
                        )
                    j += nc

        # call the user-defined discrete function
        if problem.nd > 0:
            discrete_function(di)
            cf.discrete[:] = di.discrete

        return cf.c.copy()

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


def make_nlp_constraint_jacobian(nlp: NLP) -> Callable[[FloatArray], FloatArray]:
    """Construct the NLP constraint Jacobian function from the problem definition.

    Returns
    -------
    :meth:`jacobian` function
    """
    jacobian_constant: FloatArray
    nj: int

    problem = nlp.problem
    spectral_method = problem.spectral_method
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, int)
    cf.c[:] = list(range(len(cf.c)))

    jacobian_constant = nlp.jconst

    nj = nlp.nj
    eval_discrete_jacobian = make_eval_discrete_jacobian(nlp)
    eval_continuous = make_eval_continuous(nlp)

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)

    # begin callback function

    def eval_nlp_jacobian(
        z: FloatArray,
    ) -> FloatArray:
        mesh = nlp.mesh

        j = nj
        # linear terms
        jacobian: FloatArray = jacobian_constant.copy()

        # nonlinear terms
        if problem.np > 0:
            c_output = c_out_2 = eval_continuous(z, 1)
            dv.z[:] = z

        for p in range(problem.np):
            f = c_output.phase[p].dynamics
            g = c_output.phase[p].integrand

            col_points = problem.mesh.phase[p].collocation_points
            nc = sum(col_points)
            ni = 1
            if spectral_method == "lgl":
                nw = nc - len(col_points) + 1
                defect_index = cf.phase[p].defect_index
                nf = 1
            elif spectral_method == "lg":
                nw = nc
                defect_index = list(range(nc))
                nf = 0
                ni = 0
            else:
                nw = nc
                defect_index = list(range(nc))
                nf = 0

            w = mesh.w[p]
            t0 = dv.phase[p].t0[0]
            tf = dv.phase[p].tf[0]
            dt = (tf - t0) / 2

            for cjs_term in nlp.functions.continuous_jacobian_structure[p]:
                (cf_name, _), (cv_name, _) = cjs_term

                # sanitize inputs in case they are scalar
                jac_term: NDArray[np.float64] = np.zeros([nw], dtype=float)
                jac_term[:] = c_out_2.phase[p].jacobian[cjs_term]
                if cf_name == "f":
                    jac_term = dt * jac_term[defect_index]
                    npoints = nc
                    index = defect_index
                elif cf_name == "g":
                    jac_term = dt * w * jac_term
                    npoints = nw
                    index = list(range(nw))
                elif cf_name == "h":
                    npoints = nw
                    index = list(range(nw))
                else:  # pragma: no cover
                    msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                    raise ValueError(msg)

                if cv_name in ("x", "u", "s"):
                    jacobian[j : j + npoints] = jac_term
                    j += npoints
                elif cv_name == "t":
                    jac_term1 = jac_term * (1 - mesh.tau_u[p][index]) / 2
                    jacobian[j : j + npoints - nf] = jac_term1[: npoints - nf]
                    j += npoints - nf
                    jac_term2 = jac_term * (1 + mesh.tau_u[p][index]) / 2
                    jacobian[j : j + npoints - ni] = jac_term2[ni:]
                    j += npoints - ni
                else:  # pragma: no cover
                    msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                    raise ValueError(msg)

            # df/t0 and df/dtf due to (tf-t0)/2 in state equation
            for i in range(problem.nx[p]):
                jacobian[j : j + nc] = -0.5 * f[i][defect_index]
                j += nc
                jacobian[j : j + nc] = 0.5 * f[i][defect_index]
                j += nc

            # d integral / dt0 and d integral / dtf due to (tf-t0)/2 in quadrature
            for i in range(problem.nq[p]):
                integral = (w * g[i]).sum()
                jacobian[j] = -0.5 * integral
                j += 1
                jacobian[j] = +0.5 * integral
                j += 1

            # TODO: move duration up front with other constant terms
            # duration
            j += 2

            # zero_mode for lgr
            if problem.spectral_method == "lgr":
                j += (sum(col_points) + len(col_points)) * problem.nx[p]

        # discrete terms
        jacobian[j:] = eval_discrete_jacobian(z)

        return jacobian

    # end callback function

    return eval_nlp_jacobian


def make_nlp_hessian(nlp: NLP) -> Callable[[FloatArray, FloatArray, np.float64], FloatArray]:
    """Construct the Hessian of the Lagrangian of the NLP.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    Callable[[FloatArray, FloatArray, float], FloatArray]
        The :meth:`hessian` function
    """
    problem = nlp.problem
    spectral_method = problem.spectral_method
    mesh = nlp.mesh
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)
    objective_input = ObjectiveHessianArg(problem, dv)
    discrete_input = DiscreteHessianArg(problem, dv)

    row, _ = nlp.nlp_hessian_structure
    nnzh = len(row)
    hessian: FloatArray
    hessian = np.zeros([nnzh], dtype=np.float64)
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    lambda_: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)
    eval_continuous = make_eval_continuous(nlp)

    # begin callback function

    def eval_nlp_hessian(
        z: FloatArray,
        lam: FloatArray,
        objective_factor: np.float64,
    ) -> FloatArray:
        # typing
        chs_term: CHSTerm
        cjs_term: CJSTerm
        ohs_term: OHSTerm
        dhs_term: DHSTerm

        rhs: FloatArray

        hessian[:] = 0.0

        continuous_jacobian_structure = nlp.functions.continuous_jacobian_structure
        lambda_.c[:] = lam
        if problem.np > 0:
            c_output1_ = eval_continuous(z, 2)
            if c_output1_ is not None:
                c_output1: ContinuousArg[np.float64] = c_output1_
            else:
                msg = "Internal error: continuous output is None"  # type: ignore[unreachable]
                raise RuntimeError(msg)
        ih = 0
        for p in range(problem.np):
            tau = mesh.tau_u[p]
            t0 = dv.phase[p].t0[0]
            tf = dv.phase[p].tf[0]

            # get mesh structure
            col_points = problem.mesh.phase[p].collocation_points
            nc = sum(col_points)
            if spectral_method == "lgl":
                nw = nc - len(col_points) + 1
                defect_index = cf.phase[p].defect_index
            else:
                nw = nc
                defect_index = list(range(nc))

            for chs_term in nlp.functions.continuous_hessian_structure[p]:
                (cf_name, i), (cv_name1, _), (cv_name2, _) = chs_term
                if cf_name == "f":
                    n = nc
                    index = defect_index
                else:
                    n = nw
                    index = list(range(n))

                if cv_name1 == "t" and cv_name2 == "t":
                    term: FloatArray = c_output1.phase[p].hessian[chs_term][index]
                    if cf_name == "f":
                        term *= 0.125 * (tf - t0) * lambda_.phase[p].defect[i]
                    elif cf_name == "g":
                        term *= mesh.w[p] * 0.125 * (tf - t0) * lambda_.phase[p].integral[i]
                    elif cf_name == "h":
                        term *= 0.25 * lambda_.phase[p].path[i]
                    else:  # pragma: no cover
                        msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                        raise ValueError(msg)
                    hessian[ih] += ((1 - tau[index]) ** 2 * term).sum()
                    ih += 1
                    hessian[ih] += ((1 - tau[index] ** 2) * term).sum()
                    ih += 1
                    hessian[ih] += ((1 + tau[index]) ** 2 * term).sum()
                    ih += 1

                elif cv_name1 == "t" or cv_name2 == "t":
                    term = c_output1.phase[p].hessian[chs_term][index]
                    if cf_name == "f":
                        term *= 0.25 * (tf - t0) * lambda_.phase[p].defect[i]
                    elif cf_name == "g":
                        term *= lambda_.phase[p].integral[i]
                        term *= 0.25 * (tf - t0) * mesh.w[p]
                    elif cf_name == "h":
                        term *= 0.5 * lambda_.phase[p].path[i]
                    else:  # pragma: no cover
                        msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                        raise ValueError(msg)
                    hessian[ih : ih + n] += (1 - tau[index]) * term
                    ih += n
                    hessian[ih : ih + n] += (1 + tau[index]) * term
                    ih += n

                else:
                    term = c_output1.phase[p].hessian[chs_term]
                    if cf_name == "f":
                        term = 0.5 * (tf - t0) * lambda_.phase[p].defect[i] * term[index]
                    elif cf_name == "g":
                        term *= lambda_.phase[p].integral[i] * (0.5 * (tf - t0)) * mesh.w[p]
                    elif cf_name == "h":
                        term = lambda_.phase[p].path[i] * term
                    else:  # pragma: no cover
                        msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                        raise ValueError(msg)

                    hessian[ih : ih + n] += term
                    ih += n

            # defect and integral jacobian terms
            for cjs_term in continuous_jacobian_structure[p]:
                (cf_name, jj), (cv_name, _) = cjs_term

                if cf_name in ("f", "g"):
                    if cf_name == "f":
                        index = defect_index
                        n = nc
                    else:
                        n = nw
                        index = list(range(n))

                    if cv_name == "t":
                        if cf_name == "f":
                            term = (
                                lambda_.phase[p].defect[jj]
                                * c_output1.phase[p].jacobian[cjs_term][index]
                            )
                        elif cf_name == "g":
                            term = (
                                lambda_.phase[p].integral[jj]
                                * c_output1.phase[p].jacobian[cjs_term]
                                * mesh.w[p]
                            )
                        else:  # pragma: no cover
                            msg = (
                                f"Invalid continuous Jacobian structure term {cjs_term} "
                                f"in phase {p}"
                            )
                            raise ValueError(msg)

                        hessian[ih] += -0.5 * ((1 - tau[index]) * term).sum()  # t0, t0
                        ih += 1
                        hessian[ih] += -0.5 * (tau[index] * term).sum()  # t0, tf
                        ih += 1
                        hessian[ih] += 0.5 * ((1 + tau[index]) * term).sum()  # tf, tf
                        ih += 1

                    else:
                        rhs = np.zeros([nw], dtype=float)
                        rhs[:] = 0.5 * c_output1.phase[p].jacobian[cjs_term]
                        rhs = rhs[index]

                        if cf_name == "f":
                            rhs *= lambda_.phase[p].defect[jj]
                        else:  # cf_name == "g"
                            rhs *= mesh.w[p] * lambda_.phase[p].integral[jj]

                        if cv_name in ("x", "u", "s"):
                            hessian[ih : ih + n] = -rhs
                            ih += n
                            hessian[ih : ih + n] = +rhs
                            ih += n
                        else:  # pragma: no cover
                            msg = (
                                f"Invalid continuous Jacobian structure term {cjs_term} "
                                f"in phase {p}"
                            )
                            raise ValueError(msg)

                elif cf_name == "h":
                    pass

                else:  # pragma: no cover
                    msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                    raise ValueError(msg)

        dv.z[:] = z
        objective_input.hessian.clear()
        nlp.functions.objective_hessian(objective_input)

        ohs = nlp.functions.objective_hessian_structure
        objective_hessian = np.zeros(len(ohs))
        for i, ohs_term in enumerate(ohs):
            objective_hessian[i] = objective_input.hessian[ohs_term]

        nc = len(ohs)
        hessian[ih : ih + nc] = objective_hessian * objective_factor
        ih += nc

        if problem.nd > 0:
            dv.z[:] = z
            objective_input.hessian.clear()
            nlp.functions.discrete_hessian(discrete_input)

            dhs = nlp.functions.discrete_hessian_structure
            discrete_hessian = np.zeros(len(dhs))

            for ii, dhs_term in enumerate(dhs):
                discrete_hessian[ii] += (
                    discrete_input.hessian[dhs_term] * lambda_.discrete[dhs_term[0]]
                )

            nc = len(dhs)
            hessian[ih : ih + nc] = discrete_hessian
            ih += nc

        return hessian

    # end callback function

    return eval_nlp_hessian


def make_eval_continuous(nlp: NLP) -> Callable[[FloatArray, int], ContinuousArg[np.float64]]:
    """Make callback function to evaluate the problem continuous functions.

    Make callback function that returns the results of calling the continuous function,
    and optionally the continuous_jacobian and continuous_hessian functions. Called by
    `make_nlp_constraint_jacobian`.

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
    ci: ContinuousArg[np.float64] = ContinuousArg(problem, dv, dtype=np.float64)
    nlp_dv: DVStructure[np.float64]
    nlp_dv = ci._dv

    if problem.np > 0:
        continuous_function = cast(ContinuousFunctionFloat, nlp.functions.continuous)

    # begin callback function

    def eval_continuous(z: FloatArray, order: int = 0) -> ContinuousArg[np.float64]:
        # distribute nlp decision variables passed from pyipopt to x0, xf, q, t0, tf
        # (for each phase) and s
        nlp_dv.z[:] = z

        for p in range(problem.np):
            nlp_dv_phase = nlp_dv.phase[p]

            # form time vector
            t0 = nlp_dv_phase.t0[0]
            tf = nlp_dv_phase.tf[0]
            time = mesh.tau_u[p] * (tf - t0) / 2 + (t0 + tf) / 2
            ci.phase[p].time[:] = time

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
    # This section of code could be inside eval_nlp_jacobian, but that function is already
    # too long

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

        discrete_jacobian = []

        # call and return the user-defined discrete jacobian
        if problem.nd > 0:
            nlp.functions.discrete_jacobian(arg)
            djs_term: DJSTerm
            for djs_term in nlp.functions.discrete_jacobian_structure:
                discrete_jacobian.append(arg.jacobian[djs_term])  # noqa: PERF401

        return discrete_jacobian

    # end callback function

    return eval_discrete_jacobian


def get_nlp_jacobian_structure(
    nlp: NLP,
) -> tuple[tuple[int, ...], tuple[int, ...], FloatArray, int]:
    """Find the Jacobian structure of the constraints for the NLP.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    tuple[int, ...]
        The row indices of the nonzero elements of the Jacobian.
    tuple[int, ...]
        The column indices of the nonzero elements of the Jacobian.
    NDArray
        An array with the constant values of the Jacobian.
    int
        The number of constant values.
    """
    # typing
    dv_key: DVKey

    problem = nlp.problem
    spectral_method = problem.spectral_method
    mesh = nlp.mesh

    r: list[int]

    c: list[int] = []
    c1: list[int] = []
    c2: list[int] = []
    defect_index: list[int] = []

    row: list[int] = []
    col: list[int] = []
    linear_jacobian: list[np.float64 | float] = []

    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, int)
    dv.z[:] = list(range(len(dv.z)))
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, int)
    cf.c[:] = list(range(len(cf.c)))

    # for each phase
    for p in range(problem.np):
        dv_phase = dv.phase[p]
        cf_phase = cf.phase[p]

        # d(defect)/dx due to D terms
        for i in range(problem.nx[p]):
            r_, c_ = mesh.d[p].nonzero()
            row += list(cf_phase.defect[i][r_])
            col += list(dv_phase.xa[i][c_])
            linear_jacobian += list((-mesh.d[p]).data)

        # d(defect)/dx due to b terms
        if problem.spectral_method == "lg":
            for i in range(problem.nx[p]):
                r_, c_ = mesh.b_lg[p].nonzero()
                row += list(cf_phase.lg_defect[i][r_])
                col += list(dv_phase.xa[i][c_])
                linear_jacobian += list(mesh.b_lg[p].data)

        # d(integral defect)/dq
        nq = problem.nq[p]
        row += list(cf_phase.integral)
        col += list(dv_phase.q)
        linear_jacobian += nq * [-1.0]

    # Variable terms
    nj = len(row)
    jacobian_structure = nlp.functions.continuous_jacobian_structure

    # for each phase
    for p in range(problem.np):
        dv_phase = dv.phase[p]
        cf_phase = cf.phase[p]
        if spectral_method == "lgl":
            defect_index = cf_phase.defect_index
        col_points = problem.mesh.phase[p].collocation_points
        nc = sum(col_points)
        if spectral_method == "lgl":
            ni = nc - len(col_points) + 1
            nw = ni
        else:
            ni = nc + 1
            nw = nc

        # d {f | g | h} / d {x | u | t | s} terms
        for cjs_term in jacobian_structure[p]:
            (cf_name, i), (cv_name, j) = cjs_term

            ni_ = 1
            if spectral_method == "lgl":
                if cf_name == "f":
                    index = defect_index
                    n = nc
                else:
                    index = list(range(ni))
                    n = ni
                n0 = n - 1
            else:
                index = list(range(nc))
                n = nc
                n0 = nc
            if spectral_method == "lg":
                ni_ = 0

            # generate row indices
            if cf_name == "f":
                r = list(cf_phase.defect[i])
            elif cf_name == "g":
                r = nw * [cf_phase.integral[i]]
            elif cf_name == "h":
                r = list(cf_phase.path[i])
            else:  # pragma: no cover
                msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                raise ValueError(msg)

            # generate column indices
            if cv_name == "x":
                c = list(dv_phase.x[j][index])
            elif cv_name == "u":
                c = list(dv_phase.u[j][index])
            elif cv_name == "s":
                c = n * [dv.s[j]]
            elif cv_name == "t":
                c1 = n0 * [dv_phase.t0[0]]
                c2 = (n - ni_) * [dv_phase.tf[0]]
            else:  # pragma: no cover
                msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                raise ValueError(msg)

            if cv_name in ("x", "u", "s"):
                row += r
                col += c
                linear_jacobian += n * [0.0]
            elif cv_name == "t":
                row += r[:n0]
                row += r[ni_:]
                col += c1
                col += c2
                linear_jacobian += n0 * [0.0]
                linear_jacobian += (n - ni_) * [0.0]
            else:  # pragma: no cover
                msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
                raise ValueError(msg)

        # df/dt0, df/dtf due to (tf-t0)/2 term
        for i in range(problem.nx[p]):
            row += list(cf_phase.defect[i])
            col += nc * [dv_phase.t0[0]]
            row += list(cf_phase.defect[i])
            col += nc * [dv_phase.tf[0]]
            linear_jacobian += 2 * nc * [0.0]

        # d integral / dt0 and d integral / dtf due to (tf-t0)/2 in quadrature
        nq = problem.nq[p]
        for i in range(nq):
            row += list(2 * [cf_phase.integral[i]])
        col += list(nq * [dv_phase.t0[0], dv_phase.tf[0]])
        linear_jacobian += 2 * nq * [0.0]

        # duration
        row += list(cf_phase.duration)
        col += list(dv_phase.tf)
        row += list(cf_phase.duration)
        col += list(dv_phase.t0)
        linear_jacobian += [1.0, -1.0]

        # zero_mode for lgr
        if problem.spectral_method == "lgr":
            col_points = problem.mesh.phase[p].collocation_points
            j = 0
            for i, nc in enumerate(col_points):
                for ix in range(problem.nx[p]):
                    row += (nc + 1) * [cf_phase.zero_mode[ix][i]]
                    col += list(dv_phase.x[ix][j : j + nc + 1])
                    linear_jacobian += list(mesh.b_lgr[p][i])
                j += nc

    # discrete constraints
    if problem.nd > 0:
        for i, dv_key in nlp.functions.discrete_jacobian_structure:
            col.append(dv.var_dict[dv_key][0])
            row.append(cf.discrete[i])
            linear_jacobian.append(0)

    # store results for use by jacobian function
    irow = tuple(row)
    jcol = tuple(col)
    jconst: NDArray[np.float64] = np.array(linear_jacobian, dtype=float)

    # a crude check to make sure nothing has gone wrong:
    if len(irow) != len(jcol) or len(irow) != len(jconst):
        raise RuntimeError
    nlp.irow = irow
    nlp.jcol = jcol
    nlp.jconst = jconst
    # TODO: remove nj?
    nlp.nj = nj

    return irow, jcol, jconst, nj


def get_nlp_hessian_structure(nlp: NLP) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Find the Hessian structure for the Lagrangian of the NLP.

    Parameters
    ----------
    nlp : NLP

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...]]
        Row and column indices of the nonzero elements of the Hessian.
    """
    # typing
    dv_key1: DVKey
    dv_key2: DVKey
    chs_term: CHSTerm
    cjs_term: CJSTerm
    dhs_term: DHSTerm
    ohs_term: OHSTerm
    ohs: OHS
    dhs: DHS

    ohs = nlp.functions.objective_hessian_structure
    dhs = nlp.functions.discrete_hessian_structure

    problem = nlp.problem
    spectral_method = problem.spectral_method
    if spectral_method not in ("lg", "lgr", "lgl"):
        raise RuntimeError

    # index structure of nlp problem
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, int)
    dv.z[:] = list(range(len(dv.z)))
    cf: CFStructure[np.float64] = get_nlp_cf_structure(problem, int)
    cf.c[:] = list(range(len(cf.c)))

    # continuous jacobian and hessian structures
    cjs = nlp.functions.continuous_jacobian_structure
    chs = nlp.functions.continuous_hessian_structure

    row2: list[int] = []
    col2: list[int] = []
    defect_index: list[int] = []

    for p in range(problem.np):
        dv_phase = dv.phase[p]

        cf_phase = cf.phase[p]
        if spectral_method == "lgl":
            defect_index = cf_phase.defect_index
        col_points = problem.mesh.phase[p].collocation_points
        nc = sum(col_points)

        for chs_term in chs[p]:
            (cf_name, _), (cv_name1, j), (cv_name2, k) = chs_term

            # determine number of elements n in hessian
            if spectral_method == "lgl":
                if cf_name == "f":
                    n = nc
                    index = defect_index
                else:
                    n = nc - len(col_points) + 1
                    index = list(range(n))
            else:
                n = nc
                index = list(range(nc))

            # t,t derivative terms
            if cv_name1 == "t" and cv_name2 == "t":
                i_t0, i_tf = dv_phase.t0[0], dv_phase.tf[0]
                row2 += [i_t0, i_t0, i_tf]
                col2 += [i_t0, i_tf, i_tf]

            # {x|u|s},t derivative terms
            elif cv_name1 == "t" or cv_name2 == "t":
                # swap keys so that key3 is not "t"
                if cv_name2 == "t":
                    cv_name2, k = cv_name1, j

                # x, u, and s cases
                if cv_name2 == "x":
                    row2 += 2 * list(dv_phase.x[k][index])
                elif cv_name2 == "u":
                    row2 += 2 * list(dv_phase.u[k][index])
                elif cv_name1 == "s":
                    row2 += 2 * n * [dv.s[j]]
                else:  # pragma: no cover
                    msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                    raise ValueError(msg)
                col2 += n * [dv_phase.t0[0]]
                col2 += n * [dv_phase.tf[0]]

            # {x|u|s},{x|u|s} derivative terms
            else:
                if cv_name1 == "x":
                    row2 += list(dv_phase.x[j][index])
                elif cv_name1 == "u":
                    row2 += list(dv_phase.u[j][index])
                elif cv_name1 == "s":
                    row2 += n * [dv.s[j]]
                else:  # pragma: no cover
                    msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                    raise ValueError(msg)

                if cv_name2 == "x":
                    col2 += list(dv_phase.x[k][index])
                elif cv_name2 == "u":
                    col2 += list(dv_phase.u[k][index])
                elif cv_name2 == "s":
                    col2 += n * [dv.s[k]]
                else:  # pragma: no cover
                    msg = f"Invalid continuous Hessian structure term {chs_term} in phase {p}"
                    raise ValueError(msg)

        # defect and integral jacobian terms
        for cjs_term in cjs[p]:
            (cf_name, _), (cv_name, j) = cjs_term

            if spectral_method == "lgl":
                if cf_name == "f":
                    n = nc
                    index = defect_index
                else:
                    n = nc - len(col_points) + 1
                    index = list(range(n))
            else:
                n = nc
                index = list(range(nc))

            # t derivative terms
            if cv_name == "t":
                if cf_name in ("f", "g"):
                    i_t0, i_tf = dv_phase.t0[0], dv_phase.tf[0]
                    row2 += [i_t0, i_t0, i_tf]
                    col2 += [i_t0, i_tf, i_tf]
                elif cf_name == "h":
                    pass
                else:  # pragma: no cover
                    msg = f"Invalid Jacobian continuous structure term {cjs_term} in phase {p}"
                    raise ValueError(msg)

            elif cf_name in ("f", "g"):
                if cv_name == "x":
                    col2 += 2 * list(dv_phase.x[j][index])
                elif cv_name == "u":
                    col2 += 2 * list(dv_phase.u[j][index])
                elif cv_name == "s":
                    col2 += 2 * n * [dv.s[j]]
                else:  # pragma: no cover
                    msg = f"Invalid  continuous Jacobian structure term {cf_name} in phase {p}"
                    raise ValueError(msg)
                row2 += n * [dv_phase.t0[0]]
                row2 += n * [dv_phase.tf[0]]

            elif cf_name == "h":
                pass

            else:  # pragma: no cover
                msg = f"Invalid  continuous Jacobian structure term {cf_name} in phase {p}"
                raise ValueError(msg)

    row = row2
    col = col2

    for ohs_term in ohs:
        dv_key1, dv_key2 = ohs_term
        row.append(dv.var_dict[dv_key1][0])
        col.append(dv.var_dict[dv_key2][0])

    if problem.nd > 0:
        for dhs_term in dhs:
            _, dv_key1, dv_key2 = dhs_term
            row.append(dv.var_dict[dv_key1][0])
            col.append(dv.var_dict[dv_key2][0])

    return tuple(row), tuple(col)
