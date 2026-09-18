"""

Assembly of the Hessian of the NLP Lagrangian.

The Hessian is assembled from the second derivatives of the user's continuous and
discrete functions by applying the chain rule of the collocation transcription: the
mapping from problem time t to mesh time tau is affine in the phase endpoints t0 and
tf, so each term picks up powers of (tf - t0)/2 and endpoint sensitivity factors
(1 -+ tau)/2, and the first derivatives of the continuous functions contribute
cross-terms in (t0, tf) as well.

Historically this lived in ``nlp.py`` as two functions that had to agree positionally:
a structure builder emitting (row, col) index pairs, and an evaluator writing values at
a manually-advanced cursor, each iterating the same term lists in the same order. A
mismatch did not crash -- the sparse fold applied before handing the structure to
Ipopt sums whatever entries it is given -- it silently produced a wrong Hessian.

This module instead builds a single **plan**: a sequence of :class:`HessianBlock`
objects, each owning both its index pairs and the closure that produces the matching
values. The structure is the concatenation of the blocks' indices, and the evaluator
fills the concatenation of the blocks' values, so the correspondence between an index
pair and its value is held by one object rather than by two loop bodies kept in step
by hand. Everything that is constant across evaluations -- index spans, quadrature
weights, the tau endpoint-sensitivity factors -- is computed once at build time.

The assembled values are identical to the historical implementation; the golden tests
in ``tests/modules/test_nlp_hessian_golden.py`` pin that equivalence.

"""

# future imports
from __future__ import annotations

# standard imports
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, assert_never

# third party imports
import numpy as np

# package imports
from .assembly import (
    Block,
    ContinuousContext,
    IndexTwins,
    PhaseGeometry,
    index_twins,
    over_points,
    phase_geometry,
    plan_layout,
    split_constants,
)
from .fold import fold_structure
from .input_args import DiscreteHessianArg, ObjectiveHessianArg, call_callback, require_keys
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    # third party imports
    from numpy.typing import NDArray

    from .input_args import ContinuousStore
    from .nlp import NLP
    from .types_ import CFName, CHSTerm, CJSTerm

    # package imports

    FloatArray = NDArray[np.float64]

__all__ = ["make_nlp_hessian"]


@dataclass
class HessianContext(ContinuousContext):
    """Per-evaluation inputs shared by all blocks.

    The evaluator fills this once per Hessian call, after synchronizing the decision
    variable and multiplier views; blocks read from it and from the views they
    captured at build time.
    """

    objective_factor: float = 0.0
    objective_hessian: dict[Any, Any] = field(default_factory=dict)
    discrete_hessian: dict[Any, Any] = field(default_factory=dict)


HessianBlock = Block[HessianContext]


def make_nlp_hessian(
    nlp: NLP,
    eval_continuous: Callable[[FloatArray, int], ContinuousStore[np.float64]],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    Callable[[FloatArray, FloatArray, np.float64], FloatArray],
]:
    """Build the Hessian structure and evaluator from a single assembly plan.

    Parameters
    ----------
    nlp : NLP
    eval_continuous : Callable
        Evaluator returning the continuous functions and their derivatives at a point,
        the NLP's shared ``nlp.ContinuousEvaluator``.

    Returns
    -------
    tuple
        The ``(rows, cols)`` Hessian structure, and the ``hessian(z, lam,
        objective_factor)`` callback that fills the matching values.
    """
    problem = nlp.problem
    functions = nlp.functions

    # float structures synchronized at the top of every evaluation; blocks capture
    # views into these, so the syncs update what the closures read
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)
    lambda_: CFStructure[np.float64] = get_nlp_cf_structure(problem, np.float64)

    objective_input = ObjectiveHessianArg(problem, dv)
    discrete_input = DiscreteHessianArg(problem, dv)
    if problem.derivative_method == "user":
        require_keys(objective_input, functions.objective_hessian_structure)
        require_keys(discrete_input, functions.discrete_hessian_structure)
    twins = index_twins(problem)

    blocks: list[HessianBlock] = []
    for p in range(problem.np):
        blocks += build_phase_blocks(nlp, phase_geometry(nlp, dv, twins, p), lambda_)
    blocks.append(build_objective_block(nlp, twins))
    if problem.nd > 0:
        blocks.append(build_discrete_block(nlp, twins, lambda_))

    row, col, slices = plan_layout(blocks)
    structure = tuple(row), tuple(col)

    hessian = np.zeros(len(row), dtype=np.float64)
    evaluated = split_constants(blocks, slices, hessian)  # the Hessian has no constant blocks
    context = HessianContext()

    def eval_nlp_hessian(
        z: FloatArray,
        lam: FloatArray,
        objective_factor: np.float64,
    ) -> FloatArray:
        # synchronize the views captured by the blocks
        dv.z[:] = z
        lambda_.c[:] = lam

        context.objective_factor = float(objective_factor)
        if problem.np > 0:
            context.continuous = eval_continuous(z, 2)

        call_callback(functions.objective_hessian, objective_input)
        context.objective_hessian = objective_input.hessian

        if problem.nd > 0:
            call_callback(functions.discrete_hessian, discrete_input)
            context.discrete_hessian = discrete_input.hessian

        for evaluate, block_slice in evaluated:
            hessian[block_slice] = evaluate(context)
        return hessian

    # fold the long structure onto unique coordinates, mirrored to the lower triangle;
    # the summing matrix adds coincident entries, which is what makes the totals right
    folded_rows, folded_cols, summing = fold_structure(row, col, lower_triangular=True)
    if summing is None:
        return structure, eval_nlp_hessian

    def eval_folded_hessian(
        z: FloatArray,
        lam: FloatArray,
        objective_factor: np.float64,
    ) -> FloatArray:
        return np.asarray(
            summing @ eval_nlp_hessian(z, lam, np.float64(objective_factor)),
            dtype=np.float64,
        )

    return (folded_rows, folded_cols), eval_folded_hessian


def build_phase_blocks(
    nlp: NLP,
    geometry: PhaseGeometry,
    lambda_: CFStructure[np.float64],
) -> list[HessianBlock]:
    """Build the blocks for one phase: continuous Hessian terms, then chain-rule terms.

    The block order -- all second-derivative terms in structure order, then all
    first-derivative chain-rule terms -- is part of the pinned NLP interface.
    """
    p = geometry.p
    blocks: list[HessianBlock] = [
        continuous_hessian_block(chs_term, geometry, lambda_)
        for chs_term in nlp.functions.continuous_hessian_structure[p]
    ]
    blocks += [
        block
        for cjs_term in nlp.functions.continuous_jacobian_structure[p]
        if (block := chain_rule_block(cjs_term, geometry, lambda_)) is not None
    ]
    return blocks


def multiplier_scale(
    cf_name: CFName,
    i: int,
    geometry: PhaseGeometry,
    lambda_: CFStructure[np.float64],
) -> Callable[[float], FloatArray]:
    """Return the per-call factor common to every term of one function kind.

    Each application of the chain rule through t = t(tau; t0, tf) contributes a factor
    dt/dtau = (tf - t0)/2; the returned closures carry one such factor for defect and
    integrand terms (whose constraint rows are themselves scaled by the interval length)
    and none for path terms. The caller multiplies by 1/2 for each *additional* time
    derivative in its term.
    """
    p, w = geometry.p, geometry.w
    match cf_name:
        case "f":
            lam_defect = lambda_.phase[p].defect[i]
            return lambda dt: 0.5 * dt * lam_defect
        case "g":
            lam_integral = lambda_.phase[p].integral
            return lambda dt: 0.5 * dt * w * lam_integral[i]
        case "h":
            lam_path = lambda_.phase[p].path
            return lambda _dt: lam_path[i]
        case _:
            assert_never(cf_name)


def continuous_hessian_block(
    chs_term: CHSTerm,
    geometry: PhaseGeometry,
    lambda_: CFStructure[np.float64],
) -> HessianBlock:
    """Build the block for one second-derivative term of a continuous function."""
    (cf_name, i), (cv_name1, j), (cv_name2, k) = chs_term
    p = geometry.p
    n, index = geometry.span(cf_name)
    scale = multiplier_scale(cf_name, i, geometry, lambda_)
    i_t0, i_tf = geometry.i_t0, geometry.i_tf
    t0_view, tf_view = geometry.t0_view, geometry.tf_view
    tau_index = geometry.tau[index]
    n_points = len(geometry.tau)
    points_shape = (n_points,)

    def term_values(context: HessianContext) -> FloatArray:
        dt = tf_view[0] - t0_view[0]
        term = over_points(
            context.continuous_phase(p).hessian[chs_term], points_shape, p, "hessian", chs_term
        )
        return term[index] * scale(dt)

    def mixed_block(var_rows: tuple[int, ...]) -> HessianBlock:
        """Mixed variable/time terms: n entries against t0, then n against tf."""
        weight_t0 = 1 - tau_index
        weight_tf = 1 + tau_index

        def evaluate(context: HessianContext) -> FloatArray:
            term = 0.5 * term_values(context)
            return np.concatenate((weight_t0 * term, weight_tf * term))

        return HessianBlock(2 * var_rows, n * (i_t0,) + n * (i_tf,), evaluate=evaluate)

    if cv_name1 == "t":
        if cv_name2 != "t":
            return mixed_block(geometry.columns(cv_name2, k, index))

        # d2t/d{t0,tf}2 terms: one entry per endpoint pair, each a weighted sum with
        # the endpoint sensitivities dtau/dt0 = -(1 - tau)/2, dtau/dtf = (1 + tau)/2
        weight_t0t0 = (1 - tau_index) ** 2
        weight_t0tf = 1 - tau_index**2
        weight_tftf = (1 + tau_index) ** 2

        def evaluate(context: HessianContext) -> FloatArray:
            term = 0.25 * term_values(context)
            return np.array(
                [
                    (weight_t0t0 * term).sum(),
                    (weight_t0tf * term).sum(),
                    (weight_tftf * term).sum(),
                ],
            )

        return HessianBlock((i_t0, i_t0, i_tf), (i_t0, i_tf, i_tf), evaluate=evaluate)

    if cv_name2 == "t":
        return mixed_block(geometry.columns(cv_name1, j, index))

    # variable/variable terms: n entries, no endpoint sensitivity
    rows = geometry.columns(cv_name1, j, index)
    cols = geometry.columns(cv_name2, k, index)
    return HessianBlock(rows, cols, evaluate=term_values)


def chain_rule_block(
    cjs_term: CJSTerm,
    geometry: PhaseGeometry,
    lambda_: CFStructure[np.float64],
) -> HessianBlock | None:
    """Build the block for one first-derivative chain-rule term, if it has one.

    Only defect and integrand terms depend on (t0, tf) through the interval length;
    path constraints carry no dt/dtau factor, so their Jacobian contributes nothing.
    """
    (cf_name, jj), (cv_name, j) = cjs_term
    p, w = geometry.p, geometry.w
    if cf_name == "h":
        return None

    n, index = geometry.span(cf_name)
    i_t0, i_tf = geometry.i_t0, geometry.i_tf
    tau_index = geometry.tau[index]
    n_points = len(geometry.tau)
    points_shape = (n_points,)

    if cf_name == "f":
        lam_defect = lambda_.phase[p].defect[jj]

        def term_values(context: HessianContext) -> FloatArray:
            jac = over_points(
                context.continuous_phase(p).jacobian[cjs_term],
                points_shape,
                p,
                "jacobian",
                cjs_term,
            )
            return lam_defect * jac[index]

    else:
        lam_integral = lambda_.phase[p].integral

        def term_values(context: HessianContext) -> FloatArray:
            jac = over_points(
                context.continuous_phase(p).jacobian[cjs_term],
                points_shape,
                p,
                "jacobian",
                cjs_term,
            )
            return np.asarray(lam_integral[jj] * w * jac, dtype=np.float64)

    if cv_name == "t":

        def evaluate(context: HessianContext) -> FloatArray:
            term = term_values(context)
            return np.array(
                [
                    -0.5 * ((1 - tau_index) * term).sum(),
                    -0.5 * (tau_index * term).sum(),
                    0.5 * ((1 + tau_index) * term).sum(),
                ],
            )

        return HessianBlock((i_t0, i_t0, i_tf), (i_t0, i_tf, i_tf), evaluate=evaluate)

    # variable terms
    var_cols = geometry.columns(cv_name, j, index)

    if cf_name == "f":

        def evaluate(context: HessianContext) -> FloatArray:
            jac = over_points(
                context.continuous_phase(p).jacobian[cjs_term],
                points_shape,
                p,
                "jacobian",
                cjs_term,
            )
            rhs = 0.5 * jac[index] * lam_defect
            return np.concatenate((-rhs, rhs))

    else:

        def evaluate(context: HessianContext) -> FloatArray:
            jac = over_points(
                context.continuous_phase(p).jacobian[cjs_term],
                points_shape,
                p,
                "jacobian",
                cjs_term,
            )
            rhs = 0.5 * jac[index] * (w * lam_integral[jj])
            return np.concatenate((-rhs, rhs))

    return HessianBlock(n * (i_t0,) + n * (i_tf,), 2 * var_cols, evaluate=evaluate)


def build_objective_block(nlp: NLP, twins: IndexTwins) -> HessianBlock:
    """Build the block for the objective Hessian terms."""
    ohs = nlp.functions.objective_hessian_structure
    rows = tuple(int(twins.dv.var_dict[key1][0]) for key1, _ in ohs)
    cols = tuple(int(twins.dv.var_dict[key2][0]) for _, key2 in ohs)

    def evaluate(context: HessianContext) -> FloatArray:
        return context.objective_factor * np.array(
            [context.objective_hessian[term] for term in ohs],
        )

    return HessianBlock(rows, cols, evaluate=evaluate)


def build_discrete_block(
    nlp: NLP,
    twins: IndexTwins,
    lambda_: CFStructure[np.float64],
) -> HessianBlock:
    """Build the block for the discrete-constraint Hessian terms."""
    dhs = nlp.functions.discrete_hessian_structure
    lam_discrete = lambda_.discrete
    rows = tuple(int(twins.dv.var_dict[key1][0]) for _, key1, _ in dhs)
    cols = tuple(int(twins.dv.var_dict[key2][0]) for _, _, key2 in dhs)

    def evaluate(context: HessianContext) -> FloatArray:
        return np.array(
            [context.discrete_hessian[term] * lam_discrete[term[0]] for term in dhs],
        )

    return HessianBlock(rows, cols, evaluate=evaluate)
