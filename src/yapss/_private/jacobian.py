"""

Assembly of the constraint Jacobian of the transcribed NLP.

Like the Hessian (see the ``hessian`` module), the Jacobian was historically assembled
by two functions that had to agree positionally: a structure builder emitting
(row, col) index pairs together with a vector of constant values, and an evaluator
overwriting the non-constant slots at a manually-advanced cursor, each iterating the
same term lists in the same order. This module replaces both with a single plan of
:class:`JacobianBlock` objects, each owning its index pairs and either a constant
value vector or the closure that produces the matching values per evaluation.

Two Jacobian-specific points:

* **Constant blocks.** The collocation differentiation matrices, the Legendre-Gauss
  ``b`` terms, the integral defect identities, and the phase duration rows are linear
  in the decision variables, so their Jacobian entries never change. They are written
  into the value buffer once at build time; evaluation touches only the nonlinear
  blocks.

* **Duplicate coordinates are load-bearing.** A defect row's derivative with respect
  to a state has both a constant differentiation-matrix entry and a
  ``(tf - t0)/2 * df/dx`` entry at the same (row, col) coordinate, emitted as separate
  entries. The fold applied before handing the structure to Ipopt sums duplicates,
  which is exactly what makes the total correct. Entry multiplicity and order are
  therefore part of the interface, pinned by the golden tests.

"""

# future imports
from __future__ import annotations

# standard imports
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never

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
from .structure import DVStructure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable, Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    from .input_args import ContinuousArg
    from .nlp import NLP
    from .types_ import CJSTerm

    FloatArray = NDArray[np.float64]

__all__ = ["make_nlp_jacobian"]


@dataclass
class JacobianContext(ContinuousContext):
    """Per-evaluation inputs shared by all blocks: the point and the continuous outputs."""

    z: FloatArray = field(default_factory=lambda: np.zeros(0))


JacobianBlock = Block[JacobianContext]


def make_nlp_jacobian(
    nlp: NLP,
    eval_continuous: Callable[[FloatArray, int], ContinuousArg[np.float64]],
    eval_discrete_jacobian: Callable[[FloatArray], Sequence[np.float64 | float]],
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    Callable[[FloatArray], FloatArray],
]:
    """Build the Jacobian structure and evaluator from a single assembly plan.

    Parameters
    ----------
    nlp : NLP
    eval_continuous : Callable
        Evaluator returning the continuous functions and their first derivatives.
    eval_discrete_jacobian : Callable
        Evaluator returning the discrete Jacobian values in structure order.

    Returns
    -------
    tuple
        The ``(rows, cols)`` Jacobian structure, and the ``jacobian(z)`` callback that
        fills the matching values.
    """
    problem = nlp.problem
    twins = index_twins(problem)

    # float structure synchronized per evaluation; blocks capture views into it
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)

    blocks: list[JacobianBlock] = []
    # the constant collocation terms for every phase come first, then the variable
    # terms for every phase, then the discrete constraints -- the historical order,
    # which the golden tests pin
    for p in range(problem.np):
        blocks += constant_phase_blocks(nlp, twins, p)
    for p in range(problem.np):
        blocks += variable_phase_blocks(nlp, phase_geometry(nlp, dv, twins, p))
    if problem.nd > 0:
        blocks.append(discrete_block(nlp, twins, eval_discrete_jacobian))

    row, col, slices = plan_layout(blocks)
    structure = tuple(row), tuple(col)

    # constant entries are written once; evaluation only touches the nonlinear blocks
    jacobian = np.zeros(len(row), dtype=np.float64)
    evaluated = split_constants(blocks, slices, jacobian)

    context = JacobianContext()

    def eval_nlp_jacobian(z: FloatArray) -> FloatArray:
        dv.z[:] = z
        context.z = z
        if problem.np > 0:
            context.continuous = eval_continuous(z, 1)
        for evaluate, block_slice in evaluated:
            jacobian[block_slice] = evaluate(context)
        return jacobian

    # fold the long structure onto unique coordinates; the summing matrix adds
    # coincident entries -- see the module docstring on why that is load-bearing
    folded_rows, folded_cols, summing = fold_structure(row, col)
    if summing is None:
        return structure, eval_nlp_jacobian

    def eval_folded_jacobian(z: FloatArray) -> FloatArray:
        return np.asarray(summing @ eval_nlp_jacobian(z), dtype=np.float64)

    return (folded_rows, folded_cols), eval_folded_jacobian


def constant_phase_blocks(nlp: NLP, twins: IndexTwins, p: int) -> list[JacobianBlock]:
    """Build the constant collocation blocks for one phase.

    These are the linear terms of the transcription: the differentiation matrix
    acting on the state samples, the Legendre-Gauss ``b`` terms, and the integral
    defect identities.
    """
    problem = nlp.problem
    mesh = nlp.mesh
    dv_phase = twins.dv.phase[p]
    cf_phase = twins.cf.phase[p]
    blocks: list[JacobianBlock] = []

    # d(defect)/dx due to differentiation-matrix terms
    for i in range(problem.nx[p]):
        r_, c_ = mesh.d[p].nonzero()
        blocks.append(
            JacobianBlock(
                rows=tuple(int(k) for k in cf_phase.defect[i][r_]),
                cols=tuple(int(k) for k in dv_phase.xa[i][c_]),
                constant=np.asarray((-mesh.d[p]).data, dtype=np.float64),
            ),
        )

    # d(boundary defect)/dx: LG only; b_lg has no rows under the other methods
    for i in range(problem.nx[p]):
        r_, c_ = mesh.b_lg[p].nonzero()
        blocks.append(
            JacobianBlock(
                rows=tuple(int(k) for k in cf_phase.lg_defect[i][r_]),
                cols=tuple(int(k) for k in dv_phase.xa[i][c_]),
                constant=np.asarray(mesh.b_lg[p].data, dtype=np.float64),
            ),
        )

    # d(integral defect)/dq
    nq = problem.nq[p]
    blocks.append(
        JacobianBlock(
            rows=tuple(int(k) for k in cf_phase.integral),
            cols=tuple(int(k) for k in dv_phase.q),
            constant=np.full(nq, -1.0),
        ),
    )
    return blocks


def variable_phase_blocks(nlp: NLP, geometry: PhaseGeometry) -> list[JacobianBlock]:
    """Build the evaluated blocks for one phase, in the historical order."""
    problem = nlp.problem
    p = geometry.p
    blocks: list[JacobianBlock] = [
        continuous_jacobian_block(cjs_term, geometry)
        for cjs_term in nlp.functions.continuous_jacobian_structure[p]
    ]
    blocks += [defect_time_block(i, geometry) for i in range(problem.nx[p])]
    blocks += [integral_time_block(i, geometry) for i in range(problem.nq[p])]
    blocks.append(duration_block(geometry))
    return blocks


def continuous_jacobian_block(
    cjs_term: CJSTerm,
    geometry: PhaseGeometry,
) -> JacobianBlock:
    """Build the block for one first-derivative term of a continuous function."""
    (cf_name, i), (cv_name, j) = cjs_term
    p = geometry.p
    cf_phase = geometry.twins.cf.phase[p]
    w = geometry.w
    nw = geometry.nw
    t0_view, tf_view = geometry.t0_view, geometry.tf_view
    n, index = geometry.span(cf_name)

    # rows for this function kind: one per defect, or the single integral row repeated
    match cf_name:
        case "f":
            term_rows = tuple(int(k) for k in cf_phase.defect[i])
        case "g":
            term_rows = nw * (int(cf_phase.integral[i]),)
        case "h":
            term_rows = tuple(int(k) for k in cf_phase.path[i])
        case _:
            assert_never(cf_name)

    def base_values(context: JacobianContext) -> FloatArray:
        """Return the term's values over its index span, scaled for its row kind.

        Defect and integrand rows are scaled by the interval length dt/dtau =
        (tf - t0)/2; path rows are not. The Jacobian value may be a scalar (a
        constant derivative), so it is broadcast over the evaluation points first.
        """
        values = over_points(context.continuous_phase(p).jacobian[cjs_term], nw)
        if cf_name == "f":
            return np.asarray(0.5 * (tf_view[0] - t0_view[0]) * values[index], dtype=np.float64)
        if cf_name == "g":
            return np.asarray(0.5 * (tf_view[0] - t0_view[0]) * w * values, dtype=np.float64)
        return values

    if cv_name != "t":
        return JacobianBlock(
            rows=term_rows,
            cols=geometry.columns(cv_name, j, index),
            evaluate=base_values,
        )

    # time terms: the values against t0 and tf carry the endpoint sensitivities
    # dtau/dt0 and dtau/dtf; the trailing t0 point and leading tf point are structural
    # zeros for some spectral methods and are trimmed from both indices and values
    n_t0 = n - geometry.trim_t0
    skip_tf = geometry.trim_tf
    tau_index = geometry.tau[index]

    def evaluate(context: JacobianContext) -> FloatArray:
        term = base_values(context)
        return np.concatenate(
            (
                (term * (1 - tau_index) / 2)[:n_t0],
                (term * (1 + tau_index) / 2)[skip_tf:],
            ),
        )

    rows = term_rows[:n_t0] + term_rows[skip_tf:]
    cols = n_t0 * (geometry.i_t0,) + (n - skip_tf) * (geometry.i_tf,)
    return JacobianBlock(rows=rows, cols=cols, evaluate=evaluate)


def defect_time_block(i: int, geometry: PhaseGeometry) -> JacobianBlock:
    """Build the d(defect)/d{t0, tf} block for one state.

    The defect constraints are scaled by the interval length (tf - t0)/2, so each
    picks up -+ f/2 with respect to the endpoints.
    """
    p = geometry.p
    defect_rows = tuple(int(k) for k in geometry.twins.cf.phase[p].defect[i])
    defect_index = geometry.defect_index

    def evaluate(context: JacobianContext) -> FloatArray:
        dynamics = context.continuous_phase(p).dynamics.view(np.ndarray)[i]
        term = dynamics[defect_index]
        return np.concatenate((-0.5 * term, 0.5 * term))

    return JacobianBlock(
        rows=2 * defect_rows,
        cols=geometry.nc * (geometry.i_t0,) + geometry.nc * (geometry.i_tf,),
        evaluate=evaluate,
    )


def integral_time_block(i: int, geometry: PhaseGeometry) -> JacobianBlock:
    """Build the d(integral defect)/d{t0, tf} block for one integral."""
    p = geometry.p
    w = geometry.w
    integral_row = int(geometry.twins.cf.phase[p].integral[i])

    def evaluate(context: JacobianContext) -> FloatArray:
        integrand = context.continuous_phase(p).integrand.view(np.ndarray)[i]
        integral = (w * integrand).sum()
        return np.array([-0.5 * integral, 0.5 * integral])

    return JacobianBlock(
        rows=(integral_row, integral_row),
        cols=(geometry.i_t0, geometry.i_tf),
        evaluate=evaluate,
    )


def duration_block(geometry: PhaseGeometry) -> JacobianBlock:
    """Build the constant block for the phase duration constraint tf - t0."""
    duration_row = int(geometry.twins.cf.phase[geometry.p].duration[0])
    return JacobianBlock(
        rows=(duration_row, duration_row),
        cols=(geometry.i_tf, geometry.i_t0),
        constant=np.array([1.0, -1.0]),
    )


def discrete_block(
    nlp: NLP,
    twins: IndexTwins,
    eval_discrete_jacobian: Callable[[FloatArray], Sequence[np.float64 | float]],
) -> JacobianBlock:
    """Build the block for the discrete-constraint Jacobian terms."""
    djs = nlp.functions.discrete_jacobian_structure
    rows = tuple(int(twins.cf.discrete[i]) for i, _ in djs)
    cols = tuple(int(twins.dv.var_dict[dv_key][0]) for _, dv_key in djs)

    def evaluate(context: JacobianContext) -> FloatArray:
        return np.asarray(eval_discrete_jacobian(context.z), dtype=np.float64)

    return JacobianBlock(rows=rows, cols=cols, evaluate=evaluate)
