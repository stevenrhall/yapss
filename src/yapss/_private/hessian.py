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
Ipopt sums whatever entries
it is given -- it silently produced a wrong Hessian.

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
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

# package imports
from .fold import fold_structure
from .input_args import DiscreteHessianArg, ObjectiveHessianArg
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    # third party imports
    from numpy.typing import NDArray

    from .input_args import ContinuousArg
    from .nlp import NLP
    from .types_ import CHSTerm, CJSTerm

    # package imports

    FloatArray = NDArray[np.float64]

__all__ = ["HessianBlock", "make_nlp_hessian"]


@dataclass
class HessianContext:
    """Per-evaluation inputs shared by all blocks.

    The evaluator fills this once per Hessian call, after synchronizing the decision
    variable and multiplier views; blocks read from it and from the views they
    captured at build time.
    """

    objective_factor: float = 0.0
    continuous: ContinuousArg[np.float64] | None = None
    objective_hessian: dict[Any, Any] = field(default_factory=dict)
    discrete_hessian: dict[Any, Any] = field(default_factory=dict)

    def continuous_phase(self, p: int) -> Any:
        """Return the continuous output for phase ``p``, which the evaluator has set."""
        if self.continuous is None:  # pragma: no cover - set whenever phases exist
            msg = "Internal error: continuous output is None"
            raise RuntimeError(msg)
        return self.continuous.phase[p]


@dataclass(frozen=True)
class HessianBlock:
    """One contiguous run of Hessian entries.

    ``rows`` and ``cols`` are the NLP-space coordinates of the entries, fixed at build
    time; ``evaluate`` returns exactly ``len(rows)`` values for them. Because both come
    from the same object, the structure/value correspondence cannot desynchronize.
    """

    rows: tuple[int, ...]
    cols: tuple[int, ...]
    evaluate: Callable[[HessianContext], FloatArray]


@dataclass(frozen=True)
class PhaseGeometry:
    """Phase-constant inputs to the block builders.

    Everything here is fixed once the mesh and problem dimensions are known: the mesh
    times and quadrature weights, the NLP indices of the phase endpoints, the live
    views through which the endpoints are read per evaluation, and the helpers that
    resolve index spans, multiplier scales, and variable indices for one term.
    """

    p: int
    tau: FloatArray
    w: FloatArray
    i_t0: int
    i_tf: int
    t0_view: FloatArray
    tf_view: FloatArray
    span: Callable[[str], tuple[int, NDArray[np.intp]]]
    multiplier_scale: Callable[[str, int], Callable[[float], FloatArray]]
    variable_indices: Callable[[str, int, NDArray[np.intp]], tuple[int, ...]]


def make_nlp_hessian(
    nlp: NLP,
    eval_continuous: Callable[[FloatArray, int], ContinuousArg[np.float64]],
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
        as built by ``nlp.make_eval_continuous``.

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

    blocks: list[HessianBlock] = []
    for p in range(problem.np):
        blocks += build_phase_blocks(nlp, dv, lambda_, p)
    blocks.append(build_objective_block(nlp))
    if problem.nd > 0:
        blocks.append(build_discrete_block(nlp, lambda_))

    row: list[int] = []
    col: list[int] = []
    slices: list[slice] = []
    for block in blocks:
        slices.append(slice(len(row), len(row) + len(block.rows)))
        row += block.rows
        col += block.cols
    structure = tuple(row), tuple(col)

    hessian = np.zeros(len(row), dtype=np.float64)
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

        objective_input.hessian.clear()
        functions.objective_hessian(objective_input)
        context.objective_hessian = objective_input.hessian

        if problem.nd > 0:
            discrete_input.hessian.clear()
            functions.discrete_hessian(discrete_input)
            context.discrete_hessian = discrete_input.hessian

        for block, block_slice in zip(blocks, slices, strict=True):
            hessian[block_slice] = block.evaluate(context)
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
    dv: DVStructure[np.float64],
    lambda_: CFStructure[np.float64],
    p: int,
) -> list[HessianBlock]:
    """Build the blocks for one phase: continuous Hessian terms, then chain-rule terms.

    The block order -- all second-derivative terms in structure order, then all
    first-derivative chain-rule terms -- is part of the pinned NLP interface.
    """
    problem = nlp.problem
    spectral_method = problem.spectral_method
    if spectral_method not in ("lg", "lgr", "lgl"):  # pragma: no cover
        raise RuntimeError

    # integer twin of `dv`: same layout, values are NLP indices
    dv_index: DVStructure[np.int_] = get_nlp_dv_structure(problem, int)
    dv_index.z[:] = list(range(len(dv_index.z)))
    cf_index: CFStructure[np.int_] = get_nlp_cf_structure(problem, int)

    tau = nlp.mesh.tau_u[p]
    w = nlp.mesh.w[p]
    col_points = problem.mesh.phase[p].collocation_points
    nc = sum(col_points)
    if spectral_method == "lgl":
        nw = nc - len(col_points) + 1
        defect_index = np.asarray(cf_index.phase[p].defect_index)
    else:
        nw = nc
        defect_index = np.arange(nc)

    def span(cf_name: str) -> tuple[int, NDArray[np.intp]]:
        """Return the entry count and callback-output index for a function kind.

        Defect ("f") terms produce one entry per collocation point; integrand and path
        terms produce one per callback evaluation point. The two differ only for the
        LGL method, where interval-boundary points are shared.
        """
        if cf_name == "f":
            return nc, defect_index
        return nw, np.arange(nw)

    def multiplier_scale(cf_name: str, i: int) -> Callable[[float], FloatArray]:
        """Return the per-call factor common to every term of one function kind.

        Each application of the chain rule through t = t(tau; t0, tf) contributes a
        factor dt/dtau = (tf - t0)/2; the returned closures carry one such factor for
        defect and integrand terms (whose constraint rows are themselves scaled by the
        interval length) and none for path terms. The caller multiplies by 1/2 for
        each *additional* time derivative in its term.
        """
        if cf_name == "f":
            lam_defect = lambda_.phase[p].defect[i]
            return lambda dt: 0.5 * dt * lam_defect
        if cf_name == "g":
            lam_integral = lambda_.phase[p].integral
            return lambda dt: 0.5 * dt * w * lam_integral[i]
        if cf_name == "h":
            lam_path = lambda_.phase[p].path
            return lambda _dt: lam_path[i]
        msg = f"Invalid continuous function kind {cf_name!r} in phase {p}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def variable_indices(cv_name: str, j: int, index: NDArray[np.intp]) -> tuple[int, ...]:
        """Return the NLP indices of one continuous variable over an index span."""
        phase_index = dv_index.phase[p]
        if cv_name == "x":
            return tuple(int(k) for k in phase_index.x[j][index])
        if cv_name == "u":
            return tuple(int(k) for k in phase_index.u[j][index])
        if cv_name == "s":
            return len(index) * (int(dv_index.s[j]),)
        msg = f"Invalid continuous variable kind {cv_name!r} in phase {p}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    geometry = PhaseGeometry(
        p=p,
        tau=tau,
        w=w,
        i_t0=int(dv_index.phase[p].t0[0]),
        i_tf=int(dv_index.phase[p].tf[0]),
        # per-call time views; dt = tf - t0 is read through these after the z sync
        t0_view=dv.phase[p].t0,
        tf_view=dv.phase[p].tf,
        span=span,
        multiplier_scale=multiplier_scale,
        variable_indices=variable_indices,
    )

    # second-derivative terms of the continuous functions, then the first-derivative
    # chain-rule terms d(dt/dtau)/d{t0,tf} acting on the Jacobian
    blocks: list[HessianBlock] = [
        continuous_hessian_block(chs_term, geometry)
        for chs_term in nlp.functions.continuous_hessian_structure[p]
    ]
    blocks += [
        block
        for cjs_term in nlp.functions.continuous_jacobian_structure[p]
        if (block := chain_rule_block(cjs_term, geometry, lambda_)) is not None
    ]
    return blocks


def continuous_hessian_block(chs_term: CHSTerm, geometry: PhaseGeometry) -> HessianBlock:
    """Build the block for one second-derivative term of a continuous function."""
    (cf_name, i), (cv_name1, j), (cv_name2, k) = chs_term
    p = geometry.p
    n, index = geometry.span(cf_name)
    scale = geometry.multiplier_scale(cf_name, i)
    i_t0, i_tf = geometry.i_t0, geometry.i_tf
    t0_view, tf_view = geometry.t0_view, geometry.tf_view
    tau_index = geometry.tau[index]

    def term_values(context: HessianContext) -> FloatArray:
        dt = tf_view[0] - t0_view[0]
        term = np.asarray(context.continuous_phase(p).hessian[chs_term], dtype=np.float64)
        return term[index] * scale(dt)

    if cv_name1 == "t" and cv_name2 == "t":
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

        return HessianBlock((i_t0, i_t0, i_tf), (i_t0, i_tf, i_tf), evaluate)

    if cv_name1 == "t" or cv_name2 == "t":
        # mixed variable/time terms: n entries against t0, then n against tf
        cv_name, cv_j = (cv_name1, j) if cv_name2 == "t" else (cv_name2, k)
        var_rows = geometry.variable_indices(cv_name, cv_j, index)
        weight_t0 = 1 - tau_index
        weight_tf = 1 + tau_index

        def evaluate(context: HessianContext) -> FloatArray:
            term = 0.5 * term_values(context)
            return np.concatenate((weight_t0 * term, weight_tf * term))

        return HessianBlock(2 * var_rows, n * (i_t0,) + n * (i_tf,), evaluate)

    # variable/variable terms: n entries, no endpoint sensitivity
    rows = geometry.variable_indices(cv_name1, j, index)
    cols = geometry.variable_indices(cv_name2, k, index)
    return HessianBlock(rows, cols, term_values)


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
    if cf_name not in ("f", "g"):  # pragma: no cover
        msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
        raise ValueError(msg)

    n, index = geometry.span(cf_name)
    i_t0, i_tf = geometry.i_t0, geometry.i_tf
    tau_index = geometry.tau[index]

    if cf_name == "f":
        lam_defect = lambda_.phase[p].defect[jj]

        def term_values(context: HessianContext) -> FloatArray:
            jac = np.asarray(context.continuous_phase(p).jacobian[cjs_term], dtype=np.float64)
            return lam_defect * jac[index]

    else:
        lam_integral = lambda_.phase[p].integral

        def term_values(context: HessianContext) -> FloatArray:
            jac = np.asarray(context.continuous_phase(p).jacobian[cjs_term], dtype=np.float64)
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

        return HessianBlock((i_t0, i_t0, i_tf), (i_t0, i_tf, i_tf), evaluate)

    # variable terms: the Jacobian value may be a scalar (a constant derivative), so
    # broadcast it over the evaluation points before indexing
    var_cols = geometry.variable_indices(cv_name, j, index)
    n_points = len(geometry.tau)

    if cf_name == "f":

        def evaluate(context: HessianContext) -> FloatArray:
            buffer = np.zeros(n_points)
            buffer[:] = 0.5 * context.continuous_phase(p).jacobian[cjs_term]
            rhs = buffer[index] * lam_defect
            return np.concatenate((-rhs, rhs))

    else:

        def evaluate(context: HessianContext) -> FloatArray:
            buffer = np.zeros(n_points)
            buffer[:] = 0.5 * context.continuous_phase(p).jacobian[cjs_term]
            rhs = buffer[index] * (w * lam_integral[jj])
            return np.concatenate((-rhs, rhs))

    return HessianBlock(n * (i_t0,) + n * (i_tf,), 2 * var_cols, evaluate)


def build_objective_block(nlp: NLP) -> HessianBlock:
    """Build the block for the objective Hessian terms."""
    problem = nlp.problem
    dv_index: DVStructure[np.int_] = get_nlp_dv_structure(problem, int)
    dv_index.z[:] = list(range(len(dv_index.z)))
    ohs = nlp.functions.objective_hessian_structure

    rows = tuple(int(dv_index.var_dict[key1][0]) for key1, _ in ohs)
    cols = tuple(int(dv_index.var_dict[key2][0]) for _, key2 in ohs)

    def evaluate(context: HessianContext) -> FloatArray:
        return context.objective_factor * np.array(
            [context.objective_hessian[term] for term in ohs],
        )

    return HessianBlock(rows, cols, evaluate)


def build_discrete_block(nlp: NLP, lambda_: CFStructure[np.float64]) -> HessianBlock:
    """Build the block for the discrete-constraint Hessian terms."""
    problem = nlp.problem
    dv_index: DVStructure[np.int_] = get_nlp_dv_structure(problem, int)
    dv_index.z[:] = list(range(len(dv_index.z)))
    dhs = nlp.functions.discrete_hessian_structure
    lam_discrete = lambda_.discrete

    rows = tuple(int(dv_index.var_dict[key1][0]) for _, key1, _ in dhs)
    cols = tuple(int(dv_index.var_dict[key2][0]) for _, _, key2 in dhs)

    def evaluate(context: HessianContext) -> FloatArray:
        return np.array(
            [context.discrete_hessian[term] * lam_discrete[term[0]] for term in dhs],
        )

    return HessianBlock(rows, cols, evaluate)
