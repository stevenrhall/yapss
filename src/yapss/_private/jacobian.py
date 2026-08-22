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
  entries. ``simplify_jacobian`` folds the triple through a sparse matrix that *sums*
  duplicates, which is exactly what makes the total correct. Entry multiplicity and
  order are therefore part of the interface, pinned by the golden tests.

"""

# future imports
from __future__ import annotations

# standard imports
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# third party imports
import numpy as np

# package imports
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

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

__all__ = ["JacobianBlock", "make_nlp_jacobian"]


@dataclass
class JacobianContext:
    """Per-evaluation inputs shared by all blocks."""

    z: FloatArray = field(default_factory=lambda: np.zeros(0))
    continuous: ContinuousArg[np.float64] | None = None

    def continuous_phase(self, p: int) -> Any:
        """Return the continuous output for phase ``p``, which the evaluator has set."""
        if self.continuous is None:  # pragma: no cover - set whenever phases exist
            msg = "Internal error: continuous output is None"
            raise RuntimeError(msg)
        return self.continuous.phase[p]


@dataclass(frozen=True)
class JacobianBlock:
    """One contiguous run of Jacobian entries.

    ``rows`` and ``cols`` are fixed at build time. Exactly one of ``constant`` and
    ``evaluate`` is set: a constant block's values never change, and an evaluated
    block's closure returns exactly ``len(rows)`` values per call.
    """

    rows: tuple[int, ...]
    cols: tuple[int, ...]
    constant: FloatArray | None = None
    evaluate: Callable[[JacobianContext], FloatArray] | None = None


@dataclass(frozen=True)
class JacobianPhaseGeometry:
    """Phase-constant inputs to the Jacobian block builders."""

    p: int
    nc: int
    nw: int
    # trailing point of the t0 columns and leading point of the tf columns are
    # structural zeros for some spectral methods; these give the per-side trims
    trim_t0: int
    trim_tf: int
    defect_index: NDArray[np.intp]
    tau: FloatArray
    w: FloatArray
    i_t0: int
    i_tf: int
    t0_view: FloatArray
    tf_view: FloatArray
    dv_index: DVStructure[np.int_]
    cf_index: CFStructure[np.int_]


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

    # integer twins of the NLP vectors: values are NLP indices
    dv_index: DVStructure[np.int_] = get_nlp_dv_structure(problem, int)
    dv_index.z[:] = list(range(len(dv_index.z)))
    cf_index: CFStructure[np.int_] = get_nlp_cf_structure(problem, int)
    cf_index.c[:] = list(range(len(cf_index.c)))

    # float structure synchronized per evaluation; blocks capture views into it
    dv: DVStructure[np.float64] = get_nlp_dv_structure(problem, float)

    blocks: list[JacobianBlock] = []
    # the constant collocation terms for every phase come first, then the variable
    # terms for every phase, then the discrete constraints -- the historical order,
    # which the golden tests pin
    for p in range(problem.np):
        blocks += constant_phase_blocks(nlp, dv_index, cf_index, p)
    for p in range(problem.np):
        geometry = phase_geometry(nlp, dv, dv_index, cf_index, p)
        blocks += variable_phase_blocks(nlp, geometry)
    if problem.nd > 0:
        blocks.append(discrete_block(nlp, dv_index, cf_index, eval_discrete_jacobian))

    row: list[int] = []
    col: list[int] = []
    slices: list[slice] = []
    for block in blocks:
        slices.append(slice(len(row), len(row) + len(block.rows)))
        row += block.rows
        col += block.cols
    structure = tuple(row), tuple(col)

    # constant entries are written once; evaluation only touches the nonlinear blocks
    jacobian = np.zeros(len(row), dtype=np.float64)
    evaluated: list[tuple[Callable[[JacobianContext], FloatArray], slice]] = []
    for block, block_slice in zip(blocks, slices, strict=True):
        if block.evaluate is None:
            jacobian[block_slice] = block.constant
        else:
            evaluated.append((block.evaluate, block_slice))

    context = JacobianContext()

    def eval_nlp_jacobian(z: FloatArray) -> FloatArray:
        dv.z[:] = z
        context.z = z
        if problem.np > 0:
            context.continuous = eval_continuous(z, 1)
        for evaluate, block_slice in evaluated:
            jacobian[block_slice] = evaluate(context)
        return jacobian

    return structure, eval_nlp_jacobian


def phase_geometry(
    nlp: NLP,
    dv: DVStructure[np.float64],
    dv_index: DVStructure[np.int_],
    cf_index: CFStructure[np.int_],
    p: int,
) -> JacobianPhaseGeometry:
    """Collect the phase-constant geometry for the block builders."""
    problem = nlp.problem
    spectral_method = problem.spectral_method
    if spectral_method not in ("lg", "lgr", "lgl"):  # pragma: no cover
        raise RuntimeError

    col_points = problem.mesh.phase[p].collocation_points
    nc = sum(col_points)
    if spectral_method == "lgl":
        nw = nc - len(col_points) + 1
        defect_index = np.asarray(cf_index.phase[p].defect_index)
    else:
        nw = nc
        defect_index = np.arange(nc)

    return JacobianPhaseGeometry(
        p=p,
        nc=nc,
        nw=nw,
        trim_t0=1 if spectral_method == "lgl" else 0,
        trim_tf=0 if spectral_method == "lg" else 1,
        defect_index=defect_index,
        tau=nlp.mesh.tau_u[p],
        w=nlp.mesh.w[p],
        i_t0=int(dv_index.phase[p].t0[0]),
        i_tf=int(dv_index.phase[p].tf[0]),
        t0_view=dv.phase[p].t0,
        tf_view=dv.phase[p].tf,
        dv_index=dv_index,
        cf_index=cf_index,
    )


def constant_phase_blocks(
    nlp: NLP,
    dv_index: DVStructure[np.int_],
    cf_index: CFStructure[np.int_],
    p: int,
) -> list[JacobianBlock]:
    """Build the constant collocation blocks for one phase.

    These are the linear terms of the transcription: the differentiation matrix
    acting on the state samples, the Legendre-Gauss ``b`` terms, and the integral
    defect identities.
    """
    problem = nlp.problem
    mesh = nlp.mesh
    dv_phase = dv_index.phase[p]
    cf_phase = cf_index.phase[p]
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

    # d(defect)/dx due to b terms
    if problem.spectral_method == "lg":
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


def variable_phase_blocks(nlp: NLP, geometry: JacobianPhaseGeometry) -> list[JacobianBlock]:
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
    geometry: JacobianPhaseGeometry,
) -> JacobianBlock:
    """Build the block for one first-derivative term of a continuous function."""
    (cf_name, i), (cv_name, j) = cjs_term
    p = geometry.p
    cf_phase = geometry.cf_index.phase[p]
    dv_phase = geometry.dv_index.phase[p]
    w = geometry.w
    defect_index = geometry.defect_index
    nw = geometry.nw
    t0_view, tf_view = geometry.t0_view, geometry.tf_view

    # rows, entry count, and callback-output index for this function kind
    if cf_name == "f":
        term_rows = tuple(int(k) for k in cf_phase.defect[i])
        n, index = geometry.nc, defect_index
    elif cf_name == "g":
        term_rows = nw * (int(cf_phase.integral[i]),)
        n, index = nw, np.arange(nw)
    elif cf_name == "h":
        term_rows = tuple(int(k) for k in cf_phase.path[i])
        n, index = nw, np.arange(nw)
    else:  # pragma: no cover
        msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
        raise ValueError(msg)

    def base_values(context: JacobianContext) -> FloatArray:
        """Return the term's values over its index span, scaled for its row kind.

        Defect and integrand rows are scaled by the interval length dt/dtau =
        (tf - t0)/2; path rows are not. The Jacobian value may be a scalar (a
        constant derivative), so broadcast it over the evaluation points first.
        """
        buffer = np.zeros(nw)
        buffer[:] = context.continuous_phase(p).jacobian[cjs_term]
        if cf_name == "f":
            return np.asarray(0.5 * (tf_view[0] - t0_view[0]) * buffer[index], dtype=np.float64)
        if cf_name == "g":
            return np.asarray(0.5 * (tf_view[0] - t0_view[0]) * w * buffer, dtype=np.float64)
        return buffer

    if cv_name in ("x", "u", "s"):
        if cv_name == "x":
            cols = tuple(int(k) for k in dv_phase.x[j][index])
        elif cv_name == "u":
            cols = tuple(int(k) for k in dv_phase.u[j][index])
        else:
            cols = n * (int(geometry.dv_index.s[j]),)
        return JacobianBlock(rows=term_rows, cols=cols, evaluate=base_values)

    if cv_name != "t":  # pragma: no cover
        msg = f"Invalid continuous Jacobian structure term {cjs_term} in phase {p}"
        raise ValueError(msg)

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


def defect_time_block(i: int, geometry: JacobianPhaseGeometry) -> JacobianBlock:
    """Build the d(defect)/d{t0, tf} block for one state.

    The defect constraints are scaled by the interval length (tf - t0)/2, so each
    picks up -+ f/2 with respect to the endpoints.
    """
    p = geometry.p
    defect_rows = tuple(int(k) for k in geometry.cf_index.phase[p].defect[i])
    defect_index = geometry.defect_index

    def evaluate(context: JacobianContext) -> FloatArray:
        dynamics = np.asarray(context.continuous_phase(p).dynamics[i], dtype=np.float64)
        term = dynamics[defect_index]
        return np.concatenate((-0.5 * term, 0.5 * term))

    return JacobianBlock(
        rows=2 * defect_rows,
        cols=geometry.nc * (geometry.i_t0,) + geometry.nc * (geometry.i_tf,),
        evaluate=evaluate,
    )


def integral_time_block(i: int, geometry: JacobianPhaseGeometry) -> JacobianBlock:
    """Build the d(integral defect)/d{t0, tf} block for one integral."""
    p = geometry.p
    w = geometry.w
    integral_row = int(geometry.cf_index.phase[p].integral[i])

    def evaluate(context: JacobianContext) -> FloatArray:
        integrand = np.asarray(context.continuous_phase(p).integrand[i], dtype=np.float64)
        integral = (w * integrand).sum()
        return np.array([-0.5 * integral, 0.5 * integral])

    return JacobianBlock(
        rows=(integral_row, integral_row),
        cols=(geometry.i_t0, geometry.i_tf),
        evaluate=evaluate,
    )


def duration_block(geometry: JacobianPhaseGeometry) -> JacobianBlock:
    """Build the constant block for the phase duration constraint tf - t0."""
    duration_row = int(geometry.cf_index.phase[geometry.p].duration[0])
    return JacobianBlock(
        rows=(duration_row, duration_row),
        cols=(geometry.i_tf, geometry.i_t0),
        constant=np.array([1.0, -1.0]),
    )


def discrete_block(
    nlp: NLP,
    dv_index: DVStructure[np.int_],
    cf_index: CFStructure[np.int_],
    eval_discrete_jacobian: Callable[[FloatArray], Sequence[np.float64 | float]],
) -> JacobianBlock:
    """Build the block for the discrete-constraint Jacobian terms."""
    djs = nlp.functions.discrete_jacobian_structure
    rows = tuple(int(cf_index.discrete[i]) for i, _ in djs)
    cols = tuple(int(dv_index.var_dict[dv_key][0]) for _, dv_key in djs)

    def evaluate(context: JacobianContext) -> FloatArray:
        return np.asarray(eval_discrete_jacobian(context.z), dtype=np.float64)

    return JacobianBlock(rows=rows, cols=cols, evaluate=evaluate)
