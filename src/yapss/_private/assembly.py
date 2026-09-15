"""

Scaffolding shared by the Jacobian and Hessian assembly plans.

Both derivatives of the transcribed NLP are assembled from a plan: a sequence of
:class:`Block` objects, each owning its (row, col) coordinates and either constant values
or the closure that produces them per evaluation (see the ``jacobian`` and ``hessian``
modules). The two plans read the same phase geometry -- how many entries a continuous
function contributes, which callback points they come from, where the phase endpoints sit
in the decision vector -- and the same integer index structures. That shared part lives
here, built once per NLP, so the two assemblers cannot disagree about the layout they
index into.

"""

# future imports
from __future__ import annotations

# standard imports
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, assert_never

# third party imports
import numpy as np

# package imports
from .layout import problem_layout
from .structure import CFStructure, DVStructure, get_nlp_cf_structure, get_nlp_dv_structure

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable, Sequence

    # third party imports
    from numpy.typing import NDArray

    # package imports
    import yapss

    from .input_args import ContinuousArg
    from .nlp import NLP
    from .types_ import CFName, VectorCVName

    FloatArray = NDArray[np.float64]

__all__ = [
    "Block",
    "ContinuousContext",
    "IndexTwins",
    "PhaseGeometry",
    "index_twins",
    "over_points",
    "phase_geometry",
    "plan_layout",
    "split_constants",
]

C = TypeVar("C")


def over_points(value: Any, n_points: int) -> FloatArray:
    """Return a derivative entry as an array over the evaluation points.

    User callbacks treat states and controls as scalars, so a derivative that happens to
    be constant is naturally written as a scalar -- ``jacobian[key] = 1.0`` -- and must be
    accepted wherever an array would be. Array entries pass through unchanged.
    """
    term = np.asarray(value, dtype=np.float64)
    if term.ndim == 0:
        return np.full(n_points, float(term))
    return term


@dataclass
class ContinuousContext:
    """Per-evaluation inputs common to both plans: the continuous-function outputs."""

    continuous: ContinuousArg[np.float64] | None = None

    def continuous_phase(self, p: int) -> Any:
        """Return the continuous output for phase ``p``, which the evaluator has set."""
        if self.continuous is None:  # pragma: no cover - set whenever phases exist
            msg = "Internal error: continuous output is None"
            raise RuntimeError(msg)
        return self.continuous.phase[p]


@dataclass(frozen=True)
class Block(Generic[C]):
    """One contiguous run of derivative entries.

    ``rows`` and ``cols`` are the NLP-space coordinates of the entries, fixed at build
    time. Exactly one of ``constant`` and ``evaluate`` is set: a constant block's values
    never change, and an evaluated block's closure returns exactly ``len(rows)`` values per
    call. Because coordinates and values come from the same object, the structure and the
    values cannot desynchronize.
    """

    rows: tuple[int, ...]
    cols: tuple[int, ...]
    constant: FloatArray | None = None
    evaluate: Callable[[C], FloatArray] | None = None


def plan_layout(
    blocks: Sequence[Block[C]],
) -> tuple[list[int], list[int], list[slice]]:
    """Concatenate the blocks' coordinates; return rows, cols, and each block's slice."""
    rows: list[int] = []
    cols: list[int] = []
    slices: list[slice] = []
    for block in blocks:
        slices.append(slice(len(rows), len(rows) + len(block.rows)))
        rows += block.rows
        cols += block.cols
    return rows, cols, slices


def split_constants(
    blocks: Sequence[Block[C]],
    slices: Sequence[slice],
    values: FloatArray,
) -> list[tuple[Callable[[C], FloatArray], slice]]:
    """Write the constant blocks into `values` once; return the evaluated blocks' closures.

    Evaluation then touches only the nonlinear blocks, each paired with its slice.
    """
    evaluated: list[tuple[Callable[[C], FloatArray], slice]] = []
    for block, block_slice in zip(blocks, slices, strict=True):
        if block.evaluate is None:
            values[block_slice] = block.constant
        else:
            evaluated.append((block.evaluate, block_slice))
    return evaluated


@dataclass(frozen=True)
class IndexTwins:
    """Integer twins of the NLP decision-variable and constraint vectors.

    Each has the same layout as the float structure it mirrors, with every entry holding
    its own position in the flat vector, so a view gives the NLP indices of that quantity.
    """

    dv: DVStructure[np.int_]
    cf: CFStructure[np.int_]


def index_twins(problem: yapss.Problem) -> IndexTwins:
    """Build the integer twins of the NLP vectors for `problem`."""
    dv: DVStructure[np.int_] = get_nlp_dv_structure(problem, int)
    dv.z[:] = list(range(len(dv.z)))
    cf: CFStructure[np.int_] = get_nlp_cf_structure(problem, int)
    cf.c[:] = list(range(len(cf.c)))
    return IndexTwins(dv, cf)


@dataclass(frozen=True)
class PhaseGeometry:
    """Phase-constant layout facts read by both assembly plans.

    ``nc`` is the number of collocation points, one defect entry each; ``nw`` is the
    number of callback evaluation points, one integrand or path entry each. The two differ
    only under LGL, where interval-boundary points are shared, and ``defect_index`` picks
    the collocation points out of the evaluation points. ``trim_t0`` and ``trim_tf`` count
    the evaluation points at which the endpoint sensitivities vanish -- the last point for
    t0 under LGL, the first point for tf except under LG -- so time terms omit them.
    """

    p: int
    nc: int
    nw: int
    defect_index: NDArray[np.intp]
    trim_t0: int
    trim_tf: int
    tau: FloatArray
    w: FloatArray
    i_t0: int
    i_tf: int
    t0_view: FloatArray
    tf_view: FloatArray
    twins: IndexTwins = field(repr=False)

    def span(self, cf_name: CFName) -> tuple[int, NDArray[np.intp]]:
        """Return the entry count and callback-output index for a function kind."""
        if cf_name == "f":
            return self.nc, self.defect_index
        return self.nw, np.arange(self.nw)

    def columns(
        self,
        cv_name: VectorCVName,
        j: int,
        index: NDArray[np.intp],
    ) -> tuple[int, ...]:
        """Return the NLP indices of one non-time continuous variable over an index span."""
        phase = self.twins.dv.phase[self.p]
        match cv_name:
            case "x":
                return tuple(int(k) for k in phase.x[j][index])
            case "u":
                return tuple(int(k) for k in phase.u[j][index])
            case "s":
                return len(index) * (int(self.twins.dv.s[j]),)
            case _:
                assert_never(cv_name)


def phase_geometry(
    nlp: NLP,
    dv: DVStructure[np.float64],
    twins: IndexTwins,
    p: int,
) -> PhaseGeometry:
    """Collect the phase-constant layout facts for phase ``p``.

    ``dv`` is the float decision-variable structure the plan synchronizes per evaluation;
    the geometry keeps views of its endpoint times.
    """
    layout = problem_layout(nlp.problem)[p]
    return PhaseGeometry(
        p=p,
        nc=layout.n_collocation,
        nw=layout.n_eval,
        defect_index=layout.defect_index,
        trim_t0=layout.trim_t0,
        trim_tf=layout.trim_tf,
        tau=nlp.mesh.tau_u[p],
        w=nlp.mesh.w[p],
        i_t0=int(twins.dv.phase[p].t0[0]),
        i_tf=int(twins.dv.phase[p].tf[0]),
        t0_view=dv.phase[p].t0,
        tf_view=dv.phase[p].tf,
        twins=twins,
    )
