"""

Folding of sparse derivative structures.

The Hessian and Jacobian assembly plans emit one entry per mathematical term, so the
same (row, col) coordinate can appear several times: a defect row's state derivative
has both a constant differentiation-matrix entry and a dynamics-derivative entry, and
low-order terms repeat coordinates within a phase. Ipopt is happier with a compact
structure, so before handing the triple over, the structure is deduplicated and a
sparse summing matrix is built that folds the assembled "long" value vector onto the
compact structure -- summing duplicates, which is what makes the totals correct.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING

# third party imports
import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["fold_structure"]


def fold_structure(
    rows: Sequence[int],
    cols: Sequence[int],
    *,
    lower_triangular: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...], csr_matrix | None]:
    """Deduplicate a sparse structure and build the matching summing matrix.

    Parameters
    ----------
    rows, cols : Sequence[int]
        Coordinates of the long structure, one entry per assembled term.
    lower_triangular : bool
        Mirror each deduplicated coordinate into the lower triangle, for the Hessian,
        which Ipopt treats as one triangle of a symmetric matrix. The mirroring is
        applied after deduplication, entry by entry: a pair present in both triangles
        of the long structure remains two entries, which Ipopt sums.

    Returns
    -------
    tuple
        The deduplicated ``(rows, cols)`` structure and a sparse matrix ``summing``
        such that ``summing @ long_values`` gives the values for it, or ``None`` when
        the structure is empty and there is nothing to fold.
    """
    n = len(rows)
    if n == 0:
        return tuple(rows), tuple(cols), None

    unique = sorted(set(zip(rows, cols, strict=True)))
    position = {pair: k for k, pair in enumerate(unique)}
    summing = csr_matrix(
        (
            np.ones(n),
            ([position[pair] for pair in zip(rows, cols, strict=True)], range(n)),
        ),
    )

    out_rows = tuple(pair[0] for pair in unique)
    out_cols = tuple(pair[1] for pair in unique)
    if lower_triangular:
        mirrored = [(r, c) if r >= c else (c, r) for r, c in unique]
        out_rows, out_cols = tuple(zip(*mirrored, strict=True))
    return out_rows, out_cols, summing
