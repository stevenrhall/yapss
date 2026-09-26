"""

The solution, read under the names the problem declared.

Every quantity of a phase is a vector with that phase's own field names, so a state is
``ps.state.h`` wherever it is reached, and a helper written against a callback's endpoint
values also accepts a solution's.

A solution is data and holds nothing but data. It keeps no callbacks, no reference to the
problem, and not even the classes the problem was declared with: a declaration may be made
inside a function, and a class made there cannot be pickled by reference. What it keeps of each
declaration is the *shape* -- its role, its name, and its fields with their sizes, which are
plain data -- and it answers under those names through a declaration rebuilt from the shape.
That costs nothing a user can see, since a vector class holds no behavior, except that
``isinstance(ps.state, State)`` is false. What it buys is that every solution pickles, however
its problem was declared.

"""

from __future__ import annotations

from functools import cache
from itertools import pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, overload

import numpy as np

# `TypeVar` from typing_extensions for PEP 696 defaults, as in `args` and `declare`.
from typing_extensions import TypeVar

from yapss._backend.layout import problem_layout
from yapss._backend.structure import get_nlp_cf_structure, get_nlp_dv_structure

from .args import C_co, D_co, EndpointValues, I_co, P_co, PR_co, S_co
from .containers import suggest
from .kinds import ReadOnlyRows
from .vector import ROLES, Vector, role_of, scalar, vector

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from yapss._backend.ipopt_status import IpoptStatus
    from yapss._backend.structure import CFStructure, DVStructure

    from .mesh import Mesh
    from .spec import PhaseSpec, ProblemSpec
    from .vector import Control, Integral, Path, State

__all__ = [
    "ConstraintIndex",
    "Convergence",
    "EndpointMultiplier",
    "Jacobian",
    "NLPIndex",
    "NLPRecord",
    "NLPScale",
    "PhaseIndex",
    "PhaseMultiplier",
    "PhaseNLP",
    "PhasePoint",
    "PhaseSolution",
    "ProblemConstraintIndex",
    "ProblemMultiplier",
    "ProblemVariableIndex",
    "Solution",
    "VariableIndex",
]

Shape = tuple[str, str, tuple[tuple[str, int | None], ...]]
"""What a solution keeps of a declaration: its role, its name, and its fields with their sizes."""


def shape_of(declaration: type[Vector]) -> Shape:
    """Return the shape of `declaration`, which is plain data and pickles by value."""
    fields = tuple((name, declaration._meta[name].size) for name in declaration._fields)
    return (role_of(declaration) or "", declaration.__name__, fields)


def _reduce_vector(self: Vector) -> tuple[Any, ...]:
    """Pickle a solution vector as its shape, its label and its rows."""
    shape: Shape = getattr(type(self), "_shape")  # noqa: B009
    return (_rebuild_vector, (shape, self._label, self._source))


@cache
def _declaration(shape: Shape) -> type[Vector]:
    """Return a declaration with the fields `shape` names, rebuilt from the shape alone.

    It has the user's class name, so a message reads as it would have, and it answers every read
    the user's class would. It is made once per shape and cached, and it pickles its instances
    as their shape, so nothing about it needs to be importable.
    """
    role, name, fields = shape
    base = next((cls for cls in ROLES if cls._role == role), Vector)
    namespace: dict[str, Any] = {
        "__module__": __name__,
        "__qualname__": name,
        "_shape": shape,
        "__reduce__": _reduce_vector,
    }
    for field, size in fields:
        namespace[field] = scalar() if size is None else vector(size)
    declaration: type[Vector] = type(name, (base,), namespace)
    return declaration


_NAMES_FIXED = "a solution's names are fixed, and its arrays can be edited in place"
"""Why a solution refuses to assign or delete a name, and what it allows instead (spec 8)."""


class SolutionRows(ReadOnlyRows):
    """A solution's rows.

    Each name is one solved quantity, so it cannot be rebound; the array it holds is the
    user's, and can be edited in place.
    """

    @classmethod
    def refusal(cls, label: str, name: str | None, verb: str = "assigned") -> str:
        """Return the message refusing to rebind `name`, or, for None, the rows by position."""
        what = f"{label}'s rows" if name is None else f"{label} '{name}'"
        return f"{what} cannot be {verb}; {_NAMES_FIXED}"


def _rebuild_vector(shape: Shape, label: str, rows: Any) -> Any:
    """Return a solution vector of `shape` holding `rows`. Also what unpickling calls."""
    obj = _declaration(shape)._new(SolutionRows, label, None)
    obj._fill(rows)
    return obj


def _vector(declaration: type[Vector], rows: Any, label: str) -> Any:
    """Return a solution vector with the fields of `declaration`, holding `rows`."""
    return _rebuild_vector(shape_of(declaration), label, rows)


# -- one grid per phase ------------------------------------------------------------------------
#
# The state lives on the state points and everything evaluated per point -- control, dynamics,
# path, integrand, costate, Hamiltonian, the per-point multipliers -- on the collocation points.
# Under LGL the two coincide. Under LGR the collocation points miss the phase's final point, and
# under LG they miss both ends of every segment. Every per-point quantity is reported on the
# state points anyway, so that the whole interval plots under every method, and the points the
# method produced no value at are filled from the method's own polynomial: barycentric Lagrange
# interpolation on the segment's collocation points, evaluated at the segment's end, and at an
# LG join between two segments the average of the two segments' values. The filled values are
# the polynomial where the method imposed nothing -- a filled control can lie outside its
# bounds -- and `ps.collocated` marks the points that are the solver's own.


def _barycentric(nodes: Any, values: Any, at: float) -> Any:
    """Evaluate at `at` the polynomial through (`nodes`, `values`), along the last axis.

    The nodes are mapped onto [-1, 1] first, which keeps the weights well scaled whatever the
    segment's length. `at` may lie outside the nodes' span: evaluating at a segment's end is
    extrapolation, and the barycentric form is exact there too.
    """
    lo, hi = nodes[0], nodes[-1]
    x = (2.0 * (nodes - lo) / (hi - lo) - 1.0) if hi > lo else nodes - lo
    t = (2.0 * (at - lo) / (hi - lo) - 1.0) if hi > lo else at - lo
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    weights = 1.0 / diff.prod(axis=1)
    d = t - x
    exact = np.flatnonzero(d == 0.0)
    if exact.size:
        return values[..., exact[0]]
    terms = weights / d
    return (values * terms).sum(axis=-1) / terms.sum()


class _Grid:
    """Where a phase's collocation points sit among its state points, and how to fill the rest.

    Parameters
    ----------
    method : str
        The spectral method.
    points : tuple of int
        The collocation points in each segment.
    time, time_c : numpy.ndarray
        The state points and the collocation points, both in time order.
    """

    def __init__(self, method: str, points: tuple[int, ...], time: Any, time_c: Any) -> None:
        self.time = time
        self.time_c = time_c
        n = len(time)
        mask = np.ones(n, dtype=bool)
        # each missing state point, with the segments whose polynomials meet there
        self.missing: list[tuple[int, list[int]]] = []
        starts = np.concatenate(([0], np.cumsum(points)))
        self.segments = [slice(int(a), int(b)) for a, b in pairwise(starts)]
        if method == "lgr":
            mask[-1] = False
            self.missing.append((n - 1, [len(points) - 1]))
        elif method == "lg":
            # state points: segment start, its collocation points, next start, ..., final point
            position = 0
            for k, count in enumerate(points):
                mask[position] = False
                self.missing.append((position, [k] if k == 0 else [k - 1, k]))
                position += count + 1
            mask[-1] = False
            self.missing.append((n - 1, [len(points) - 1]))
        self.collocated = mask

    def fill(self, values: Any) -> Any:
        """Return `values`, given on the collocation points, on every state point."""
        values = np.asarray(values)
        if values.shape[-1] == len(self.time):
            return values
        out = np.empty((*values.shape[:-1], len(self.time)), dtype=values.dtype)
        out[..., self.collocated] = values
        if values.shape[-1] == 0 or self.time[-1] == self.time[0]:
            # a zero-duration phase has no polynomial to evaluate: its points coincide
            out[..., ~self.collocated] = np.nan
            return out
        for position, segments in self.missing:
            at = self.time[position]
            out[..., position] = np.mean(
                [
                    _barycentric(self.time_c[self.segments[k]], values[..., self.segments[k]], at)
                    for k in segments
                ],
                axis=0,
            )
        return out


class _Fixed:
    """A value read at an endpoint of a solution, which does not change, unlike a callback's."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __call__(self) -> Any:
        return self.value

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Fixed, (self.value,))


def _endpoint(phase: PhaseSpec, rows: Any, independent: Any, label: str) -> EndpointValues:
    """Return one end of a phase, read as an endpoint callback reads it."""
    return EndpointValues(
        _vector(phase.state, rows, f"{label} state"), _Fixed(independent), phase.independent
    )


class _Record:
    """Base of the groups a solution holds: named slots, fixed, pickled as their values."""

    __slots__: tuple[str, ...] = ()
    _label = "the solution"

    def __init__(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def _set(self) -> dict[str, Any]:
        """Return the slots that hold a value: a group may leave one unset under a method."""
        values = {}
        for name in self.__slots__:
            try:
                values[name] = object.__getattribute__(self, name)
            except AttributeError:
                continue
        return values

    def _names(self) -> tuple[str, ...]:
        return tuple(n for n in self._set() if not n.startswith("_"))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as the values, which are data all the way down."""
        return (type(self), (self._set(),))

    # Hidden from type checkers: one that sees a reader answering any name stops
    # reporting misspellings. The names are declared for them instead.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Refuse an unknown name with a suggestion."""
            if name.startswith("_"):
                raise AttributeError(name)
            msg = f"{self._label} has no '{name}'.{suggest(name, self._names())}"
            raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: each name is one solved quantity."""
        del value
        msg = f"'{name}' cannot be assigned; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: each name is one solved quantity."""
        msg = f"'{name}' cannot be deleted; {_NAMES_FIXED}"
        raise AttributeError(msg)


_STATE_BOUNDS_OWED = (
    "A state's bound multiplier has to be divided by the quadrature weight and, at a collocated "
    "endpoint, folded into the costate, and that is still to be done."
)


class EndpointMultiplier(_Record):
    """The multipliers at one end of a phase, in the namespace `ps.initial` and `ps.final` use.

    The independent variable's is reported, under the phase's name for it. A state's -- the
    multiplier of its initial or final condition -- is owed, and asking for one says so rather
    than calling the name unknown.
    """

    __slots__ = ("_fields", "_independent", "_value")
    _label = "the endpoint multipliers"

    # Hidden from type checkers: one that sees a reader answering any name stops
    # reporting misspellings. The names are declared for them instead.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Return the independent variable's multiplier, or explain a state's absence."""
            if name.startswith("_"):
                raise AttributeError(name)
            independent = object.__getattribute__(self, "_independent")
            if name == independent:
                return object.__getattribute__(self, "_value")
            fields: tuple[str, ...] = object.__getattribute__(self, "_fields")
            if name in fields:
                msg = f"the multiplier of '{name}' at this endpoint is not reported yet. " + (
                    _STATE_BOUNDS_OWED
                )
                raise AttributeError(msg)
            msg = f"{self._label} have no '{name}'.{suggest(name, (*fields, independent))}"
            raise AttributeError(msg)


class PhaseMultiplier(_Record, Generic[S_co, C_co, P_co, I_co]):
    """A phase's multipliers, in the shapes of what they belong to: ``ps.multiplier``.

    Attributes
    ----------
    dynamics : Vector
        The costate: the multiplier of the dynamics, named by the state's fields. The same
        object as ``ps.costate``.
    control : Vector
        The multipliers of the controls' bounds, densities in time.
    path : Vector
        The multipliers of the path constraints, densities in time.
    integral : Vector
        One multiplier per integral.
    initial, final : EndpointMultiplier
        The multipliers at each end of the phase.
    duration : float
        The multiplier of the phase's extent.
    """

    __slots__ = ("control", "duration", "dynamics", "final", "initial", "integral", "path")
    _label = "the phase multipliers"

    if TYPE_CHECKING:
        dynamics: S_co
        # Reserved and owed: reading it raises, saying so. Declared so that it checks when it
        # is delivered, and a stub may be permissive.
        state: S_co
        control: C_co
        path: P_co
        integral: I_co
        initial: Any
        final: Any
        duration: float

    # Hidden from type checkers: one that sees a reader answering any name stops
    # reporting misspellings. The names are declared for them instead.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Explain the owed state-bound multipliers, or refuse an unknown name."""
            if name == "state":
                msg = (
                    "the multipliers of the state's bounds are not reported yet. "
                    f"{_STATE_BOUNDS_OWED}"
                )
                raise AttributeError(msg)
            return super().__getattr__(name)


class ProblemMultiplier(_Record, Generic[D_co, PR_co]):
    """The problem's multipliers: ``solution.multiplier``.

    Attributes
    ----------
    parameter : Vector
        The multipliers of the parameters' bounds.
    discrete : Vector
        The multipliers of the discrete constraints.
    """

    __slots__ = ("discrete", "parameter")
    _label = "the problem multipliers"

    if TYPE_CHECKING:
        parameter: PR_co
        discrete: D_co


# -- the solver's record ------------------------------------------------------------------------
#
# What Ipopt saw and returned, as flat vectors in Ipopt's order, and beside them the positions of
# every variable and constraint under the user's names. The numbers are held once, in the flat
# vectors; a quantity in the solver's terms is one indexing expression, ``nlp.g[con.dynamics.h]``,
# and since indexing with an integer array copies, nothing read that way shares memory with the
# record.


class _MethodOnly(_Record):
    """A group with a slot that exists under one spectral method only, and says so elsewhere."""

    _method_only: ClassVar[Mapping[str, str]] = MappingProxyType({})

    # Hidden from type checkers: one that sees a reader answering any name stops
    # reporting misspellings. The names are declared for them instead.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Explain a slot this method does not have, or refuse an unknown name."""
            if name in self._method_only:
                msg = f"'{name}' exists only under {self._method_only[name]}"
                raise AttributeError(msg)
            return super().__getattr__(name)


class VariableIndex(_MethodOnly, Generic[S_co, C_co, I_co]):
    """The positions of a phase's decision variables in any vector of Ipopt's length ``n``.

    Each is an integer array shaped like the quantity it locates, so ``nlp.x[var.state.r]`` has
    the shape of ``ps.state.r``.

    Attributes
    ----------
    state : Vector
        Every stored value of each state, in time order, on the phase's points.
    zero_mode : Vector
        One per segment for each state; LGL only.
    control : Vector
        Each control, on the collocated points.
    integral : Vector
        One per integral.
    initial, final : EndpointValues
        The state at each end, and the phase's independent variable under its own name. A
        state's entries repeat the first and last of `state`.
    """

    __slots__ = ("control", "final", "initial", "integral", "state", "zero_mode")
    _label = "the variable index"
    _method_only = MappingProxyType({"zero_mode": "LGL"})

    if TYPE_CHECKING:
        state: S_co
        zero_mode: S_co
        control: C_co
        integral: I_co
        initial: Any
        final: Any


class ConstraintIndex(_MethodOnly, Generic[S_co, P_co, I_co]):
    """The positions of a phase's constraint rows in any vector of Ipopt's length ``m``.

    Attributes
    ----------
    dynamics : Vector
        The defect rows of each state, segment by segment. Under LGR and LG they are on the
        collocated points; under LGL each segment is collocated on its own, so a boundary
        between two segments has a row from each.
    continuity : Vector
        One row per segment for each state; LG only.
    path : Vector
        Each path constraint, on the collocated points.
    integral : Vector
        One row per integral.
    duration : int
        The row bounding the phase's extent.
    """

    __slots__ = ("continuity", "duration", "dynamics", "integral", "path")
    _label = "the constraint index"
    _method_only = MappingProxyType({"continuity": "LG"})

    if TYPE_CHECKING:
        dynamics: S_co
        continuity: S_co
        path: P_co
        integral: I_co
        duration: int


class PhaseIndex(_Record, Generic[S_co, C_co, P_co, I_co]):
    """A phase's positions in the solver's record: ``ps.nlp.index``.

    Attributes
    ----------
    variable : VariableIndex
        Positions in ``x`` and every vector of its length.
    constraint : ConstraintIndex
        Positions in ``g`` and every vector of its length.
    """

    __slots__ = ("constraint", "variable")
    _label = "the phase index"

    if TYPE_CHECKING:
        variable: VariableIndex[S_co, C_co, I_co]
        constraint: ConstraintIndex[S_co, P_co, I_co]


class PhasePoint(_MethodOnly):
    """The point each row of a group sits at, as an index into the phase's points.

    Given only for the rows whose points cannot be read off ``ps.collocated``: every other
    per-point entry of the index trees is on ``ps.<time>[ps.collocated]``, or, for the state,
    on all of ``ps.<time>``.

    Attributes
    ----------
    dynamics : numpy.ndarray
        The point each defect row is collocated at. Under LGR and LG these are the collocated
        points; under LGL a boundary between two segments appears twice, once per segment.
        The same for every state.
    continuity : numpy.ndarray
        The segment end each continuity row ties the state at; LG only.
    """

    __slots__ = ("continuity", "dynamics")
    _label = "the phase's row points"
    _method_only = MappingProxyType({"continuity": "LG"})

    if TYPE_CHECKING:
        dynamics: NDArray[np.intp]
        continuity: NDArray[np.intp]


class PhaseNLP(_Record, Generic[S_co, C_co, P_co, I_co]):
    """A phase's part of the solver's record: ``ps.nlp``.

    Attributes
    ----------
    index : PhaseIndex
        The positions of the phase's variables and constraints in ``solution.nlp``.
    point : PhasePoint
        The point each defect and continuity row sits at.
    """

    __slots__ = ("index", "point")
    _label = "the phase's solver record"

    if TYPE_CHECKING:
        index: PhaseIndex[S_co, C_co, P_co, I_co]
        point: PhasePoint


class ProblemVariableIndex(_Record, Generic[PR_co]):
    """The positions of the parameters in any vector of length ``n``.

    Attributes
    ----------
    parameter : Vector
    """

    __slots__ = ("parameter",)
    _label = "the variable index"

    if TYPE_CHECKING:
        parameter: PR_co


class ProblemConstraintIndex(_Record, Generic[D_co]):
    """The positions of the discrete constraints in any vector of length ``m``.

    Attributes
    ----------
    discrete : Vector
    """

    __slots__ = ("discrete",)
    _label = "the constraint index"

    if TYPE_CHECKING:
        discrete: D_co


class NLPIndex(_Record, Generic[D_co, PR_co]):
    """The problem's positions in the solver's record: ``solution.nlp.index``.

    Attributes
    ----------
    variable : ProblemVariableIndex
    constraint : ProblemConstraintIndex
    """

    __slots__ = ("constraint", "variable")
    _label = "the problem index"

    if TYPE_CHECKING:
        variable: ProblemVariableIndex[PR_co]
        constraint: ProblemConstraintIndex[D_co]


class Jacobian(_Record):
    """The constraints' Jacobian at the returned point, in the structure Ipopt was given.

    Attributes
    ----------
    row, col : numpy.ndarray
        Each entry's position in ``g`` and in ``x``, in the positions the index trees give.
    value : numpy.ndarray
        Each entry's value.
    """

    __slots__ = ("col", "row", "value")
    _label = "the Jacobian"

    if TYPE_CHECKING:
        row: NDArray[np.intp]
        col: NDArray[np.intp]
        value: NDArray[np.float64]


class NLPScale(_Record):
    """The scaling YAPSS gave Ipopt.

    Attributes
    ----------
    objective : float
        The factor Ipopt multiplied the objective by. Its sign is the problem's sense: Ipopt
        minimizes, so a maximized objective has a negative factor.
    x, g : numpy.ndarray
        The factors for each variable and each constraint row.
    """

    __slots__ = ("g", "objective", "x")
    _label = "the scaling"

    if TYPE_CHECKING:
        objective: float
        x: NDArray[np.float64]
        g: NDArray[np.float64]


class Convergence(_Record):
    """Ipopt's own measures of convergence at the returned point, in its scaled terms.

    They are what Ipopt's tolerance is tested against, and the scaled column of its final
    statistics. NaN where Ipopt had none to give, as when it stopped in its restoration phase.

    Attributes
    ----------
    inf_pr : float
        Primal infeasibility: the largest violation of a constraint or a bound.
    inf_du : float
        Dual infeasibility: the largest entry of the Lagrangian's gradient.
    complementarity : float
        The largest violation of complementary slackness.
    iterations : int
        The number of iterations Ipopt took.
    """

    __slots__ = ("complementarity", "inf_du", "inf_pr", "iterations")
    _label = "the convergence measures"

    if TYPE_CHECKING:
        inf_pr: float
        inf_du: float
        complementarity: float
        iterations: int


class NLPRecord(_Record, Generic[D_co, PR_co]):
    """What Ipopt saw and what it returned: ``solution.nlp``.

    The vectors are in the order Ipopt saw them, which is part of what produced the result and
    is valid for the YAPSS `version` recorded. A position means something through `index` and
    ``ps.nlp.index``, and not otherwise. The vectors are unscaled -- the problem as posed -- with
    the scaling beside them.

    Attributes
    ----------
    version : str
        The YAPSS version the order is valid for.
    status : int
        Ipopt's return code.
    objective : float
        The objective at `x`, as the problem states it.
    x_L, x_U, z0 : numpy.ndarray
        The variables' bounds and the starting point.
    x, mult_x_L, mult_x_U : numpy.ndarray
        The variables and their bound multipliers at the returned point.
    grad_f : numpy.ndarray
        The objective's gradient at `x`.
    g_L, g_U, g, mult_g : numpy.ndarray
        The constraints' bounds, values and multipliers.
    jac_g : Jacobian
        The constraints' Jacobian at `x`.
    scale : NLPScale
        The scaling YAPSS applied.
    convergence : Convergence
        Ipopt's final measures of convergence.
    index : NLPIndex
        The positions of the parameters and the discrete constraints.
    """

    __slots__ = (
        "convergence",
        "g",
        "g_L",
        "g_U",
        "grad_f",
        "index",
        "jac_g",
        "mult_g",
        "mult_x_L",
        "mult_x_U",
        "objective",
        "scale",
        "status",
        "version",
        "x",
        "x_L",
        "x_U",
        "z0",
    )
    _label = "the solver's record"

    if TYPE_CHECKING:
        version: str
        status: int
        objective: float
        x_L: NDArray[np.float64]  # noqa: N815 -- Ipopt's names
        x_U: NDArray[np.float64]  # noqa: N815
        z0: NDArray[np.float64]
        x: NDArray[np.float64]
        mult_x_L: NDArray[np.float64]  # noqa: N815
        mult_x_U: NDArray[np.float64]  # noqa: N815
        grad_f: NDArray[np.float64]
        g_L: NDArray[np.float64]  # noqa: N815
        g_U: NDArray[np.float64]  # noqa: N815
        g: NDArray[np.float64]
        mult_g: NDArray[np.float64]
        jac_g: Jacobian
        scale: NLPScale
        convergence: Convergence
        index: NLPIndex[D_co, PR_co]

    @classmethod
    def _from(cls, info: Any, index: NLPIndex[Any, Any]) -> NLPRecord[Any, Any]:
        """Return the record from the back end's `NLPInfo`, copying every array."""
        from yapss import __version__  # noqa: PLC0415 -- the package imports this module

        def own(name: str) -> Any:
            return np.array(getattr(info, name))

        return cls(
            {
                "version": __version__,
                "status": int(info.ipopt_status),
                "objective": float(info.obj_val),
                **{
                    name: own(name)
                    for name in (
                        "x_L",
                        "x_U",
                        "z0",
                        "x",
                        "mult_x_L",
                        "mult_x_U",
                        "grad_f",
                        "g_L",
                        "g_U",
                        "g",
                        "mult_g",
                    )
                },
                "jac_g": Jacobian(
                    {"row": own("jac_g_row"), "col": own("jac_g_col"), "value": own("jac_g")}
                ),
                "scale": NLPScale(
                    {
                        "objective": float(info.obj_scaling),
                        "x": own("x_scaling"),
                        "g": own("g_scaling"),
                    }
                ),
                "convergence": Convergence(
                    {
                        "inf_pr": float(info.inf_pr),
                        "inf_du": float(info.inf_du),
                        "complementarity": float(info.complementarity),
                        "iterations": int(info.iterations),
                    }
                ),
                "index": index,
            }
        )


def _positions_of(transcription: Any) -> tuple[Any, Any, Any]:
    """Return the back end's structures over the flat vectors, holding each entry's position."""
    dv: DVStructure[np.intp] = get_nlp_dv_structure(transcription, np.intp)
    dv.z[:] = np.arange(dv.z.size)
    cf: CFStructure[np.intp] = get_nlp_cf_structure(transcription, np.intp)
    cf.c[:] = np.arange(cf.c.size)
    return dv, cf, problem_layout(transcription)


def _positions(views: Any, npoints: int) -> Any:
    """Stack a structure's per-row views of positions into a (rows, npoints) integer array."""
    return np.array([np.asarray(v) for v in views], dtype=np.intp).reshape(len(views), npoints)


def _phase_nlp(phase: PhaseSpec, dv_phase: Any, cf_phase: Any, layout: Any) -> PhaseNLP:
    """Return a phase's positions in the solver's record, under its declared names.

    `layout` is the phase's `PhaseLayout`, which fixes every size here, and whose time order
    puts LG's stored state, collocation values first, back in time order.
    """
    label = f"phase '{phase.name}' index"
    n_eval = layout.n_eval
    state = _positions([x[layout.time_order] for x in dv_phase.x], layout.n_time)
    variable: dict[str, Any] = {
        "state": _vector(phase.state, state, f"{label} state"),
        "control": _vector(phase.control, _positions(dv_phase.u, n_eval), f"{label} control"),
        "integral": _vector(
            phase.integral, np.array(dv_phase.q, dtype=np.intp), f"{label} integral"
        ),
        "initial": _endpoint(phase, state[:, 0], int(dv_phase.t0[0]), f"{label} initial"),
        "final": _endpoint(phase, state[:, -1], int(dv_phase.tf[0]), f"{label} final"),
    }
    constraint: dict[str, Any] = {
        "dynamics": _vector(
            phase.state, _positions(cf_phase.defect, layout.n_collocation), f"{label} dynamics"
        ),
        "path": _vector(phase.path, _positions(cf_phase.path, n_eval), f"{label} path"),
        "integral": _vector(
            phase.integral, np.array(cf_phase.integral, dtype=np.intp), f"{label} integral"
        ),
        "duration": int(cf_phase.duration[0]),
    }
    n_zero_modes = layout.n_state_storage - layout.n_time
    if n_zero_modes:
        variable["zero_mode"] = _vector(
            phase.state, _positions(dv_phase.xs, n_zero_modes), f"{label} zero_mode"
        )
    if layout.n_boundary_defect:
        constraint["continuity"] = _vector(
            phase.state,
            _positions(cf_phase.lg_defect, layout.n_boundary_defect),
            f"{label} continuity",
        )
    index = PhaseIndex(
        {"variable": VariableIndex(variable), "constraint": ConstraintIndex(constraint)}
    )
    return PhaseNLP({"index": index, "point": _row_points(layout)})


def _row_points(layout: Any) -> PhasePoint:
    """Return the point each defect and continuity row sits at, as indices into the points.

    The state is stored at the evaluation points first, so time point ``t`` is an evaluation
    point exactly when ``time_order[t] < n_eval``, and evaluation point ``e`` is the time point
    whose ``time_order`` is ``e``. LG's continuity rows determine the state at each segment's
    end: the points it does not collocate, after the phase's start.
    """
    time_of = np.empty(layout.n_time, dtype=np.intp)
    time_of[layout.time_order] = np.arange(layout.n_time)
    points: dict[str, Any] = {"dynamics": time_of[layout.defect_index]}
    if layout.n_boundary_defect:
        points["continuity"] = np.flatnonzero(layout.time_order >= layout.n_eval)[1:]
    return PhasePoint(points)


_S_co = TypeVar("_S_co", bound="State", covariant=True)
_C_co = TypeVar("_C_co", bound="Control", covariant=True)
_P_co = TypeVar("_P_co", bound="Path", covariant=True)
_I_co = TypeVar("_I_co", bound="Integral", covariant=True)


class _PhaseShape(Protocol[_S_co, _C_co, _P_co, _I_co]):
    """A phase handle, as far as its vectors: what a `yapss.Phase` subclass satisfies.

    ``solution[ph]`` differs by phase, so nothing written on the solution could type it; the
    handle can, as ``arg[ph]`` is typed from it. A checker that does not follow the match --
    PyCharm's engine does not -- is given the type by annotating the variable instead.
    """

    @property
    def state(self) -> _S_co: ...
    @property
    def control(self) -> _C_co: ...
    @property
    def path(self) -> _P_co: ...
    @property
    def integral(self) -> _I_co: ...


class PhaseSolution(Generic[S_co, C_co, P_co, I_co]):
    """One phase of a solution.

    Attributes
    ----------
    time : numpy.ndarray
        The points every quantity of the phase is given on, under whatever the phase calls
        its independent variable -- `time` unless it was named otherwise.
    state, dynamics : Vector
        Arrays over `time`, named by the phase's state class.
    control : Vector
        Arrays over `time`, named by the phase's control class.
    path : Vector
        Arrays over `time`, named by the phase's path class.
    integrand : Vector
        Arrays over `time`, named by the phase's integral class.
    integral : Vector
        One value per integral.
    multiplier : PhaseMultiplier
        The multipliers, in the same shapes: ``ps.multiplier.path.g``.
    costate : Vector
        The multiplier of the dynamics, which is ``ps.multiplier.dynamics`` under the name the
        field uses for it -- the same object.
    initial, final : EndpointValues
        The phase's variables at each end, read as a callback reads them.
    duration : float
        The extent of the phase.
    hamiltonian : numpy.ndarray
        The Hamiltonian over `time`.
    collocated : numpy.ndarray
        Which points of `time` the solver produced values at. Every per-point quantity is given
        on every point of `time`; where this is false, the value is the method's polynomial
        extrapolated there -- close, and fine to plot, but not the solver's, and not bound by
        anything the solver imposed. Checks and statistics use ``quantity[ps.collocated]``.
    mesh : Mesh
        The mesh the phase was solved on.
    nlp : PhaseNLP
        The phase's positions in the solver's record, ``solution.nlp``.
    """

    __slots__ = (
        "_independent",
        "_points",
        "collocated",
        "control",
        "costate",
        "duration",
        "dynamics",
        "final",
        "hamiltonian",
        "initial",
        "integral",
        "integrand",
        "mesh",
        "multiplier",
        "nlp",
        "path",
        "state",
    )

    if TYPE_CHECKING:
        time: NDArray[np.float64]
        state: S_co
        dynamics: S_co
        costate: S_co
        control: C_co
        path: P_co
        integrand: I_co
        integral: I_co
        multiplier: PhaseMultiplier[S_co, C_co, P_co, I_co]
        # The state and the independent variable in one namespace, which is an intersection
        # Python's typing cannot write; see `args.Endpoint`. `ps.state.h[0]` is the typed read.
        initial: Any
        final: Any
        duration: float
        hamiltonian: NDArray[np.float64]
        collocated: NDArray[np.bool_]
        mesh: Mesh
        nlp: PhaseNLP[S_co, C_co, P_co, I_co]

    def __init__(self, independent: str, values: dict[str, Any]) -> None:
        object.__setattr__(self, "_independent", independent)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def _from(
        cls, phase: PhaseSpec, data: Any, method: str, nlp: PhaseNLP[Any, Any, Any, Any]
    ) -> PhaseSolution[Any, Any, Any, Any]:
        """Return the solution of `phase` from the back end's record of it."""
        label = f"phase '{phase.name}' solution"
        state = np.asarray(data.state)
        grid = _Grid(method, phase.mesh.collocation_points, data.time, data.time_c)
        fill = grid.fill
        costate = _vector(phase.state, fill(data.costate), f"{label} costate")
        fields = phase.state._fields

        def at_end(value: float) -> EndpointMultiplier:
            return EndpointMultiplier(
                {"_fields": fields, "_independent": phase.independent, "_value": value}
            )

        multiplier = PhaseMultiplier(
            {
                "dynamics": costate,
                "control": _vector(
                    phase.control, fill(data.control_multiplier), f"{label} control multiplier"
                ),
                "path": _vector(phase.path, fill(data.path_multiplier), f"{label} path multiplier"),
                "integral": _vector(
                    phase.integral, data.integral_multiplier, f"{label} integral multiplier"
                ),
                "initial": at_end(data.initial_time_multiplier),
                "final": at_end(data.final_time_multiplier),
                "duration": data.duration_multiplier,
            }
        )
        return cls(
            phase.independent,
            {
                "_points": data.time,
                "collocated": grid.collocated,
                "state": _vector(phase.state, state, f"{label} state"),
                "costate": costate,
                "multiplier": multiplier,
                "dynamics": _vector(phase.state, fill(data.dynamics), f"{label} dynamics"),
                "control": _vector(phase.control, fill(data.control), f"{label} control"),
                "path": _vector(phase.path, fill(data.path), f"{label} path"),
                "integrand": _vector(phase.integral, fill(data.integrand), f"{label} integrand"),
                "integral": _vector(phase.integral, data.integral, f"{label} integral"),
                "initial": _endpoint(phase, state[:, 0], data.time[0], f"{label} initial"),
                "final": _endpoint(phase, state[:, -1], data.time[-1], f"{label} final"),
                "duration": data.time[-1] - data.time[0],
                "hamiltonian": fill(data.hamiltonian),
                "mesh": phase.mesh,
                "nlp": nlp,
            },
        )

    def _values(self) -> dict[str, Any]:
        return {
            name: object.__getattribute__(self, name)
            for name in self.__slots__
            if name != "_independent"
        }

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the phase solution as its independent variable's name and its values."""
        return (PhaseSolution, (object.__getattribute__(self, "_independent"), self._values()))

    def __getattr__(self, name: str) -> Any:
        """Return the points under the independent variable's name, or refuse the name."""
        if name.startswith("_"):
            raise AttributeError(name)
        independent = object.__getattribute__(self, "_independent")
        if name == independent:
            return object.__getattribute__(self, "_points")
        names = (*(n for n in self.__slots__ if not n.startswith("_")), independent)
        msg = f"the phase solution has no '{name}'.{suggest(name, names)}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: each name is one solved quantity."""
        del value
        msg = f"'{name}' cannot be assigned; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: each name is one solved quantity."""
        msg = f"'{name}' cannot be deleted; {_NAMES_FIXED}"
        raise AttributeError(msg)


class Solution(Generic[D_co, PR_co]):
    """The result of a solve.

    Attributes
    ----------
    objective : float
        The objective value.
    converged : bool
        Whether Ipopt reported a converged solve.
    status : IpoptStatus
        What Ipopt reported.
    name : str
        The name of the problem this is a solution to.
    spectral_method : str
        The spectral method it was solved with: ``"lgl"``, ``"lgr"`` or ``"lg"``.
    parameter, discrete : Vector
        The problem-level values, named by the classes the problem declared.
    multiplier : ProblemMultiplier
        Their multipliers, in the same shapes: ``solution.multiplier.discrete.d``.
    nlp : NLPRecord
        What Ipopt saw and returned, with the positions of every variable and constraint.
    """

    __slots__ = (
        "_names",
        "_phases",
        "converged",
        "discrete",
        "multiplier",
        "name",
        "nlp",
        "objective",
        "parameter",
        "spectral_method",
        "status",
    )

    if TYPE_CHECKING:
        objective: float
        converged: bool
        status: IpoptStatus
        name: str
        spectral_method: str
        parameter: PR_co
        discrete: D_co
        multiplier: ProblemMultiplier[D_co, PR_co]
        nlp: NLPRecord[D_co, PR_co]

    def __init__(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @classmethod
    def _from(cls, spec: ProblemSpec, record: Any, transcription: Any) -> Solution[Any, Any]:
        """Return the solution of `spec` from the back end's record of the solve.

        `transcription` is the back end's spec the solve ran from, which lays out the flat
        vectors. Only data is kept: nothing reached from here refers to the problem, its
        callbacks, or the classes it was declared with.
        """
        dv, cf, layouts = _positions_of(transcription)
        return cls(
            {
                "_names": tuple(phase.name for phase in spec.phases),
                "_phases": tuple(
                    PhaseSolution._from(
                        phase,
                        record.phase[phase.index],
                        spec.spectral_method,
                        _phase_nlp(
                            phase,
                            dv.phase[phase.index],
                            cf.phase[phase.index],
                            layouts[phase.index],
                        ),
                    )
                    for phase in spec.phases
                ),
                "nlp": NLPRecord._from(
                    record.nlp_info,
                    NLPIndex(
                        {
                            "variable": ProblemVariableIndex(
                                {
                                    "parameter": _vector(
                                        spec.parameter,
                                        np.array(dv.s, dtype=np.intp),
                                        "parameter index",
                                    )
                                }
                            ),
                            "constraint": ProblemConstraintIndex(
                                {
                                    "discrete": _vector(
                                        spec.discrete,
                                        np.array(cf.discrete, dtype=np.intp),
                                        "discrete index",
                                    )
                                }
                            ),
                        }
                    ),
                ),
                "objective": record.objective,
                "converged": record.converged,
                "status": record.status,
                "name": spec.name,
                "spectral_method": spec.spectral_method,
                "parameter": _vector(spec.parameter, record.parameter, "parameter"),
                "discrete": _vector(spec.discrete, record.discrete, "discrete"),
                "multiplier": ProblemMultiplier(
                    {
                        "parameter": _vector(
                            spec.parameter, record.parameter_multiplier, "parameter multiplier"
                        ),
                        "discrete": _vector(
                            spec.discrete, record.discrete_multiplier, "discrete multiplier"
                        ),
                    }
                ),
            }
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle the solution as its values, which are data all the way down."""
        values = {name: object.__getattribute__(self, name) for name in self.__slots__}
        return (Solution, (values,))

    @overload
    def __getitem__(self, phase: str) -> PhaseSolution: ...

    @overload
    def __getitem__(
        self, phase: _PhaseShape[_S_co, _C_co, _P_co, _I_co]
    ) -> PhaseSolution[_S_co, _C_co, _P_co, _I_co]: ...

    def __getitem__(self, phase: Any) -> PhaseSolution:
        """Return the solution for `phase`: a phase handle, or the phase's name.

        A handle is matched by its position and its name, not by identity, and a name is enough
        on its own, so a solution unpickled in another process is readable with no problem at
        all: ``solution["boost"]``.
        """
        names: tuple[str, ...] = object.__getattribute__(self, "_names")
        phases: tuple[PhaseSolution, ...] = object.__getattribute__(self, "_phases")
        if isinstance(phase, str):
            if phase in names:
                return phases[names.index(phase)]
            hint = suggest(phase, names) if names else " The problem declared no phases."
            msg = f"the solution has no phase {phase!r}.{hint}"
            raise KeyError(msg)
        index = getattr(phase, "_index", None)
        name = getattr(phase, "_name", None)
        if isinstance(index, int) and 0 <= index < len(names) and names[index] == name:
            return phases[index]
        msg = (
            f"solution[...] takes a phase handle, such as 'problem.phases.<name>', or a phase's "
            f"name; got {phase!r}"
        )
        raise KeyError(msg)

    # Hidden from type checkers: one that sees a reader answering any name stops
    # reporting misspellings. The names are declared for them instead.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Refuse an unknown name with a suggestion."""
            if name.startswith("_"):
                raise AttributeError(name)
            names = tuple(n for n in self.__slots__ if not n.startswith("_"))
            if name == "nlp_info":
                # the 0.3.0 name, which a script being ported reaches for
                msg = "the solution has no 'nlp_info'; what 0.3.0 called nlp_info is 'nlp'."
            else:
                msg = f"the solution has no '{name}'.{suggest(name, names)}"
            raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse every assignment: each name is one solved quantity."""
        del value
        msg = f"'{name}' cannot be assigned; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> None:
        """Refuse every deletion: each name is one solved quantity."""
        msg = f"'{name}' cannot be deleted; {_NAMES_FIXED}"
        raise AttributeError(msg)

    def __repr__(self) -> str:
        """Return a short representation naming the objective and status."""
        return f"<Solution objective={self.objective!r} converged={self.converged!r}>"
