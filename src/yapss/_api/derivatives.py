"""

What a derivative callback fills in, under ``derivatives.method = "user"``.

A derivative is reached by the names of the things it relates, read in the order it is
spoken::

    jacobian.dynamics.x.v      the derivative of the dynamics of `x` with respect to `v`
    hessian.dynamics.x.v.u     the second derivative, chaining twice
    gradient[ph].final.time    the derivative of the objective by a phase's final time

The variable is named on its own, because a phase has one differentiation namespace and that
is what the Jacobian's columns are: the phase's state, its control, its independent variable,
and the problem's parameters. The output stays qualified, because a dynamics row is named for
its state and so can never be distinct from that namespace.

**What is written is the sparsity structure.** A name not written is a derivative that is zero
everywhere; a derivative that is zero only at this point is written as ``0.0``. The set of
names written must therefore be the same on every call, which `Structure.compare` checks.

The objects here are targets, not values: they hold no numbers of their own, and every write
lands in one dictionary keyed the way the transcription expects -- ``(("f", 0), ("x", 2))``
for a continuous Jacobian, ``(0, "tf", 0)`` for an endpoint. Navigation and assignment are
split across classes and modes so that each node refuses what it cannot do with a message
naming the form that works: a Jacobian row assigns and cannot be navigated further, a Hessian
row navigates and cannot be assigned.

A block field is addressed a row at a time, with an index on whichever side it appears:
``jacobian.dynamics.r[0].v[1]``. The index lands on a `_Block` node, which knows only where
the block starts and how long it is; what follows the row -- more of the derivative, or the
end of it -- belongs to the site that built the node, not to the index.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .containers import suggest

if TYPE_CHECKING:
    from .spec import PhaseSpec
    from .vector import Vector

__all__ = [
    "ContinuousHessian",
    "ContinuousJacobian",
    "DiscreteHessian",
    "DiscreteJacobian",
    "EndpointColumns",
    "ObjectiveGradient",
    "ObjectiveHessian",
    "PhaseColumns",
    "Structure",
    "endpoint_columns",
    "phase_columns",
]

# The transcription's own names for the groups of columns and rows. They are an internal
# vocabulary and appear in no message; everything the user sees is spelled in their names.
_STATE, _CONTROL, _INDEPENDENT, _PARAMETER = "x", "u", "t", "s"
_DYNAMICS, _INTEGRAND, _PATH = "f", "g", "h"

_ENDS = ("initial", "final", "integral")

_NEEDS_ROW = (
    "{spelling} is a block field of {size} rows, so it names no single one. Give the row, "
    "'{spelling}[0]'."
)

_ROW_RANGE = "{spelling}[{index}] is out of range for a block field of {size} rows."

_NOT_A_BLOCK = "{spelling} has one row, so it takes no row index; write '{spelling}' alone."


def _rows_of(declaration: type[Vector], group: str) -> dict[str, tuple[str, int]]:
    """Return the key of each scalar field of `declaration`, under the group name `group`."""
    return {name: (group, row) for name, row in declaration._single.items()}


def _blocks_of(declaration: type[Vector], group: str) -> dict[str, tuple[str, int, int]]:
    """Return `(group, offset, size)` for each block field of `declaration`.

    A block field names as many rows as it has, so it becomes a key only once a row index has
    been given. The offset is where its first row sits in the flat vector, which is what the
    index is added to; the group is what the key is built with once it is.
    """
    return {name: (group, offset, size) for name, (offset, size) in declaration._block.items()}


def _row(offset: int, size: int, index: Any, spelling: str) -> int:
    """Return the flat row that `index` picks out of a block field, or refuse the index.

    A negative index is refused rather than wrapped: it would name the same row as a
    non-negative one under a second spelling, and the two would be two names for one
    derivative -- which `Structure.store` catches, but later and less clearly.
    """
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool):
        msg = f"{spelling} takes a row index, an integer from 0 to {size - 1}; got {index!r}"
        raise TypeError(msg)
    if not 0 <= int(index) < size:
        raise IndexError(_ROW_RANGE.format(spelling=spelling, index=index, size=size))
    return offset + int(index)


# --------------------------------------------------------------- the namespaces, per solve


class PhaseColumns:
    """The differentiation namespace of one phase, and the output rows written against it.

    The state, the control, the independent variable and the problem's parameters are one
    space, which is why a derivative names a variable without saying which vector it came
    from. `phase()` and `Problem.__init__` between them guarantee those names do not collide,
    so this mapping is well defined.

    Attributes
    ----------
    label : str
        How the phase is named in a message.
    variables : dict
        Each variable name, mapped to the column key the transcription uses.
    outputs : dict
        Each output group, mapped to its rows by name.
    blocks : dict
        Each block field, mapped to the group its key is built with, where its first row sits
        in the flat vector, and how many rows it has.
    """

    __slots__ = ("blocks", "label", "output_blocks", "outputs", "variables")

    def __init__(self, phase: PhaseSpec, parameter: type[Vector]) -> None:
        self.label = f"phase '{phase.name}'"
        self.variables: dict[str, tuple[str, int]] = {
            **_rows_of(phase.state, _STATE),
            **_rows_of(phase.control, _CONTROL),
            phase.independent: (_INDEPENDENT, 0),
            **_rows_of(parameter, _PARAMETER),
        }
        self.blocks: dict[str, tuple[str, int, int]] = {
            **_blocks_of(phase.state, _STATE),
            **_blocks_of(phase.control, _CONTROL),
            **_blocks_of(parameter, _PARAMETER),
        }
        self.outputs: dict[str, dict[str, tuple[str, int]]] = {
            "dynamics": _rows_of(phase.state, _DYNAMICS),
            "integrand": _rows_of(phase.integral, _INTEGRAND),
            "path": _rows_of(phase.path, _PATH),
        }
        self.output_blocks: dict[str, dict[str, tuple[str, int, int]]] = {
            "dynamics": _blocks_of(phase.state, _DYNAMICS),
            "integrand": _blocks_of(phase.integral, _INTEGRAND),
            "path": _blocks_of(phase.path, _PATH),
        }


def phase_columns(phase: PhaseSpec, parameter: type[Vector]) -> PhaseColumns:
    """Return the differentiation namespace of `phase`.

    Parameters
    ----------
    phase : PhaseSpec
        The phase, as the snapshot holds it.
    parameter : type[Vector]
        The problem's parameter declaration, whose names join the phase's.

    Returns
    -------
    PhaseColumns
        The namespace.
    """
    return PhaseColumns(phase, parameter)


class PhaseEndpointNames:
    """One phase's endpoint variables, grouped by the end they are read at."""

    __slots__ = (
        "blocks",
        "final",
        "independent",
        "initial",
        "integral",
        "label",
        "name",
    )

    def __init__(self, phase: PhaseSpec) -> None:
        self.name = phase.name
        self.label = f"phase '{phase.name}'"
        self.independent = phase.independent
        index = phase.index
        self.initial: dict[str, Any] = {
            **{name: (index, "x0", row) for name, row in phase.state._single.items()},
            phase.independent: (index, "t0", 0),
        }
        self.final: dict[str, Any] = {
            **{name: (index, "xf", row) for name, row in phase.state._single.items()},
            phase.independent: (index, "tf", 0),
        }
        self.integral: dict[str, Any] = {
            name: (index, "q", row) for name, row in phase.integral._single.items()
        }
        # A block field's key is built once its row is known, so what is kept here is where
        # the block starts and how long it is, per end. The phase index leads every endpoint
        # key, so it is folded into the group name rather than carried separately.
        self.blocks: dict[str, dict[str, tuple[Any, int, int]]] = {
            "initial": {
                name: ((index, "x0"), offset, size)
                for name, (offset, size) in phase.state._block.items()
            },
            "final": {
                name: ((index, "xf"), offset, size)
                for name, (offset, size) in phase.state._block.items()
            },
            "integral": {
                name: ((index, "q"), offset, size)
                for name, (offset, size) in phase.integral._block.items()
            },
        }


class EndpointColumns:
    """The differentiation namespace of the endpoint callbacks: every phase's endpoints.

    A phase's endpoint variables are its state at each end, its independent variable at each
    end, and its integrals. The parameters belong to the problem rather than to a phase and
    are named without one. A phase's extent is not here: it is a constraint on the two ends,
    not a variable, so it is on the far side of a derivative (see `args.Endpoint`).

    Attributes
    ----------
    phases : dict
        Each phase handle, mapped to that phase's endpoint namespace.
    parameter : dict
        Each parameter name, mapped to its column key.
    """

    __slots__ = (
        "discrete",
        "discrete_blocks",
        "example",
        "parameter",
        "parameter_blocks",
        "phases",
    )

    def __init__(
        self,
        phases: tuple[PhaseSpec, ...],
        parameter: type[Vector],
        discrete: type[Vector] | None = None,
    ) -> None:
        self.phases: dict[Any, PhaseEndpointNames] = {
            phase.handle: PhaseEndpointNames(phase) for phase in phases
        }
        self.example = phases[0].name if phases else "<name>"
        self.parameter: dict[str, Any] = {
            name: (0, _PARAMETER, row) for name, row in parameter._single.items()
        }
        self.parameter_blocks: dict[str, tuple[Any, int, int]] = {
            name: ((0, _PARAMETER), offset, size)
            for name, (offset, size) in parameter._block.items()
        }
        self.discrete: dict[str, int] = dict(discrete._single) if discrete is not None else {}
        self.discrete_blocks: dict[str, tuple[int, int]] = (
            dict(discrete._block) if discrete is not None else {}
        )

    def lookup(self, handle: Any, spelling: str) -> PhaseEndpointNames:
        """Return the endpoint namespace `handle` names, or refuse what is not a handle."""
        try:
            return self.phases[handle]
        except (KeyError, TypeError):
            msg = (
                f"{spelling}[...] takes a phase handle, such as "
                f"'problem.phases.{self.example}'; got {handle!r}"
            )
            raise KeyError(msg) from None


def endpoint_columns(
    phases: tuple[PhaseSpec, ...],
    parameter: type[Vector],
    discrete: type[Vector] | None = None,
) -> EndpointColumns:
    """Return the namespace the endpoint derivative callbacks write in.

    Parameters
    ----------
    phases : tuple of PhaseSpec
        The problem's phases, in declaration order.
    parameter : type[Vector]
        The problem's parameter declaration.
    discrete : type[Vector], optional
        The discrete constraint declaration, whose fields are the rows a discrete derivative
        is written against. The objective's derivatives need none.

    Returns
    -------
    EndpointColumns
        The namespace.
    """
    return EndpointColumns(phases, parameter, discrete)


# ------------------------------------------------------------------------------ the store


class Structure:
    """The entries one derivative callback wrote, and the names they were written under.

    The values go to the transcription and the names stay here, for the messages: a mirrored
    Hessian pair and a structure that changed between calls are both reported in the user's
    own spelling, which is the only spelling they can act on.

    Parameters
    ----------
    what : str
        How the callback is named in a message, such as ``"jacobian"``.
    """

    __slots__ = ("entries", "pairs", "spelling", "what")

    def __init__(self, what: str) -> None:
        self.what = what
        self.entries: dict[Any, Any] = {}
        self.spelling: dict[Any, str] = {}
        self.pairs: dict[Any, str] = {}

    def store(self, key: Any, spelling: str, value: Any) -> None:
        """Record one derivative, refusing one key reached by two different names."""
        previous = self.spelling.get(key)
        if previous is not None and previous != spelling:
            msg = (
                f"'{spelling}' and '{previous}' are the same derivative, and both were "
                f"written. Write each one once."
            )
            raise ValueError(msg)
        self.entries[key] = value
        self.spelling[key] = spelling

    def store_pair(self, key: Any, canonical: Any, spelling: str, value: Any) -> None:
        """Record one second derivative, refusing the same unordered pair written twice.

        The two orders name one derivative. Assembled, they would become mirrored coordinates
        that the structure fold sums -- right for a user who split one derivative across the
        two keys and wrong (doubled) for one who supplied both triangles, and nothing
        downstream can tell which was meant. So the ambiguity is refused here, where the
        message names what was written rather than the transcription's coordinates.
        """
        previous = self.pairs.get(canonical)
        if previous is not None and previous != spelling:
            msg = (
                f"'{spelling}' and '{previous}' are the same second derivative written both "
                f"ways round. Write each unordered pair once."
            )
            raise ValueError(msg)
        self.pairs[canonical] = spelling
        self.store(key, spelling, value)

    def compare(self, first: dict[Any, str], callback: str) -> None:
        """Refuse a key set that differs from the one the first call established.

        What is written is the sparsity structure, so it is fixed at the first call and every
        later call must write the same names. A name that appeared later would be dropped
        silently; one that disappeared would leave a stale value in its place.

        Parameters
        ----------
        first : dict
            The spellings the first call wrote, keyed as this call's are.
        callback : str
            How the callback is named in the message.
        """
        added = sorted(self.spelling[key] for key in self.spelling.keys() - first.keys())
        removed = sorted(first[key] for key in first.keys() - self.spelling.keys())
        if not added and not removed:
            return
        parts = []
        if added:
            parts.append(f"wrote {', '.join(added)}, which it did not write before")
        if removed:
            parts.append(f"did not write {', '.join(removed)}, which it wrote before")
        msg = (
            f"the {callback} callback {' and '.join(parts)}. What is written is the sparsity "
            f"structure, so the same names must be written on every call; a derivative that "
            f"is zero only at this point is written as 0.0."
        )
        raise ValueError(msg)


def _variable(name: str, columns: PhaseColumns, spelling: str) -> tuple[str, int]:
    """Return the column `name` addresses, or refuse it.

    A block field reaches this only when it was written without a row index, which is why the
    message names the form that works rather than saying the name is unknown.
    """
    column = columns.variables.get(name)
    if column is not None:
        return column
    block = columns.blocks.get(name)
    if block is not None:
        raise AttributeError(_NEEDS_ROW.format(spelling=spelling, size=block[2]))
    msg = (
        f"{spelling}: {columns.label} has no variable '{name}'."
        f"{suggest(name, (*columns.variables, *columns.blocks))} A derivative names a "
        f"variable on its own: the phase's state, its control, its independent variable, or "
        f"a parameter."
    )
    raise AttributeError(msg)


class _Block:
    """A block field awaiting the row index that picks one of its rows.

    Every place a block field can be named produces one of these. What follows the index
    differs -- another piece of the derivative, or the end of it -- which is the difference
    between the two subclasses, and it is decided by the site rather than by the index: a
    block output row is always stepped through, a block second variable is always written to.
    """

    __slots__ = ("_offset", "_size", "_spelling", "_then")

    def __init__(self, offset: int, size: int, spelling: str, then: Any) -> None:
        object.__setattr__(self, "_offset", offset)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_spelling", spelling)
        object.__setattr__(self, "_then", then)

    def _resolve(self, index: Any) -> tuple[int, str]:
        """Return the flat row `index` picks, and how that row is spelled."""
        spelling: str = object.__getattribute__(self, "_spelling")
        row = _row(
            object.__getattribute__(self, "_offset"),
            object.__getattribute__(self, "_size"),
            index,
            spelling,
        )
        return row, f"{spelling}[{index}]"

    def _needs_row(self) -> AttributeError:
        return AttributeError(
            _NEEDS_ROW.format(
                spelling=object.__getattribute__(self, "_spelling"),
                size=object.__getattribute__(self, "_size"),
            )
        )

    def __getattr__(self, name: str) -> Any:
        """Refuse a name written where a row index belongs."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise self._needs_row()

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment written where a row index belongs."""
        del name, value
        raise self._needs_row()


class _BlockStep(_Block):
    """A block field whose row leads on to the rest of the derivative."""

    __slots__ = ()

    def __getitem__(self, index: Any) -> Any:
        """Return whatever follows the row `index`."""
        row, spelling = self._resolve(index)
        return object.__getattribute__(self, "_then")(row, spelling)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse an assignment where more of the derivative is still expected."""
        del value
        _row_, spelling = self._resolve(index)
        del _row_
        msg = f"{spelling} names part of a derivative; more of it is expected before the '='."
        raise TypeError(msg)


class _BlockWrite(_Block):
    """A block field whose row ends the derivative, so the row is assigned."""

    __slots__ = ()

    def __setitem__(self, index: Any, value: Any) -> None:
        """Record the derivative at the row `index`."""
        row, spelling = self._resolve(index)
        object.__getattribute__(self, "_then")(row, spelling, value)

    def __getitem__(self, index: Any) -> Any:
        """Refuse a read: a row of this block is written, not navigated."""
        _row_, spelling = self._resolve(index)
        del _row_
        msg = f"{spelling} is the whole of the derivative; write '{spelling} = ...'."
        raise TypeError(msg)


# -------------------------------------------------------------- the continuous derivatives


class _JacobianRow:
    """One output row of a continuous Jacobian: assign a variable to give its derivative."""

    __slots__ = ("_columns", "_row", "_spelling", "_store")

    def __init__(self, store: Structure, columns: PhaseColumns, row: Any, spelling: str) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_row", row)
        object.__setattr__(self, "_spelling", spelling)

    def __setattr__(self, name: str, value: Any) -> None:
        """Record the derivative of this row with respect to the variable `name`."""
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        column = _variable(name, object.__getattribute__(self, "_columns"), spelling)
        store: Structure = object.__getattribute__(self, "_store")
        store.store((object.__getattribute__(self, "_row"), column), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return a block variable awaiting its row, or refuse a chained name."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: PhaseColumns = object.__getattribute__(self, "_columns")
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        block = columns.blocks.get(name)
        if block is not None:
            store: Structure = object.__getattribute__(self, "_store")
            row = object.__getattribute__(self, "_row")
            group, offset, size = block

            def write(column_row: int, at: str, value: Any) -> None:
                store.store((row, (group, column_row)), at, value)

            return _BlockWrite(offset, size, spelling, write)
        msg = (
            f"{spelling} is a first derivative and is written by assigning it, "
            f"'{spelling} = ...'. A second derivative chains one more name and belongs in "
            f"the hessian callback."
        )
        raise AttributeError(msg)


class _HessianPair:
    """A second derivative with its first variable named: assign the second to give it."""

    __slots__ = ("_columns", "_first", "_row", "_spelling", "_store")

    def __init__(
        self, store: Structure, columns: PhaseColumns, row: Any, first: Any, spelling: str
    ) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_row", row)
        object.__setattr__(self, "_first", first)
        object.__setattr__(self, "_spelling", spelling)

    def __setattr__(self, name: str, value: Any) -> None:
        """Record the second derivative, refusing the same pair written the other way."""
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        second = _variable(name, object.__getattribute__(self, "_columns"), spelling)
        row = object.__getattribute__(self, "_row")
        first = object.__getattribute__(self, "_first")
        store: Structure = object.__getattribute__(self, "_store")
        store.store_pair((row, first, second), (row, *sorted((first, second))), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return a block second variable awaiting its row, or refuse a third name."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling: str = object.__getattribute__(self, "_spelling")
        columns: PhaseColumns = object.__getattribute__(self, "_columns")
        block = columns.blocks.get(name)
        if block is not None:
            store: Structure = object.__getattribute__(self, "_store")
            row = object.__getattribute__(self, "_row")
            first = object.__getattribute__(self, "_first")
            group, offset, size = block

            def write(column_row: int, at: str, value: Any) -> None:
                second = (group, column_row)
                store.store_pair((row, first, second), (row, *sorted((first, second))), at, value)

            return _BlockWrite(offset, size, f"{spelling}.{name}", write)
        msg = (
            f"{spelling}.{name}: a second derivative names two variables, so "
            f"'{spelling} = ...' is the whole of it."
        )
        raise AttributeError(msg)


class _HessianRow:
    """One output row of a continuous Hessian: navigate to the first variable."""

    __slots__ = ("_columns", "_row", "_spelling", "_store")

    def __init__(self, store: Structure, columns: PhaseColumns, row: Any, spelling: str) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_row", row)
        object.__setattr__(self, "_spelling", spelling)

    def __getattr__(self, name: str) -> Any:
        """Return the node holding this row and its first variable."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: PhaseColumns = object.__getattribute__(self, "_columns")
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        store: Structure = object.__getattribute__(self, "_store")
        row = object.__getattribute__(self, "_row")
        block = columns.blocks.get(name)
        if block is not None:
            group, offset, size = block
            return _BlockStep(
                offset,
                size,
                spelling,
                lambda column_row, at: _HessianPair(store, columns, row, (group, column_row), at),
            )
        return _HessianPair(store, columns, row, _variable(name, columns, spelling), spelling)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse assigning after one variable: a second derivative needs two."""
        del value
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        msg = (
            f"{spelling} names one variable. A second derivative names two, for example "
            f"'{spelling}.<variable> = ...'."
        )
        raise AttributeError(msg)


class _OutputGroup:
    """One group of output rows -- the dynamics, the integrands, or the path constraints."""

    __slots__ = ("_columns", "_group", "_node", "_rows", "_store", "_what")

    def __init__(
        self, store: Structure, columns: PhaseColumns, group: str, node: Any, what: str
    ) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_rows", columns.outputs[group])
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name: str) -> Any:
        """Return the node for the output row `name`."""
        if name.startswith("_"):
            raise AttributeError(name)
        rows: dict[str, tuple[str, int]] = object.__getattribute__(self, "_rows")
        columns: PhaseColumns = object.__getattribute__(self, "_columns")
        what: str = object.__getattribute__(self, "_what")
        group: str = object.__getattribute__(self, "_group")
        spelling = f"{what}.{group}.{name}"
        node: Any = object.__getattribute__(self, "_node")
        store: Structure = object.__getattribute__(self, "_store")
        row = rows.get(name)
        if row is not None:
            return node(store, columns, row, spelling)
        block = columns.output_blocks[group].get(name)
        if block is not None:
            group_name, offset, size = block
            return _BlockStep(
                offset,
                size,
                spelling,
                lambda output_row, at: node(store, columns, (group_name, output_row), at),
            )
        blocks = columns.output_blocks[group]
        msg = f"{what}.{group} has no '{name}'.{suggest(name, (*rows, *blocks))}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse assigning an output row, which names no variable to differentiate by."""
        del value
        what: str = object.__getattribute__(self, "_what")
        group: str = object.__getattribute__(self, "_group")
        msg = (
            f"{what}.{group}.{name} names a row but no variable. A derivative names both, "
            f"for example '{what}.{group}.{name}.<variable> = ...'."
        )
        raise AttributeError(msg)


class _ContinuousTarget:
    """What a continuous derivative callback fills in: the three groups of output rows."""

    __slots__ = ("_groups", "_what")

    def __init__(self, store: Structure, columns: PhaseColumns, node: Any, what: str) -> None:
        object.__setattr__(
            self,
            "_groups",
            {group: _OutputGroup(store, columns, group, node, what) for group in columns.outputs},
        )
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name: str) -> Any:
        """Return one of the three output groups."""
        if name.startswith("_"):
            raise AttributeError(name)
        groups: dict[str, Any] = object.__getattribute__(self, "_groups")
        try:
            return groups[name]
        except KeyError:
            what: str = object.__getattribute__(self, "_what")
            msg = f"{what} has no '{name}'.{suggest(name, tuple(groups))}"
            raise AttributeError(msg) from None

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse replacing an output group."""
        del value
        what: str = object.__getattribute__(self, "_what")
        msg = (
            f"{what}.{name} cannot be replaced; write one derivative at a time, for example "
            f"'{what}.dynamics.<row>.<variable> = ...'."
        )
        raise AttributeError(msg)


class ContinuousJacobian(_ContinuousTarget):
    """What ``ph.register.continuous_jacobian`` fills in.

    ``jacobian.dynamics.x.v = ...`` is the derivative of the dynamics of ``x`` with respect
    to ``v``.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: PhaseColumns) -> None:
        super().__init__(store, columns, _JacobianRow, "jacobian")


class ContinuousHessian(_ContinuousTarget):
    """What ``ph.register.continuous_hessian`` fills in.

    ``hessian.dynamics.x.v.u = ...`` is the second derivative of the dynamics of ``x`` with
    respect to ``v`` and ``u``. The two orders name one derivative, so each unordered pair is
    written once and writing both is refused rather than summed.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: PhaseColumns) -> None:
        super().__init__(store, columns, _HessianRow, "hessian")


# ---------------------------------------------------------------- the endpoint derivatives
#
# An endpoint derivative names a phase and then a variable of it. A second one names two,
# each through its own phase, so the chain is twice as long::
#
#     gradient[ph].final.time = 1.0
#     hessian[ph].initial.r[ph].initial.r = 8.0
#
# One set of nodes serves both, carrying a mode: `first` means one more variable is expected
# and assigning is refused, `only` means this variable ends the derivative. The second half
# of a Hessian runs in `only` mode with the first variable's key held alongside it.


class _Context:
    """What every node of one endpoint derivative shares.

    `mode` is ``"first"`` while a second variable is still expected and ``"only"`` once the
    next name ends the derivative, which is what tells a gradient's nodes from a Hessian's
    and a Hessian's first half from its second.

    `prefix` is what the key is built on. The objective has one derivative, so nothing goes in
    front of the variables; a discrete constraint has one per group row, so its row does. That
    is the only difference between the two, which is why one set of nodes serves both.
    """

    __slots__ = ("columns", "mode", "prefix", "store", "what")

    def __init__(
        self,
        store: Structure,
        columns: EndpointColumns,
        what: str,
        mode: str,
        prefix: tuple[Any, ...] = (),
    ) -> None:
        self.store = store
        self.columns = columns
        self.what = what
        self.mode = mode
        self.prefix = prefix

    def at(self, prefix: tuple[Any, ...], what: str) -> _Context:
        """Return the same context scoped to one output row."""
        return _Context(self.store, self.columns, what, self.mode, prefix)

    def ending(self) -> _Context:
        """Return the same context with the next name ending the derivative."""
        return _Context(self.store, self.columns, self.what, "only", self.prefix)

    def key(self, *variables: Any) -> Any:
        """Return the transcription's key for a derivative by `variables` at this row.

        The back end spells a *lone* decision-variable key bare rather than wrapped, so an
        objective gradient is keyed `(0, "tf", 0)` where a discrete Jacobian is keyed
        `(row, (0, "tf", 0))`. That irregularity is confined here.
        """
        if self.prefix:
            return (*self.prefix, *variables)
        return variables[0] if len(variables) == 1 else variables

    def canonical(self, first: Any, second: Any) -> Any:
        """Return the order-free key of one unordered pair, for the mirrored-pair check."""
        return (*self.prefix, *sorted((first, second)))


class _EndpointEnd:
    """One end of one phase -- its state there, its independent variable, or its integrals."""

    __slots__ = ("_context", "_end", "_first", "_names", "_phase", "_spelling")

    def __init__(
        self,
        context: _Context,
        phase: PhaseEndpointNames,
        end: str,
        spelling: str,
        first: Any,
    ) -> None:
        for attribute, value in (
            ("_context", context),
            ("_phase", phase),
            ("_end", end),
            ("_names", getattr(phase, end)),
            ("_spelling", spelling),
            ("_first", first),
        ):
            object.__setattr__(self, attribute, value)

    def _blocks(self) -> dict[str, tuple[Any, int, int]]:
        phase: PhaseEndpointNames = object.__getattribute__(self, "_phase")
        return phase.blocks[object.__getattribute__(self, "_end")]

    def _key(self, name: str, spelling: str) -> Any:
        """Return the column `name` addresses, for a scalar field only."""
        names: dict[str, Any] = object.__getattribute__(self, "_names")
        key = names.get(name)
        if key is not None:
            return key
        blocks = self._blocks()
        if name in blocks:
            raise AttributeError(_NEEDS_ROW.format(spelling=spelling, size=blocks[name][2]))
        phase: PhaseEndpointNames = object.__getattribute__(self, "_phase")
        msg = f"{spelling}: {phase.label} has no '{name}'." f"{suggest(name, (*names, *blocks))}"
        raise AttributeError(msg)

    def _store(self, key: Any, spelling: str, value: Any) -> None:
        """Record a derivative by this endpoint variable, first or only."""
        context: _Context = object.__getattribute__(self, "_context")
        first = object.__getattribute__(self, "_first")
        if first is None:
            context.store.store(context.key(key), spelling, value)
        else:
            context.store.store_pair(
                context.key(first, key), context.canonical(first, key), spelling, value
            )

    def __setattr__(self, name: str, value: Any) -> None:
        """Record the derivative, unless a second variable is still expected."""
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        context: _Context = object.__getattribute__(self, "_context")
        if context.mode == "first":
            msg = (
                f"{spelling} names one variable. A second derivative names two, and the "
                f"second is reached through its phase, for example "
                f"'{spelling}[phase].final.<name> = ...'."
            )
            raise AttributeError(msg)
        self._store(self._key(name, spelling), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return a block awaiting its row, or the node awaiting the second variable."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        context: _Context = object.__getattribute__(self, "_context")
        block = self._blocks().get(name)
        if block is not None:
            prefix, offset, size = block
            if context.mode == "first":
                return _BlockStep(
                    offset,
                    size,
                    spelling,
                    lambda row, at: _FirstNamed(context, (*prefix, row), at),
                )
            return _BlockWrite(
                offset,
                size,
                spelling,
                lambda row, at, value: self._store((*prefix, row), at, value),
            )
        if context.mode != "first":
            msg = f"{spelling} is the whole of the derivative; write '{spelling} = ...'."
            raise AttributeError(msg)
        return _FirstNamed(context, self._key(name, spelling), spelling)


class _Endpoint:
    """One phase's endpoint namespace, as a derivative names it."""

    __slots__ = ("_context", "_first", "_phase", "_spelling")

    def __init__(
        self, context: _Context, phase: PhaseEndpointNames, spelling: str, first: Any
    ) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_spelling", spelling)
        object.__setattr__(self, "_first", first)

    def __getattr__(self, name: str) -> Any:
        """Return one of `initial`, `final` and `integral`."""
        if name.startswith("_"):
            raise AttributeError(name)
        phase: PhaseEndpointNames = object.__getattribute__(self, "_phase")
        spelling: str = object.__getattribute__(self, "_spelling")
        if name not in _ENDS:
            msg = (
                f"{spelling}.{name}: an endpoint derivative names 'initial', 'final' or "
                f"'integral'.{suggest(name, _ENDS)}"
            )
            raise AttributeError(msg)
        return _EndpointEnd(
            object.__getattribute__(self, "_context"),
            phase,
            name,
            f"{spelling}.{name}",
            object.__getattribute__(self, "_first"),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse assigning to an endpoint group, which names no variable."""
        del value
        spelling: str = object.__getattribute__(self, "_spelling")
        msg = (
            f"{spelling}.{name} names no variable. An endpoint derivative names the end and "
            f"the variable, for example '{spelling}.final.<name> = ...'."
        )
        raise AttributeError(msg)


class _FirstNamed:
    """A second derivative whose first variable is named: select the second's phase next."""

    __slots__ = ("_context", "_first", "_spelling")

    def __init__(self, context: _Context, first: Any, spelling: str) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_first", first)
        object.__setattr__(self, "_spelling", spelling)

    def __getitem__(self, handle: Any) -> Any:
        """Return the endpoint namespace of the phase the second variable belongs to."""
        context: _Context = object.__getattribute__(self, "_context")
        spelling: str = object.__getattribute__(self, "_spelling")
        phase = context.columns.lookup(handle, spelling)
        # the second variable ends the derivative, whatever mode the first was reached in
        return _Endpoint(
            context.ending(),
            phase,
            f"{spelling}[{phase.name}]",
            object.__getattribute__(self, "_first"),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Record a second derivative whose second variable is a parameter."""
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        key = _parameter(name, context.columns, spelling, context.what)
        first = object.__getattribute__(self, "_first")
        context.store.store_pair(
            context.key(first, key), context.canonical(first, key), spelling, value
        )

    def __getattr__(self, name: str) -> Any:
        """Return a block parameter as the second variable, or refuse a bare name."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling: str = object.__getattribute__(self, "_spelling")
        context: _Context = object.__getattribute__(self, "_context")
        block = context.columns.parameter_blocks.get(name)
        if block is not None:
            prefix, offset, size = block
            first = object.__getattribute__(self, "_first")

            def write(row: int, at: str, value: Any) -> None:
                second = (*prefix, row)
                context.store.store_pair(
                    context.key(first, second), context.canonical(first, second), at, value
                )

            return _BlockWrite(offset, size, f"{spelling}.{name}", write)
        msg = (
            f"{spelling}.{name}: the second variable of an endpoint Hessian is reached "
            f"through its phase, '{spelling}[phase].final.{name} = ...', or named directly "
            f"if it is a parameter, '{spelling}.{name} = ...'."
        )
        raise AttributeError(msg)


def _parameter(name: str, columns: EndpointColumns, spelling: str, what: str) -> Any:
    """Return the column the parameter `name` addresses, for a scalar parameter only."""
    key = columns.parameter.get(name)
    if key is not None:
        return key
    block = columns.parameter_blocks.get(name)
    if block is not None:
        raise AttributeError(_NEEDS_ROW.format(spelling=spelling, size=block[2]))
    msg = (
        f"{spelling}: there is no parameter '{name}'."
        f"{suggest(name, (*columns.parameter, *columns.parameter_blocks))} An endpoint "
        f"variable is reached through its phase, '{what}[phase].final.{name} = ...'."
    )
    raise AttributeError(msg)


class _EndpointTarget:
    """What an endpoint derivative callback fills in: phases by handle, parameters by name."""

    __slots__ = ("_context",)

    def __init__(self, store: Structure, columns: EndpointColumns, what: str, mode: str) -> None:
        object.__setattr__(self, "_context", _Context(store, columns, what, mode))

    @classmethod
    def _over(cls, context: _Context) -> _EndpointTarget:
        """Return a target over a context already built, as a discrete group's row needs."""
        target = object.__new__(_EndpointTarget)
        object.__setattr__(target, "_context", context)
        return target

    def __getitem__(self, handle: Any) -> Any:
        """Return the endpoint namespace of the phase `handle` names."""
        context: _Context = object.__getattribute__(self, "_context")
        phase = context.columns.lookup(handle, context.what)
        return _Endpoint(context, phase, f"{context.what}[{phase.name}]", None)

    def __setattr__(self, name: str, value: Any) -> None:
        """Record a derivative with respect to a parameter, which belongs to no phase."""
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{context.what}.{name}"
        key = _parameter(name, context.columns, spelling, context.what)
        if context.mode == "first":
            msg = (
                f"{spelling} names one variable. A second derivative names two, for example "
                f"'{spelling}[phase].final.<name> = ...' or '{spelling}.<parameter> = ...'."
            )
            raise AttributeError(msg)
        context.store.store(context.key(key), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return a parameter's node: a block awaiting its row, or a Hessian's first half."""
        if name.startswith("_"):
            raise AttributeError(name)
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{context.what}.{name}"
        block = context.columns.parameter_blocks.get(name)
        if block is not None:
            prefix, offset, size = block
            if context.mode == "first":
                return _BlockStep(
                    offset,
                    size,
                    spelling,
                    lambda row, at: _FirstNamed(context, (*prefix, row), at),
                )
            return _BlockWrite(
                offset,
                size,
                spelling,
                lambda row, at, value: context.store.store(context.key((*prefix, row)), at, value),
            )
        if context.mode != "first":
            msg = (
                f"{spelling} is written, not read. A derivative by an endpoint variable is "
                f"reached through its phase, '{context.what}[phase].final.{name} = ...'; one "
                f"by a parameter is '{spelling} = ...'."
            )
            raise AttributeError(msg)
        return _FirstNamed(
            context, _parameter(name, context.columns, spelling, context.what), spelling
        )


class _DiscreteGroups:
    """The discrete constraint groups of a derivative: one endpoint namespace per row.

    ``jacobian.discrete.link`` scopes the endpoint nodes to that group's row, and everything
    after it -- the phase selector, the end, the variable -- is what an objective derivative
    writes. The two differ only in what the key is built on (`_Context.prefix`).
    """

    __slots__ = ("_context", "_rows")

    def __init__(self, context: _Context, rows: dict[str, int]) -> None:
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_rows", rows)

    def __getattr__(self, name: str) -> Any:
        """Return the endpoint namespace scoped to the group `name`."""
        if name.startswith("_"):
            raise AttributeError(name)
        rows: dict[str, int] = object.__getattribute__(self, "_rows")
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{context.what}.discrete.{name}"
        row = rows.get(name)
        if row is not None:
            return _EndpointTarget._over(context.at((row,), spelling))
        blocks = context.columns.discrete_blocks
        block = blocks.get(name)
        if block is not None:
            offset, size = block
            return _BlockStep(
                offset,
                size,
                spelling,
                lambda group_row, at: _EndpointTarget._over(context.at((group_row,), at)),
            )
        msg = f"{context.what}.discrete has no group '{name}'." f"{suggest(name, (*rows, *blocks))}"
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse assigning a group, which names no variable to differentiate by."""
        del value
        context: _Context = object.__getattribute__(self, "_context")
        msg = (
            f"{context.what}.discrete.{name} names a constraint but no variable. A "
            f"derivative names both, for example "
            f"'{context.what}.discrete.{name}[phase].final.<name> = ...'."
        )
        raise AttributeError(msg)


class _DiscreteTarget:
    """What a discrete derivative callback fills in: the groups, under ``.discrete``."""

    __slots__ = ("_context", "_groups")

    def __init__(self, store: Structure, columns: EndpointColumns, what: str, mode: str) -> None:
        context = _Context(store, columns, what, mode)
        object.__setattr__(self, "_context", context)
        object.__setattr__(self, "_groups", _DiscreteGroups(context, columns.discrete))

    def __getattr__(self, name: str) -> Any:
        """Return the groups. There is one namespace here, and it is `discrete`."""
        if name.startswith("_"):
            raise AttributeError(name)
        context: _Context = object.__getattribute__(self, "_context")
        if name != "discrete":
            msg = (
                f"{context.what}.{name}: a discrete derivative names a constraint group, "
                f"'{context.what}.discrete.<group>'.{suggest(name, ('discrete',))}"
            )
            raise AttributeError(msg)
        return object.__getattribute__(self, "_groups")

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse replacing the groups."""
        del value
        context: _Context = object.__getattribute__(self, "_context")
        msg = (
            f"{context.what}.{name} cannot be replaced; write one derivative at a time, for "
            f"example '{context.what}.discrete.<group>[phase].final.<name> = ...'."
        )
        raise AttributeError(msg)


class DiscreteJacobian(_DiscreteTarget):
    """What ``problem.register.discrete_jacobian`` fills in.

    ``jacobian.discrete.link[ph].final.h = -1.0`` is the derivative of the constraint group
    ``link`` with respect to the phase's final ``h``.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "jacobian", "only")


class DiscreteHessian(_DiscreteTarget):
    """What ``problem.register.discrete_hessian`` fills in.

    ``hessian.discrete.orbit[ph].final.r[ph].final.r`` is a second derivative of one group:
    the first variable is named through its phase, then the second is. Each unordered pair is
    written once. A linear constraint contributes nothing, and saying nothing about it is how
    that is said.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "hessian", "first")


class ObjectiveGradient(_EndpointTarget):
    """What ``problem.register.objective_gradient`` fills in.

    ``gradient[ph].final.time = 1.0`` is the derivative of the objective with respect to the
    final value of that phase's independent variable.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "gradient", "only")


class ObjectiveHessian(_EndpointTarget):
    """What ``problem.register.objective_hessian`` fills in.

    ``hessian[ph].initial.r[ph].initial.r = 8.0`` is a second derivative: the first variable
    is named through its phase, then the second is. Each unordered pair is written once.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "hessian", "first")
