"""

What a derivative callback fills in, under ``derivatives.method = "user"``.

A continuous derivative is reached by the names of the things it relates, read in the order
it is spoken::

    jacobian.dynamics.x.v      the derivative of the dynamics of `x` with respect to `v`
    hessian.dynamics.x.v.u     the second derivative, chaining twice

The variable is named on its own, because a phase has one differentiation namespace and that
is what the Jacobian's columns are: the phase's state, its control, its independent variable,
and the problem's parameters. The output stays qualified, because a dynamics row is named for
its state and so can never be distinct from that namespace.

An *endpoint* variable is not one coordinate but four -- phase, end, field, row -- so it is
not a name, and a path of attributes is the wrong spelling for it. It is a value instead::

    f = hessian.phases[ph].final
    hessian.discrete.gap[f.x, f.v] = 2.0
    gradient[gradient.phases[ph].final.time] = 1.0

Binding the end is what gets a column back down to one dot, and the binding carries the three
coordinates that are not the field name under a name the reader chose. The namespaces hang
off the target -- `phases`, `parameter`, and `discrete` for the rows -- which are the only
names at that level, so no declared field can collide with them.

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
``jacobian.dynamics.r[0]`` on the output, ``f.r[0]`` on a column. The index lands on a
`_Block` or a `ColumnBlock`, which knows only where the block starts and how long it is.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .containers import suggest

if TYPE_CHECKING:
    from .spec import PhaseSpec
    from .vector import Vector

__all__ = [
    "Column",
    "ColumnBlock",
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
    from. A phase's declaration and `Problem.__init__` between them guarantee those names do
    not collide, so this mapping is well defined.

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
# An endpoint variable is four coordinates -- phase, end, field, row -- where a continuous one
# is a single name. A name is the best spelling there is for one coordinate and a bad one for
# four: spelled as a path, a column cannot be held in a variable, so it has to be written out
# in full wherever it appears, twice in every second derivative, with nothing between the two
# halves to show where one ends.
#
# So here a column is a *value*. `hessian.phases[ph].final` is a namespace, `.x` of it is a
# `Column`, and the derivative is written by subscripting the row with the columns it relates::
#
#     f = hessian.phases[ph].final
#     hessian.discrete.gap[f.x, f.v] = 2.0
#     hessian.discrete.gap[f.x, hessian.parameter.beta[1]] = 1.0
#
# Binding the *end* rather than each variable is what gets the cost of a column back down to
# one dot, and the binding carries the three coordinates that are not the field name under a
# name the user chose -- which is the right person to choose it, since `x` across two phases
# genuinely denotes two different things.
#
# The namespace hangs off the target, so nothing has to be hoisted out of the callback and no
# further argument is passed to it. Its three roots -- `phases`, `parameter`, `discrete` --
# are the only names at the top of a target, so no field a user declares can collide there.


class Column:
    """One endpoint variable, as a derivative names it.

    A column is a value: it can be bound to a name, kept in a list, built in a loop and passed
    to a helper. That is the whole point of it, and what a path spelled out of attribute
    access cannot do.

    It carries the namespace it came from, so that a column belonging to another problem is
    refused rather than landing in a structurally valid but wrong place -- which would cost
    iterations instead of raising, the one failure this surface exists to prevent.
    """

    __slots__ = ("_key", "_origin", "_spelling")

    def __init__(self, key: Any, spelling: str, origin: EndpointColumns) -> None:
        object.__setattr__(self, "_key", key)
        object.__setattr__(self, "_spelling", spelling)
        object.__setattr__(self, "_origin", origin)

    def __repr__(self) -> str:
        """Return the spelling the column was reached by, which is what a message wants."""
        spelling: str = object.__getattribute__(self, "_spelling")
        return spelling

    def __getitem__(self, index: Any) -> Any:
        """Refuse a row index on a variable that has one row."""
        del index
        raise TypeError(_NOT_A_BLOCK.format(spelling=object.__getattribute__(self, "_spelling")))


class ColumnBlock:
    """The columns of one block field: a family of variables, one per row.

    A row of it is a column; the block itself is not, because it names as many variables as it
    has rows. It is indexable and iterable, so a derivative over a block is a loop over the
    rows rather than a loop over integers that has to agree with the declaration.
    """

    __slots__ = ("_offset", "_origin", "_prefix", "_size", "_spelling")

    def __init__(
        self, prefix: tuple[Any, ...], offset: int, size: int, spelling: str, origin: Any
    ) -> None:
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_offset", offset)
        object.__setattr__(self, "_size", size)
        object.__setattr__(self, "_spelling", spelling)
        object.__setattr__(self, "_origin", origin)

    def __repr__(self) -> str:
        """Return the spelling of the block, without a row."""
        spelling: str = object.__getattribute__(self, "_spelling")
        return spelling

    def __len__(self) -> int:
        """Return how many rows the block has."""
        return int(object.__getattribute__(self, "_size"))

    def __getitem__(self, index: Any) -> Column:
        """Return the column of one row."""
        spelling: str = object.__getattribute__(self, "_spelling")
        row = _row(
            object.__getattribute__(self, "_offset"),
            object.__getattribute__(self, "_size"),
            index,
            spelling,
        )
        prefix: tuple[Any, ...] = object.__getattribute__(self, "_prefix")
        return Column(
            (*prefix, row),
            f"{spelling}[{index}]",
            object.__getattribute__(self, "_origin"),
        )

    def __iter__(self) -> Any:
        """Return each row's column in order."""
        return (self[index] for index in range(len(self)))


class _Space(NamedTuple):
    """One namespace of endpoint variables: what it holds, and how it names itself."""

    names: dict[str, Any]
    blocks: dict[str, tuple[Any, int, int]]
    label: str
    tail: str = ""


def _column_or_block(space: _Space, name: str, short: str, full: str, origin: Any) -> Any:
    """Return the column or block of columns `name` addresses in `space`, or refuse it.

    `short` is how the result is spelled inside an entry, where the target already names
    itself; `full` is how it is spelled in a message, which has to stand on its own.
    """
    key = space.names.get(name)
    if key is not None:
        return Column(key, short, origin)
    block = space.blocks.get(name)
    if block is not None:
        prefix, offset, size = block
        return ColumnBlock(prefix, offset, size, short, origin)
    near = suggest(name, (*space.names, *space.blocks))
    msg = f"{full}: {space.label} has no '{name}'.{near}{space.tail}"
    raise AttributeError(msg)


class _ColumnEnd:
    """The columns of one end of one phase, or of its integrals."""

    __slots__ = ("_columns", "_end", "_full", "_phase", "_short")

    def __init__(
        self, columns: EndpointColumns, phase: PhaseEndpointNames, end: str, short: str, full: str
    ) -> None:
        for attribute, value in (
            ("_columns", columns),
            ("_phase", phase),
            ("_end", end),
            ("_short", short),
            ("_full", full),
        ):
            object.__setattr__(self, attribute, value)

    def __getattr__(self, name: str) -> Any:
        """Return the column, or the block of columns, one variable of this end is."""
        if name.startswith("_"):
            raise AttributeError(name)
        phase: PhaseEndpointNames = object.__getattribute__(self, "_phase")
        end: str = object.__getattribute__(self, "_end")
        return _column_or_block(
            _Space(getattr(phase, end), phase.blocks[end], phase.label),
            name,
            f"{object.__getattribute__(self, '_short')}.{name}",
            f"{object.__getattribute__(self, '_full')}.{name}",
            object.__getattribute__(self, "_columns"),
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a variable, which names a column and not a derivative."""
        del value
        full: str = object.__getattribute__(self, "_full")
        short: str = object.__getattribute__(self, "_short")
        msg = (
            f"{full}.{name} is a variable, not a derivative. A derivative is written on the "
            f"target, with the variables it relates in the subscript, such as "
            f"'<target>[{short}.{name}] = ...'."
        )
        raise AttributeError(msg)


class _ColumnPhase:
    """One phase's endpoint namespace: its two ends and its integrals."""

    __slots__ = ("_columns", "_full", "_phase", "_short")

    def __init__(
        self, columns: EndpointColumns, phase: PhaseEndpointNames, short: str, full: str
    ) -> None:
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_short", short)
        object.__setattr__(self, "_full", full)

    def __getattr__(self, name: str) -> Any:
        """Return one of `initial`, `final` and `integral`."""
        if name.startswith("_"):
            raise AttributeError(name)
        full: str = object.__getattribute__(self, "_full")
        if name not in _ENDS:
            msg = (
                f"{full}.{name}: an endpoint variable is at 'initial', 'final' or "
                f"'integral'.{suggest(name, _ENDS)}"
            )
            raise AttributeError(msg)
        return _ColumnEnd(
            object.__getattribute__(self, "_columns"),
            object.__getattribute__(self, "_phase"),
            name,
            f"{object.__getattribute__(self, '_short')}.{name}",
            f"{full}.{name}",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment where an end is expected."""
        del value
        full: str = object.__getattribute__(self, "_full")
        msg = (
            f"{full}.{name} names no variable. An endpoint variable is named at an end, such "
            f"as '{full}.final.<name>'."
        )
        raise AttributeError(msg)


class _ColumnPhases:
    """Every phase's endpoint namespace, reached by handle as the values are."""

    __slots__ = ("_columns", "_full", "_short")

    def __init__(self, columns: EndpointColumns, short: str, full: str) -> None:
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_short", short)
        object.__setattr__(self, "_full", full)

    def __getitem__(self, handle: Any) -> _ColumnPhase:
        """Return the endpoint namespace of the phase `handle` names."""
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        phase = columns.lookup(handle, object.__getattribute__(self, "_full"))
        return _ColumnPhase(
            columns,
            phase,
            f"{object.__getattribute__(self, '_short')}[{phase.name}]",
            f"{object.__getattribute__(self, '_full')}[{phase.name}]",
        )

    def __getattr__(self, name: str) -> Any:
        """Refuse a phase named rather than handed over, since the values take a handle too."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        full: str = object.__getattribute__(self, "_full")
        msg = (
            f"{full}.{name}: a phase is reached by its handle, as its values are, such as "
            f"'{full}[problem.phases.{columns.example}]'."
        )
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment where a phase handle belongs."""
        del value
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        full: str = object.__getattribute__(self, "_full")
        msg = (
            f"{full}.{name} names no variable. A phase is reached by its handle, such as "
            f"'{full}[problem.phases.{columns.example}].final.<name>'."
        )
        raise AttributeError(msg)


class _ColumnParameters:
    """The problem's parameters, which belong to no phase and so are named without one."""

    __slots__ = ("_columns", "_full", "_short")

    def __init__(self, columns: EndpointColumns, short: str, full: str) -> None:
        object.__setattr__(self, "_columns", columns)
        object.__setattr__(self, "_short", short)
        object.__setattr__(self, "_full", full)

    def __getattr__(self, name: str) -> Any:
        """Return the column, or block of columns, one parameter is."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        tail = (
            " An endpoint variable is reached through its phase, "
            f"'<target>.phases[phase].final.{name}'."
        )
        return _column_or_block(
            _Space(columns.parameter, columns.parameter_blocks, "the problem", tail),
            name,
            f"{object.__getattribute__(self, '_short')}.{name}",
            f"{object.__getattribute__(self, '_full')}.{name}",
            columns,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a parameter, which names a column and not a derivative."""
        del value
        full: str = object.__getattribute__(self, "_full")
        short: str = object.__getattribute__(self, "_short")
        msg = (
            f"{full}.{name} is a variable, not a derivative. A derivative is written on the "
            f"target, with the variables it relates in the subscript, such as "
            f"'<target>[{short}.{name}] = ...'."
        )
        raise AttributeError(msg)


# ------------------------------------------------------------------------- what is written


def _entry_key(prefix: tuple[Any, ...], keys: tuple[Any, ...]) -> Any:
    """Return the transcription's key for a derivative by `keys` at the row `prefix`.

    The back end spells a *lone* decision-variable key bare rather than wrapped, so an
    objective gradient is keyed `(0, "tf", 0)` where a discrete Jacobian is keyed
    `(row, (0, "tf", 0))`. That irregularity is confined here.
    """
    if prefix:
        return (*prefix, *keys)
    return keys[0] if len(keys) == 1 else keys


class _Row:
    """One row of one derivative: subscript it with the variables the derivative relates.

    `wants` is 1 for a first derivative and 2 for a second, which is the only difference
    between the four endpoint targets once the row is fixed.
    """

    __slots__ = ("_columns", "_prefix", "_spelling", "_store", "_wants")

    def __init__(
        self,
        store: Structure,
        columns: EndpointColumns,
        spelling: str,
        prefix: tuple[Any, ...],
        wants: int,
    ) -> None:
        for attribute, value in (
            ("_store", store),
            ("_columns", columns),
            ("_spelling", spelling),
            ("_prefix", prefix),
            ("_wants", wants),
        ):
            object.__setattr__(self, attribute, value)

    def _resolve(self, index: Any) -> tuple[tuple[Any, ...], str]:
        """Return the keys the subscript names and how the whole entry is spelled."""
        spelling: str = object.__getattribute__(self, "_spelling")
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        wants: int = object.__getattribute__(self, "_wants")
        given = index if isinstance(index, tuple) else (index,)
        if len(given) != wants:
            raise TypeError(_wrong_count(spelling, given, wants))
        keys = []
        for column in given:
            if not isinstance(column, Column):
                raise TypeError(_not_a_column(spelling, column, wants))
            if object.__getattribute__(column, "_origin") is not columns:
                msg = (
                    f"{spelling}[{column!r}]: that variable belongs to another problem. Take "
                    f"it from this derivative's own namespace."
                )
                raise ValueError(msg)
            keys.append(object.__getattribute__(column, "_key"))
        written = ", ".join(repr(column) for column in given)
        return tuple(keys), f"{spelling}[{written}]"

    def __setitem__(self, index: Any, value: Any) -> None:
        """Record the derivative by the variables the subscript names."""
        keys, spelling = self._resolve(index)
        store: Structure = object.__getattribute__(self, "_store")
        prefix: tuple[Any, ...] = object.__getattribute__(self, "_prefix")
        key = _entry_key(prefix, keys)
        if len(keys) == 1:
            store.store(key, spelling, value)
        else:
            store.store_pair(key, (*prefix, *sorted(keys)), spelling, value)

    def __getitem__(self, index: Any) -> Any:
        """Refuse a read: a derivative is written here, not navigated."""
        _keys, spelling = self._resolve(index)
        del _keys
        msg = f"{spelling} is the whole of the derivative; write '{spelling} = ...'."
        raise TypeError(msg)

    def __getattr__(self, name: str) -> Any:
        """Refuse a name written where the variables belong."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(_needs_variables(self, name))

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment written where the variables belong."""
        del value
        raise AttributeError(_needs_variables(self, name))


def _needs_variables(row: _Row, name: str) -> str:
    """Return what to say when a derivative names no variable to differentiate by."""
    spelling: str = object.__getattribute__(row, "_spelling")
    wants: int = object.__getattribute__(row, "_wants")
    form = "[<variable>]" if wants == 1 else "[<variable>, <variable>]"
    return (
        f"{spelling}.{name}: a derivative names the variables it relates in the subscript, "
        f"'{spelling}{form} = ...'."
    )


def _wrong_count(spelling: str, given: tuple[Any, ...], wants: int) -> str:
    """Return what to say when a derivative is written with the wrong number of variables."""
    written = ", ".join(repr(column) for column in given)
    if wants == 1:
        return (
            f"{spelling}[{written}] names {len(given)} variables. A first derivative names "
            f"one, '{spelling}[<variable>] = ...'."
        )
    return (
        f"{spelling}[{written}] names {len(given)} variable"
        f"{'' if len(given) == 1 else 's'}. A second derivative names two, "
        f"'{spelling}[<variable>, <variable>] = ...'."
    )


def _not_a_column(spelling: str, given: Any, wants: int) -> str:
    """Return what to say when something that is not an endpoint variable is subscripted."""
    form = "[<variable>]" if wants == 1 else "[<variable>, <variable>]"
    if isinstance(given, ColumnBlock):
        return _NEEDS_ROW.format(spelling=repr(given), size=len(given))
    return (
        f"{spelling}{form} takes an endpoint variable, such as "
        f"'<target>.phases[phase].final.<name>'; got {given!r}."
    )


class _RowBlock:
    """The rows of one block constraint group, awaiting the row the derivative is of."""

    __slots__ = ("_columns", "_offset", "_size", "_spelling", "_store", "_wants")

    def __init__(
        self,
        store: Structure,
        columns: EndpointColumns,
        spelling: str,
        block: tuple[int, int],
        wants: int,
    ) -> None:
        offset, size = block
        for attribute, value in (
            ("_store", store),
            ("_columns", columns),
            ("_spelling", spelling),
            ("_offset", offset),
            ("_size", size),
            ("_wants", wants),
        ):
            object.__setattr__(self, attribute, value)

    def __getitem__(self, index: Any) -> _Row:
        """Return the derivative of one row of the group."""
        spelling: str = object.__getattribute__(self, "_spelling")
        size: int = object.__getattribute__(self, "_size")
        if isinstance(index, (Column, ColumnBlock, tuple)):
            raise TypeError(_NEEDS_ROW.format(spelling=spelling, size=size))
        row = _row(object.__getattribute__(self, "_offset"), size, index, spelling)
        return _Row(
            object.__getattribute__(self, "_store"),
            object.__getattribute__(self, "_columns"),
            f"{spelling}[{index}]",
            (row,),
            object.__getattribute__(self, "_wants"),
        )

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse an assignment to a row before the variables are named."""
        del value
        spelling: str = object.__getattribute__(self, "_spelling")
        size: int = object.__getattribute__(self, "_size")
        if isinstance(index, (Column, ColumnBlock, tuple)):
            raise TypeError(_NEEDS_ROW.format(spelling=spelling, size=size))
        row = _row(object.__getattribute__(self, "_offset"), size, index, spelling)
        del row
        wants: int = object.__getattribute__(self, "_wants")
        form = "[<variable>]" if wants == 1 else "[<variable>, <variable>]"
        msg = (
            f"{spelling}[{index}] names a constraint but no variable. A derivative names "
            f"both, '{spelling}[{index}]{form} = ...'."
        )
        raise TypeError(msg)

    def __getattr__(self, name: str) -> Any:
        """Refuse a name written where a row index belongs."""
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(
            _NEEDS_ROW.format(
                spelling=object.__getattribute__(self, "_spelling"),
                size=object.__getattribute__(self, "_size"),
            )
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment written where a row index belongs."""
        del name, value
        raise AttributeError(
            _NEEDS_ROW.format(
                spelling=object.__getattribute__(self, "_spelling"),
                size=object.__getattribute__(self, "_size"),
            )
        )


class _DiscreteGroups:
    """The discrete constraint groups: one row of the derivative each, blocks by row."""

    __slots__ = ("_columns", "_spelling", "_store", "_wants")

    def __init__(
        self, store: Structure, columns: EndpointColumns, spelling: str, wants: int
    ) -> None:
        for attribute, value in (
            ("_store", store),
            ("_columns", columns),
            ("_spelling", spelling),
            ("_wants", wants),
        ):
            object.__setattr__(self, attribute, value)

    def __getattr__(self, name: str) -> Any:
        """Return the row, or the rows, of one constraint group."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        store: Structure = object.__getattribute__(self, "_store")
        wants: int = object.__getattribute__(self, "_wants")
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        row = columns.discrete.get(name)
        if row is not None:
            return _Row(store, columns, spelling, (row,), wants)
        block = columns.discrete_blocks.get(name)
        if block is not None:
            return _RowBlock(store, columns, spelling, block, wants)
        msg = (
            f"{object.__getattribute__(self, '_spelling')} has no group '{name}'."
            f"{suggest(name, (*columns.discrete, *columns.discrete_blocks))}"
        )
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse assigning a group, which names no variable to differentiate by."""
        del value
        wants: int = object.__getattribute__(self, "_wants")
        form = "[<variable>]" if wants == 1 else "[<variable>, <variable>]"
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        msg = (
            f"{spelling} names a constraint but no variable. A derivative names both, "
            f"'{spelling}{form} = ...'."
        )
        raise AttributeError(msg)


class _EndpointTarget:
    """What an endpoint derivative callback fills in, and the namespace it writes against.

    The three roots are fixed words -- `phases`, `parameter`, and `discrete` where there are
    discrete constraints -- so nothing a user declares is spelled at this level and no field
    name can collide with them.
    """

    __slots__ = ("_columns", "_store", "_wants", "_what")

    def __init__(self, store: Structure, columns: EndpointColumns, what: str, wants: int) -> None:
        for attribute, value in (
            ("_store", store),
            ("_columns", columns),
            ("_what", what),
            ("_wants", wants),
        ):
            object.__setattr__(self, attribute, value)

    def __getattr__(self, name: str) -> Any:
        """Return one of the namespaces a derivative is written against."""
        if name.startswith("_"):
            raise AttributeError(name)
        columns: EndpointColumns = object.__getattribute__(self, "_columns")
        what: str = object.__getattribute__(self, "_what")
        if name == "phases":
            return _ColumnPhases(columns, "phases", f"{what}.phases")
        if name == "parameter":
            return _ColumnParameters(columns, "parameter", f"{what}.parameter")
        offered = ("phases", "parameter")
        msg = (
            f"{what}.{name}: a derivative is written against 'phases[phase].<end>.<name>' or "
            f"'parameter.<name>'.{suggest(name, offered)}"
        )
        raise AttributeError(msg)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse an assignment to a namespace."""
        del value
        what: str = object.__getattribute__(self, "_what")
        msg = (
            f"{what}.{name} cannot be assigned; write one derivative at a time, with the "
            f"variables it relates in the subscript."
        )
        raise AttributeError(msg)


class _ObjectiveTarget(_EndpointTarget):
    """An objective derivative: one derivative, so the target is subscripted directly."""

    __slots__ = ()

    def _as_row(self) -> _Row:
        return _Row(
            object.__getattribute__(self, "_store"),
            object.__getattribute__(self, "_columns"),
            object.__getattribute__(self, "_what"),
            (),
            object.__getattribute__(self, "_wants"),
        )

    def __setitem__(self, index: Any, value: Any) -> None:
        """Record the derivative by the variables the subscript names."""
        self._as_row()[index] = value

    def __getitem__(self, index: Any) -> Any:
        """Refuse a read, with the message the row itself would give."""
        return self._as_row()[index]


class _DiscreteTarget(_EndpointTarget):
    """A discrete derivative: one derivative per constraint row, reached under `discrete`."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        """Return the constraint groups, or one of the variable namespaces."""
        if name == "discrete":
            return _DiscreteGroups(
                object.__getattribute__(self, "_store"),
                object.__getattribute__(self, "_columns"),
                f"{object.__getattribute__(self, '_what')}.discrete",
                object.__getattribute__(self, "_wants"),
            )
        return super().__getattr__(name)

    def __setitem__(self, index: Any, value: Any) -> None:
        """Refuse a derivative written without saying which constraint it is of."""
        del index, value
        what: str = object.__getattribute__(self, "_what")
        msg = (
            f"{what}[...] names no constraint. A discrete derivative names the constraint "
            f"first, '{what}.discrete.<group>[...] = ...'."
        )
        raise TypeError(msg)


class DiscreteJacobian(_DiscreteTarget):
    """What ``problem.register.discrete_jacobian`` fills in.

    ``jacobian.discrete.link[f.h] = -1.0``, where ``f = jacobian.phases[ph].final``, is the
    derivative of the constraint group ``link`` with respect to that phase's final ``h``.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "jacobian", 1)


class DiscreteHessian(_DiscreteTarget):
    """What ``problem.register.discrete_hessian`` fills in.

    ``hessian.discrete.orbit[f.r, f.r]``, where ``f = hessian.phases[ph].final``, is a second
    derivative of one group. Each unordered pair is written once. A linear constraint
    contributes nothing, and saying nothing about it is how that is said.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "hessian", 2)


class ObjectiveGradient(_ObjectiveTarget):
    """What ``problem.register.objective_gradient`` fills in.

    ``gradient[gradient.phases[ph].final.time] = 1.0`` is the derivative of the objective with
    respect to the final value of that phase's independent variable.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "gradient", 1)


class ObjectiveHessian(_ObjectiveTarget):
    """What ``problem.register.objective_hessian`` fills in.

    ``hessian[i.r, i.r] = 8.0``, where ``i = hessian.phases[ph].initial``, is a second
    derivative of the objective. Each unordered pair is written once.
    """

    __slots__ = ()

    def __init__(self, store: Structure, columns: EndpointColumns) -> None:
        super().__init__(store, columns, "hessian", 2)
