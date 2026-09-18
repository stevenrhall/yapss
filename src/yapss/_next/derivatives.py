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

Stage 1 reaches scalar fields only. A block field is refused by name, since its rows are
addressed with an index (``jacobian.dynamics.r[0].v[1]``) that is not built yet.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .containers import suggest

if TYPE_CHECKING:
    from .spec import PhaseSpec
    from .vector import Vector

__all__ = [
    "ContinuousHessian",
    "ContinuousJacobian",
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

_BLOCKED = (
    "{spelling}: '{name}' is a block field, which derivatives supplied by hand do not yet "
    "reach. Use 'auto' or a central-difference method for this problem."
)


def _rows_of(declaration: type[Vector], group: str) -> dict[str, tuple[str, int]]:
    """Return the key of each scalar field of `declaration`, under the group name `group`."""
    return {name: (group, row) for name, row in declaration._single.items()}


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
    blocked : tuple of str
        Names that exist but are block fields, which stage 1 does not reach.
    """

    __slots__ = ("blocked", "label", "outputs", "variables")

    def __init__(self, phase: PhaseSpec, parameter: type[Vector]) -> None:
        self.label = f"phase '{phase.name}'"
        self.variables: dict[str, tuple[str, int]] = {
            **_rows_of(phase.state, _STATE),
            **_rows_of(phase.control, _CONTROL),
            phase.independent: (_INDEPENDENT, 0),
            **_rows_of(parameter, _PARAMETER),
        }
        self.outputs: dict[str, dict[str, tuple[str, int]]] = {
            "dynamics": _rows_of(phase.state, _DYNAMICS),
            "integrand": _rows_of(phase.integral, _INTEGRAND),
            "path": _rows_of(phase.path, _PATH),
        }
        self.blocked = tuple(
            name
            for declaration in (phase.state, phase.control, phase.integral, phase.path, parameter)
            for name in declaration._block
        )


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

    __slots__ = ("blocked", "final", "independent", "initial", "integral", "label", "name")

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
        self.blocked = (*phase.state._block, *phase.integral._block)


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

    __slots__ = ("example", "parameter", "parameter_blocked", "phases")

    def __init__(self, phases: tuple[PhaseSpec, ...], parameter: type[Vector]) -> None:
        self.phases: dict[Any, PhaseEndpointNames] = {
            phase.handle: PhaseEndpointNames(phase) for phase in phases
        }
        self.example = phases[0].name if phases else "<name>"
        self.parameter: dict[str, Any] = {
            name: (0, _PARAMETER, row) for name, row in parameter._single.items()
        }
        self.parameter_blocked = tuple(parameter._block)

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


def endpoint_columns(phases: tuple[PhaseSpec, ...], parameter: type[Vector]) -> EndpointColumns:
    """Return the namespace the endpoint derivative callbacks write in.

    Parameters
    ----------
    phases : tuple of PhaseSpec
        The problem's phases, in declaration order.
    parameter : type[Vector]
        The problem's parameter declaration.

    Returns
    -------
    EndpointColumns
        The namespace.
    """
    return EndpointColumns(phases, parameter)


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
    """Return the column `name` addresses, or refuse it."""
    column = columns.variables.get(name)
    if column is not None:
        return column
    if name in columns.blocked:
        raise AttributeError(_BLOCKED.format(spelling=spelling, name=name))
    msg = (
        f"{spelling}: {columns.label} has no variable '{name}'."
        f"{suggest(name, tuple(columns.variables))} A derivative names a variable on its "
        f"own: the phase's state, its control, its independent variable, or a parameter."
    )
    raise AttributeError(msg)


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
        """Refuse navigating past a first derivative, naming the callback that chains."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
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
        """Refuse a third name: a second derivative relates two variables, not three."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling: str = object.__getattribute__(self, "_spelling")
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
        first = _variable(name, columns, spelling)
        return _HessianPair(
            object.__getattribute__(self, "_store"),
            columns,
            object.__getattribute__(self, "_row"),
            first,
            spelling,
        )

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
        row = rows.get(name)
        if row is None:
            if name in columns.blocked:
                raise AttributeError(_BLOCKED.format(spelling=spelling, name=name))
            msg = f"{what}.{group} has no '{name}'.{suggest(name, tuple(rows))}"
            raise AttributeError(msg)
        node: Any = object.__getattribute__(self, "_node")
        return node(object.__getattribute__(self, "_store"), columns, row, spelling)

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
    """

    __slots__ = ("columns", "mode", "store", "what")

    def __init__(self, store: Structure, columns: EndpointColumns, what: str, mode: str) -> None:
        self.store = store
        self.columns = columns
        self.what = what
        self.mode = mode


class _EndpointEnd:
    """One end of one phase -- its state there, its independent variable, or its integrals."""

    __slots__ = ("_context", "_first", "_names", "_phase", "_spelling")

    def __init__(
        self,
        context: _Context,
        phase: PhaseEndpointNames,
        names: dict[str, Any],
        spelling: str,
        first: Any,
    ) -> None:
        for attribute, value in (
            ("_context", context),
            ("_phase", phase),
            ("_names", names),
            ("_spelling", spelling),
            ("_first", first),
        ):
            object.__setattr__(self, attribute, value)

    def _key(self, name: str, spelling: str) -> Any:
        names: dict[str, Any] = object.__getattribute__(self, "_names")
        key = names.get(name)
        if key is not None:
            return key
        phase: PhaseEndpointNames = object.__getattribute__(self, "_phase")
        if name in phase.blocked:
            raise AttributeError(_BLOCKED.format(spelling=spelling, name=name))
        msg = f"{spelling}: {phase.label} has no '{name}'.{suggest(name, tuple(names))}"
        raise AttributeError(msg)

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
        key = self._key(name, spelling)
        first = object.__getattribute__(self, "_first")
        if first is None:
            context.store.store(key, spelling, value)
        else:
            context.store.store_pair((first, key), tuple(sorted((first, key))), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return the node awaiting the second variable, when one is expected."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        context: _Context = object.__getattribute__(self, "_context")
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
            getattr(phase, name),
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
        ending = _Context(context.store, context.columns, context.what, "only")
        return _Endpoint(
            ending, phase, f"{spelling}[{phase.name}]", object.__getattribute__(self, "_first")
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Record a second derivative whose second variable is a parameter."""
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{object.__getattribute__(self, '_spelling')}.{name}"
        key = _parameter(name, context.columns, spelling, context.what)
        first = object.__getattribute__(self, "_first")
        context.store.store_pair((first, key), tuple(sorted((first, key))), spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Refuse a bare name: the second variable is reached through its phase."""
        if name.startswith("_"):
            raise AttributeError(name)
        spelling: str = object.__getattribute__(self, "_spelling")
        msg = (
            f"{spelling}.{name}: the second variable of an endpoint Hessian is reached "
            f"through its phase, '{spelling}[phase].final.{name} = ...', or named directly "
            f"if it is a parameter, '{spelling}.{name} = ...'."
        )
        raise AttributeError(msg)


def _parameter(name: str, columns: EndpointColumns, spelling: str, what: str) -> Any:
    """Return the column the parameter `name` addresses, or refuse it."""
    key = columns.parameter.get(name)
    if key is not None:
        return key
    if name in columns.parameter_blocked:
        raise AttributeError(_BLOCKED.format(spelling=spelling, name=name))
    msg = (
        f"{spelling}: there is no parameter '{name}'."
        f"{suggest(name, tuple(columns.parameter))} An endpoint variable is reached through "
        f"its phase, '{what}[phase].final.{name} = ...'."
    )
    raise AttributeError(msg)


class _EndpointTarget:
    """What an endpoint derivative callback fills in: phases by handle, parameters by name."""

    __slots__ = ("_context",)

    def __init__(self, store: Structure, columns: EndpointColumns, what: str, mode: str) -> None:
        object.__setattr__(self, "_context", _Context(store, columns, what, mode))

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
        context.store.store(key, spelling, value)

    def __getattr__(self, name: str) -> Any:
        """Return a parameter's node for a Hessian; for a gradient, nothing is read."""
        if name.startswith("_"):
            raise AttributeError(name)
        context: _Context = object.__getattribute__(self, "_context")
        spelling = f"{context.what}.{name}"
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
