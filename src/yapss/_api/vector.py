"""

`Vector`, the single container of the redesigned YAPSS API, and the six roles a vector has.

A vector class is a *declaration*. It names the members of a state, control, path, integral,
parameter, or discrete-constraint vector, and says the shape of each::

    class Rocket(yapss.State):
        h = yapss.scalar()
        v = yapss.scalar()
        r = yapss.vector(3)

A declaration is made by subclassing its role -- `State`, `Control`, `Path`, `Integral`,
`Parameter` or `Discrete` -- so what the class is for is said where it is declared, and a
state handed to a phase as its control is refused. `Vector` is the base the roles share and is
not public: nothing a user writes is a vector without a role.

A role class declares no fields, so it is also the empty declaration of that role: a phase with
no path constraints has `Path` for its path, and there is no separate `Empty`.

A field is described by the docstring written under it, in the class body, which is where
Sphinx, an IDE and a reader all look. Nothing about a field is recorded but its shape.

The user never instantiates it. YAPSS creates every instance, and each one is a *kind*-specific
subclass generated once per (declaration, kind) pair: an instance holding bounds validates its
elements as bounds, one holding a callback's output rows validates them as rows, and so on.
The declaration is reused for every aspect of the vector it names, so a member is spelled the
same way wherever it is reached.

Names address fields; integers and slices address flat rows, so a vector field declared with
``vector(3)`` occupies three consecutive positions.

"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    SupportsFloat,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import numpy as np

from yapss.math.wrapper import SXArray

from .kinds import MISSING, Kind, is_sequence
from .sampled import Interp

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "ROLES",
    "Control",
    "Discrete",
    "Field",
    "Integral",
    "Maker",
    "Parameter",
    "Path",
    "State",
    "Vector",
    "role_of",
    "scalar",
    "vector",
]


class Field:
    """What one declared field records. Created by `field`, never directly.

    Attributes
    ----------
    size : int or None
        The number of rows in a block field, or None for a scalar member. A block field keeps
        its leading axis at every size, including one row and none.
    """

    __slots__ = ("size",)

    def __init__(self, *, size: int | None = None) -> None:
        self.size = size

    def __repr__(self) -> str:
        """Return the call that would declare this field."""
        return "scalar()" if self.size is None else f"vector({self.size})"

    @property
    def rows(self) -> int:
        """int: The number of flat rows the field occupies."""
        return 1 if self.size is None else self.size


# ------------------------------------------------------------ what a type checker sees
#
# A setting is written field first -- `ph.state.x.bounds` -- so the setting is the last name,
# and its type is known. Each marker returns a type listing the settings its rank takes and
# what each accepts, transcribed from the runtime grammar in `kinds`. The transcription is the
# whole risk: a type narrower than the runtime tells a user that working code is wrong, which
# is worse than no check. So where the grammar is rich -- a guess is a pair, a sample, an
# interpolant, or a number, depending on the role -- the type is `Any`, and the runtime says the
# rest. The corpus type-checking clean, with no `# type: ignore`, is the test.
#
# One type serves every role, so a setting one role lacks -- `ph.control.u.initial` -- passes
# the checker and is refused at run time. That is the permissive direction, and acceptable.
#
# The same markers are what a field is *read* as, in a callback or a solution, so they also
# declare arithmetic and indexing. `arg.state.v * 2.0` checks as well as `ph.state.v.bounds`;
# the price is that `arg.state.v.bounds` checks too, and fails only at run time.
#
# And they are what a callback *writes*: `out.dynamics.h = 0.0` assigns a value to a field
# whose declared type is the marker. So a marker is a descriptor to the checker, reading as
# itself and accepting any value. The same permission lets `ph.state.h = 0.0` pass the checker
# in setup, where the runtime refuses it with the form to use -- the permissive direction again.

_Bound: TypeAlias = "tuple[SupportsFloat | None, SupportsFloat | None] | list[SupportsFloat | None]"
"""A bound: a lower and an upper value, either of which may be None for no bound."""

_T = TypeVar("_T")


if TYPE_CHECKING:

    class _AsValue(np.ndarray[Any, np.dtype[Any]]):
        """A field read as its value, in a callback or solution: to a checker, an array.

        At run time a value read in a continuous callback or a solution is a numpy array --
        of floats, or of symbols under the "auto" trace -- so it is typed as one, which is what
        lets it pass to numpy, to `yapss.math`, and to a helper a user annotated as taking an
        array. A parameter or an endpoint value is a single number or symbol instead, and it
        also checks as an array: the permissive direction.
        """

        def __get__(self, obj: object, owner: Any) -> Self: ...
        def __set__(self, obj: object, value: Any) -> None: ...

else:

    class _AsValue:
        """A field marker's base, which at run time holds nothing."""


class ScalarField(_AsValue):
    """A field declared with `scalar()`, as a type checker sees it: one value per setting."""

    if TYPE_CHECKING:
        bounds: _Bound
        initial: _Bound
        final: _Bound
        guess: Any
        scale: SupportsFloat
        defect_scale: SupportsFloat


class WrittenByRows:
    """What a vector's setting takes when assigned whole: nothing.

    A vector field's settings are written through its rows, ``r.bounds[:] = ...`` for every row
    alike or ``r.bounds[0] = ...`` for one, because assigning the whole setting cannot say which
    was meant. A checker reporting this type is reporting that assignment.
    """


class _Rows(Generic[_T]):
    """One setting of a vector field, indexed by row."""

    if TYPE_CHECKING:

        def __getitem__(self, index: SupportsIndex | slice) -> Any: ...
        def __setitem__(self, index: SupportsIndex | slice, value: _T | Sequence[_T]) -> None: ...
        def __len__(self) -> int: ...


class _ByRows(Generic[_T]):
    """A vector field's setting: read and indexed by row, never assigned whole."""

    if TYPE_CHECKING:

        @overload
        def __get__(self, obj: None, owner: Any) -> _ByRows[_T]: ...
        @overload
        def __get__(self, obj: object, owner: Any) -> _Rows[_T]: ...
        def __get__(self, obj: Any, owner: Any) -> Any: ...
        def __set__(self, obj: object, value: WrittenByRows) -> None: ...


class VectorField(_AsValue):
    """A field declared with `vector(n)`, as a type checker sees it: each setting by row."""

    bounds: _ByRows[_Bound] = _ByRows()
    initial: _ByRows[_Bound] = _ByRows()
    final: _ByRows[_Bound] = _ByRows()
    guess: _ByRows[Any] = _ByRows()
    scale: _ByRows[SupportsFloat] = _ByRows()
    defect_scale: _ByRows[SupportsFloat] = _ByRows()


def scalar() -> ScalarField:
    """Declare a field holding one value: a scalar member of the vector.

    A scalar is read as ``(npoints,)`` in a callback. It is not the same as ``vector(1)``, which
    is a vector of one component and is read as ``(1, npoints)``: rank is part of the shape,
    and the two markers say which.

    Returns
    -------
    ScalarField
        A marker recording the shape. YAPSS replaces it when the class is defined, so it is
        never seen again; its type is what a checker reads the field as.
    """
    return cast("ScalarField", Field(size=None))


def vector(size: int) -> VectorField:
    """Declare a field holding `size` values: a vector member, such as a position.

    Any count from 0 up is allowed, so that a declaration built by an algorithm needs no special
    case at one component or none. A vector keeps its leading axis at every size, so
    ``vector(1)`` is read as ``(1, npoints)`` and never collapses to a scalar.

    Parameters
    ----------
    size : int
        The number of components.

    Returns
    -------
    VectorField
        A marker recording the shape. YAPSS replaces it when the class is defined, so it is
        never seen again; its type is what a checker reads the field as.
    """
    if not _is_integer(size):
        msg = f"vector(size) takes a whole number of components; got {size!r}"
        raise TypeError(msg)
    size = int(size)
    if size < 0:
        msg = f"vector(size) must be 0 or more; got {size}"
        raise ValueError(msg)
    return cast("VectorField", Field(size=size))


class PerRow(tuple[Any, ...]):
    """One stored value per row of a block field, kept apart from a single tuple value.

    A bound for a two-row field is stored as a ``(lower, upper)`` pair, and so is a per-row
    pair of bounds, so the two would be indistinguishable as plain tuples.
    """

    __slots__ = ()


def _show_slice(index: slice) -> str:
    """Return a slice as a user would have written it."""
    parts = ["" if part is None else str(part) for part in (index.start, index.stop, index.step)]
    return ":".join(parts[:2] if index.step is None else parts)


class BlockRows(Sequence[Any]):
    """The rows of a block field, addressed by index or by slice.

    A block field of *k* rows is an array of *k* elements, so it is written the way an array
    is: ``bounds.r[:] = (-1, 1)`` gives every row one bound, ``bounds.r[0] = (0, 10)`` gives
    one row its own, and ``bounds.r[:] = [(0, 1), (2, 3), (4, 5)]`` gives each its own. The
    bare name is refused, because it is the only spelling in which a reader cannot see whether
    one value or many was meant.

    Reading behaves as the sequence of stored elements it is, so ``bounds.r[1]`` is row 1's
    bound and ``len(bounds.r)`` is the number of rows.
    """

    __slots__ = ("_name", "_owner", "_rows")

    def __init__(self, rows: tuple[Any, ...], owner: Any, name: str) -> None:
        self._rows = rows
        self._owner = owner
        self._name = name

    def __getitem__(self, index: Any) -> Any:
        """Return the element of one row, or of the rows a slice covers.

        The indices accepted are exactly the ones a sequence accepts; only the failures are
        reworded, to say which setting was indexed and how many rows it has, rather than
        speaking of the tuple the rows happen to be stored in.
        """
        try:
            return self._rows[index]
        except IndexError:
            msg = (
                f"{self._owner._label} '{self._name}'[{index!r}] is out of range for "
                f"{len(self._rows)} rows"
            )
            raise IndexError(msg) from None
        except TypeError:
            msg = (
                f"{self._owner._label} '{self._name}' is read by row, with a whole number or a "
                f"slice; got {index!r}"
            )
            raise TypeError(msg) from None

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Assign to the rows `index` covers. See the class docstring."""
        self._owner._set_field_rows(self._name, index, value)

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    def __eq__(self, other: object) -> bool:
        """Compare equal to any sequence holding the same elements."""
        if isinstance(other, BlockRows):
            return self._rows == other._rows
        if isinstance(other, list | tuple):
            return self._rows == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        """Hash as the tuple of elements it holds."""
        return hash(self._rows)

    def __repr__(self) -> str:
        """Return the rows, as the tuple they read as."""
        return repr(self._rows)


BLOCK_DIMENSIONS = 2
"""The dimensions of an array holding every row of a block field."""

_NO_VALUES: dict[Any, Any] = {}
"""Shared empty store for a read-only vector, which never writes into it."""


def _field_property(name: str, row: int, size: int | None, kind: type[Kind]) -> property:
    """Return the property that reads one field of a generated subclass.

    A field reached through `__getattr__` costs two failed lookups before any of our code runs.
    Reading it through a property on the class costs none of that, and `arg.state.h` is written
    exactly the same way; the subclass is generated per (declaration, kind) anyway, so this is
    where the fields belong.

    The choice between the two by-row readers is made on `size`, not on the row count: a block
    field keeps its leading axis at every size, so ``field(size=1)`` reads as ``(1, npoints)``
    while a field with no size reads as ``(npoints,)``. They are different declarations, and
    deciding on the count alone would silently collapse the one-row block to the scalar's rank.
    """
    default = kind.default
    if not kind.by_row:
        if kind.per_row and size is not None:

            def get_rows(self: Any) -> Any:
                # A block field of a setup aspect is an array of its element, so it is handed
                # back as its rows -- which is what makes `bounds.r[0] = ...` reach the vector.
                return BlockRows(self._elements(name), self, name)

            return property(get_rows)

        def get_whole(self: Any) -> Any:
            value = self._values.get(name, default)
            if value is MISSING:
                msg = f"{type(self)._label} '{name}' has not been assigned"
                raise AttributeError(msg)
            return value

        return property(get_whole)

    if size is None:

        def get_row(self: Any) -> Any:
            source = self._source
            if source is not None:
                return source[row]
            value = self._values.get(row, default)
            if value is MISSING:
                msg = f"{type(self)._label} '{name}' has not been assigned"
                raise AttributeError(msg)
            return value

        return property(get_row)

    rows = size

    def get_block(self: Any) -> Any:
        source = self._source
        if source is not None:
            if type(source) is np.ndarray and (source.ndim > 1 or source.dtype != object):
                return source[row : row + rows]
            if rows == 0:
                # No row of its own to measure, so the shape of a sibling gives the trailing
                # axes: () for endpoint values, (npoints,) over the points. Slicing an array
                # source above gets this for free; a source that is a sequence of rows does not.
                trailing = np.shape(source[0]) if len(source) else ()
                return np.empty((0, *trailing))
            return _stack([source[row + i] for i in range(rows)])
        if rows == 0:
            npoints = type(self)._npoints
            return np.empty((0,) if npoints is None else (0, npoints))
        values = self._values
        block = []
        for i in range(rows):
            value = values.get(row + i, default)
            if value is MISSING:
                msg = f"{type(self)._label} '{name}' has not been assigned"
                raise AttributeError(msg)
            block.append(value)
        return _stack(block)

    return property(get_block)


def _holds_elements(kind: type[Kind], value: object) -> bool:
    """Report whether `value` is a sequence holding at least one element of `kind`."""
    return is_sequence(value) and any(kind.is_element(item) for item in value)


def _is_integer(value: object) -> bool:
    """Report whether `value` is an integer, excluding booleans."""
    if isinstance(value, bool | np.bool_):
        return False
    return isinstance(value, int | np.integer)


def _suggest(name: str, candidates: tuple[str, ...]) -> str:
    """Return a " Did you mean 'x'?" fragment, or an empty string.

    The suggestion is part of the message rather than an exception attribute, so that
    notebooks and IPython, which print the message but not the attribute, still show it.
    """
    close = difflib.get_close_matches(name, candidates, n=1)
    return f" Did you mean '{close[0]}'?" if close else ""


class Vector:
    """Base of every YAPSS vector declaration. Subclass it to declare a vector.

    A subclass lists its fields as `field` markers and nothing else. It is never instantiated
    by the user: YAPSS creates the instances, one per aspect of the vector, and each validates
    its elements according to what that aspect holds.
    """

    _fields: tuple[str, ...] = ()
    _meta: dict[str, Field] = {}  # noqa: RUF012
    _offsets: dict[str, int] = {}  # noqa: RUF012
    _rows: tuple[tuple[str, int | None], ...] = ()
    _nrows: int = 0
    _kind: type[Kind] | None = None
    # Constant for every instance of a generated subclass, so they live on the class and an
    # instance made in a callback sets only what varies.
    _label: str = "vector"
    _aspect: str | None = None
    """The setting an instance holds, such as ``"bounds"``; see `_new`."""
    _role: str | None = None
    """The role this class declares, set by each role base; see `role_of`."""
    _npoints: int | None = None
    _source: Any = None
    _values: dict[Any, Any]
    """The stored values, keyed by field name or by flat row, depending on the kind."""
    _single: dict[str, int] = {}  # noqa: RUF012
    _block: dict[str, tuple[int, int]] = {}  # noqa: RUF012
    _kind_cache: dict[tuple[Any, ...], type[Vector]] = {}  # noqa: RUF012

    def __init_subclass__(cls, *, _generated: bool = False, **kwargs: Any) -> None:
        """Collect the declared fields and fix the flat row layout.

        Parameters
        ----------
        _generated : bool, default False
            Set by YAPSS when it generates a kind-specific subclass, which inherits its
            declaration rather than making one.
        **kwargs
            Passed to `object.__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        if _generated:
            return
        cls._check_bases()
        cls._check_annotations()
        meta: dict[str, Field] = {}
        for name, value in list(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            if not isinstance(value, Field):
                if callable(value):
                    msg = (
                        f"{cls.__name__}.{name} is not a field. A vector declaration names "
                        f"fields and holds no behavior; define {name} outside the class."
                    )
                else:
                    msg = (
                        f"{cls.__name__}.{name} is not a field. A vector declaration holds "
                        f"only fields; write '{name} = yapss.scalar()' or "
                        f"'{name} = yapss.vector(n)'."
                    )
                raise TypeError(msg)
            meta[name] = value
        for name in meta:
            delattr(cls, name)
        offsets: dict[str, int] = {}
        rows: list[tuple[str, int | None]] = []
        for name, spec in meta.items():
            offsets[name] = len(rows)
            if spec.size is None:
                rows.append((name, None))
            else:
                rows.extend((name, i) for i in range(spec.size))
        # The overwhelming majority of reads and writes are of a plain, single-row field of a
        # by-row kind. That case is resolved from this one lookup, without consulting the
        # metadata or splitting anything; everything else takes the general path below.
        cls._single = {name: offsets[name] for name, spec in meta.items() if spec.size is None}
        cls._block = {
            name: (offsets[name], spec.size) for name, spec in meta.items() if spec.size is not None
        }
        cls._fields = tuple(meta)
        cls._meta = meta
        cls._offsets = offsets
        cls._rows = tuple(rows)
        cls._nrows = len(rows)
        cls._kind_cache = {}

    @classmethod
    def _check_bases(cls) -> None:
        for base in cls.__bases__:
            if base is not Vector and issubclass(base, Vector) and base._fields:
                msg = (
                    f"{cls.__name__} cannot inherit from the vector declaration "
                    f"{base.__name__}. Declare its fields directly."
                )
                raise TypeError(msg)

    @classmethod
    def _check_annotations(cls) -> None:
        annotated = [n for n in inspect.get_annotations(cls) if not n.startswith("_")]
        if annotated:
            msg = (
                f"{cls.__name__}.{annotated[0]} is annotated. Fields are declared without "
                f"annotations: write '{annotated[0]} = yapss.scalar()' or "
                f"'{annotated[0]} = yapss.vector(n)'."
            )
            raise TypeError(msg)

    def __init__(self, **values: Any) -> None:
        """Refuse construction: a vector class is a declaration, not a value.

        Parameters
        ----------
        **values
            Ignored; present only so that the mistake gets this message rather than a
            signature error.
        """
        del values
        example = self._fields[0] if self._fields else "x"
        msg = (
            f"{type(self).__name__} is a declaration, not a value. Set fields on the aspect "
            f"that holds them, for example 'phase.state.{example}.bounds = (0, 1)'."
        )
        raise TypeError(msg)

    # -- instance creation, by YAPSS ----------------------------------------------------------

    @classmethod
    def _for(cls, kind: type[Kind], label: str, npoints: int | None = None) -> type[Vector]:
        """Return this declaration's subclass for `kind`, labelled `label`.

        The subclass is generated on first use and cached. Everything constant about an
        instance -- the element check, the label a message uses, and the number of time points
        -- lives on the class, so making one in a callback sets only what actually varies.

        Parameters
        ----------
        kind : type[Kind]
            The element kind the subclass validates against.
        label : str
            What to call instances of it in a message.
        npoints : int, optional
            The number of time points a row must cover, when that is known.

        Returns
        -------
        type[Vector]
            The generated subclass.
        """
        key = (kind, label, npoints)
        cached = cls._kind_cache.get(key)
        if cached is None:
            name = f"{cls.__name__}.{kind.__name__.lower()}"
            namespace: dict[str, Any] = {
                "_kind": kind,
                "_label": label,
                "_npoints": npoints,
            }
            for field_name, spec in cls._meta.items():
                namespace[field_name] = _field_property(
                    field_name,
                    cls._offsets[field_name],
                    spec.size,
                    kind,
                )
            generated: type[Vector] = type(name, (cls,), namespace, _generated=True)
            cls._kind_cache[key] = generated
            cached = generated
        return cached

    @classmethod
    def _new(
        cls, kind: type[Kind], label: str, npoints: int | None = None, aspect: str | None = None
    ) -> Any:
        """Create an instance holding elements of `kind`.

        Parameters
        ----------
        kind : type[Kind]
            What the elements of this instance mean.
        label : str
            What to call this instance in a message, such as ``"phase 'boost' state bounds"``.
        npoints : int, optional
            The number of time points a row must cover, when that is known.
        aspect : str, optional
            The setting this instance holds, such as ``"bounds"``, when it is one. A setting is
            written field first -- ``ph.state.r.bounds[:]`` -- so a message showing the form
            that works has to know it.

        Returns
        -------
        Vector
            A new, empty instance of the kind-specific subclass.
        """
        obj = object.__new__(cls._for(kind, label, npoints))
        object.__setattr__(obj, "_values", {})
        object.__setattr__(obj, "_aspect", aspect)
        return obj

    # -- field access -------------------------------------------------------------------------

    def _kind_or_raise(self) -> type[Kind]:
        kind = type(self)._kind
        if kind is None:
            msg = f"{type(self).__name__} was not created by YAPSS"
            raise TypeError(msg)
        return kind

    def _no_field(self, name: str) -> AttributeError:
        if not self._fields:
            msg = f"{self._label} has no fields: none were declared for it"
            return AttributeError(msg)
        msg = f"{self._label} has no field '{name}'.{_suggest(name, self._fields)}"
        return AttributeError(msg)

    def _read_row(self, row: int, name: str) -> Any:
        source = self._source
        if source is not None:
            return source[row]
        kind = self._kind_or_raise()
        value = self._values.get(row, kind.default)
        if value is MISSING:
            msg = f"{self._label} '{name}' has not been assigned"
            raise AttributeError(msg)
        return value

    # Hidden from type checkers, as `_backend.types_.Protected` hides its own. Both accept
    # any name at runtime and answer for it there; a type checker that can see them stops
    # reporting misspellings on every vector. A declaration's fields are ordinary class
    # attributes, so with these hidden a name that was never declared is flagged where it
    # is written, which is the whole static benefit the API offers.
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            """Return the value of a field. See the class docstring."""
            if name.startswith("_"):
                raise AttributeError(name)
            cls = type(self)
            source = self._source
            if source is not None:
                row = cls._single.get(name)
                if row is not None:
                    return source[row]
            if name not in cls._offsets:
                raise self._no_field(name)
            kind = self._kind_or_raise()
            spec = cls._meta[name]
            if kind.by_row:
                start = cls._offsets[name]
                rows = spec.rows
                if source is not None:
                    # When the rows are laid out in one array -- endpoint values are a contiguous
                    # run of floats, and a stored block is a 2-D array -- the block is a slice of
                    # it, and no array has to be built at all. The slice is read-only, as the
                    # source is.
                    if type(source) is np.ndarray and (source.ndim > 1 or source.dtype != object):
                        return source[start : start + rows]
                    return _stack([source[start + i] for i in range(rows)])
                values = [self._read_row(start + i, name) for i in range(rows)]
                return values[0] if spec.size is None else _stack(values)
            if kind.per_row and spec.size is not None:
                return BlockRows(self._elements(name), self, name)
            value = self._values.get(name, kind.default)
            if value is MISSING:
                msg = f"{self._label} '{name}' has not been assigned"
                raise AttributeError(msg)
            return value

        def __setattr__(self, name, value):
            """Set the value of a field, validating it against this instance's kind."""
            if name.startswith("_"):
                object.__setattr__(self, name, value)
                return
            cls = type(self)
            kind = cls._kind
            if kind is not None and kind.by_row and not kind.read_only:
                row = cls._single.get(name)
                if row is not None:
                    self._values[row] = kind.check(
                        value, label=cls._label, name=name, npoints=cls._npoints
                    )
                    return
                block = cls._block.get(name)
                if block is not None:
                    start, size = block
                    values = self._values
                    label, npoints = cls._label, cls._npoints
                    # One array of the field's own shape is what a block field is nearly always
                    # given, and every row of it is then a row of the right length by construction,
                    # so each row is stored as it stands.
                    if (
                        type(value) is np.ndarray
                        and value.ndim == BLOCK_DIMENSIONS
                        and value.shape[0] == size
                        and (npoints is None or value.shape[1] == npoints)
                    ):
                        for offset in range(size):
                            values[start + offset] = value[offset]
                        return
                    check = kind.check
                    for offset, row_value in enumerate(self._split(value, size, name)):
                        values[start + offset] = check(
                            row_value, label=label, name=name, npoints=npoints
                        )
                    return
            if name not in cls._offsets:
                raise self._no_field(name)
            kind = self._kind_or_raise()
            if kind.read_only:
                raise AttributeError(kind.refusal(self._label, name))
            spec = cls._meta[name]
            if not kind.by_row:
                self._values[name] = self._one_or_per_row(kind, spec, name, value)
                return
            start = cls._offsets[name]
            for i, row_value in enumerate(self._split(value, spec.rows, name)):
                self._values[start + i] = kind.check(
                    row_value, label=self._label, name=name, npoints=self._npoints
                )

        def __delattr__(self, name):
            """Refuse deleting a field's value; assigning a new one is how it is changed."""
            if name.startswith("_"):
                object.__delattr__(self, name)
                return
            kind = type(self)._kind
            if kind is not None and kind.read_only:
                raise AttributeError(kind.refusal(self._label, name, "deleted"))
            msg = f"{type(self)._label} '{name}' cannot be deleted"
            raise AttributeError(msg)

    def _one_or_per_row(self, kind: type[Kind], spec: Field, name: str, value: Any) -> Any:
        """Return the stored value of a scalar setup field, and refuse a block one.

        A scalar field is an array with no shape, so the bare name is the only spelling it has
        and it takes one element. A block field has rows, and is written through them.
        """
        if spec.size is not None:
            # A setting is written field first, `r.bounds[:]`, so the form shown is that one;
            # the bare `r[:]` would name the field alone, which takes no index.
            target = f"{name}.{self._aspect}" if self._aspect else name
            example = f"{target}[:]" if spec.size != 1 else f"{target}[0]"
            msg = (
                f"{self._label} '{name}' has {spec.size} rows, so say which: "
                f"'{example} = ...' gives every row the same, and an index or a slice gives "
                f"rows their own. Assigning it whole is refused because that cannot show which "
                f"was meant."
            )
            raise TypeError(msg)
        self._check_sample_rows(value, name, covered=1, whole=True)
        return kind.check(value, label=self._label, name=name, npoints=self._npoints)

    def _check_sample_rows(self, value: Any, name: str, *, covered: int, whole: bool) -> None:
        """Refuse sampled values whose rows do not match the rows they are assigned to.

        One row of samples is one row's guess, and broadcasts. Several rows are the rows of the
        field, matched by position, so they fit only an assignment covering the whole field
        with as many rows; given to fewer, one of them would be taken without a word.
        """
        if not isinstance(value, Interp) or value.values.ndim == 1:
            return
        given = value.values.shape[0]
        if whole and given == covered:
            return
        target = f"{name}.{self._aspect}" if self._aspect else name
        size = type(self)._meta[name].rows
        msg = (
            f"{self._label} '{name}': the interp values have {given} rows, and the assignment "
            f"covers {covered} of the field's {size}; give one row of samples, or {size} rows "
            f"to '{target}[:]'"
            if size > 1
            else f"{self._label} '{name}': the interp values have {given} rows, and the field "
            f"has one; give one row of samples"
        )
        raise ValueError(msg)

    def _set_field_rows(self, name: str, index: int | slice, value: Any) -> None:
        """Assign to the rows of block field `name` that `index` covers."""
        cls = type(self)
        kind = self._kind_or_raise()
        if kind.read_only:
            raise AttributeError(kind.refusal(self._label, name))
        spec = cls._meta[name]
        count = spec.rows

        if isinstance(index, slice):
            rows = list(range(*index.indices(count)))
            if kind.is_element(value):
                self._check_sample_rows(value, name, covered=len(rows), whole=len(rows) == count)
                values = [value] * len(rows)
            elif not _holds_elements(kind, value):
                # Not one element and not a sequence of them: check it as the one element it
                # was most likely meant to be, so the kind says what is wrong with it -- a
                # boolean, a bound written as one number -- rather than this method guessing.
                kind.check(value, label=self._label, name=name, npoints=self._npoints)
                values = [value] * len(rows)
            else:
                values = list(value)
                if len(values) != len(rows):
                    msg = (
                        f"{self._label} '{name}'[{_show_slice(index)}] covers {len(rows)} "
                        f"rows; got {len(values)} values"
                    )
                    raise ValueError(msg)
        else:
            row = index if index >= 0 else index + count
            if not 0 <= row < count:
                msg = f"{self._label} '{name}' has {count} rows; there is no row {index}"
                raise IndexError(msg)
            if not kind.is_element(value):
                if _holds_elements(kind, value):
                    msg = (
                        f"{self._label} '{name}'[{index}] is one row, so it takes one element, "
                        f"not a sequence of them; got {value!r}"
                    )
                    raise TypeError(msg)
                kind.check(value, label=self._label, name=name, npoints=self._npoints)
            self._check_sample_rows(value, name, covered=1, whole=count == 1)
            rows, values = [row], [value]

        stored = list(self._elements(name))
        for row, row_value in zip(rows, values, strict=True):
            stored[row] = kind.check(row_value, label=self._label, name=name, npoints=self._npoints)
        self._values[name] = PerRow(stored)

    def _elements(self, name: str) -> tuple[Any, ...]:
        """Return one stored element per row of `name`, broadcasting a single value.

        Reads the store rather than the attribute: a block field's attribute is its rows, and
        building those is what calls this.
        """
        cls = type(self)
        rows = cls._meta[name].rows
        kind = self._kind_or_raise()
        value = self._values.get(name, kind.default)
        if value is MISSING:
            msg = f"{self._label} '{name}' has not been assigned"
            raise AttributeError(msg)
        if isinstance(value, PerRow):
            return tuple(value)
        return (value,) * rows

    def _split(self, value: Any, rows: int, name: str) -> list[Any]:
        """Return `rows` row values from `value`, broadcasting a single element."""
        kind = self._kind_or_raise()
        if kind.is_element(value):
            return [value] * rows
        if rows == 1:
            return [value]
        try:
            given = list(value)
        except TypeError:
            msg = f"{self._label} '{name}' needs {rows} rows; got a " f"{type(value).__name__}"
            raise TypeError(msg) from None
        if len(given) != rows:
            msg = f"{self._label} '{name}' needs {rows} rows; got {len(given)}"
            raise ValueError(msg)
        return given

    # -- positional access --------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of flat rows, counting each row of a block field."""
        return type(self)._nrows

    def __iter__(self) -> Any:
        """Iterate over the flat rows, in declaration order."""
        source = self._source
        if source is not None and len(source) == type(self)._nrows:
            return iter(source)
        return (self[i] for i in range(len(self)))

    def _row_index(self, index: int) -> int:
        cls = type(self)
        if not _is_integer(index):
            msg = f"{self._label} indices are integers or slices, not " f"{type(index).__name__}"
            raise TypeError(msg)
        index = int(index)
        if not -cls._nrows <= index < cls._nrows:
            msg = (
                f"{self._label} index {index} is out of range for {cls._nrows} rows "
                f"({', '.join(cls._fields)})"
            )
            raise IndexError(msg)
        return index % cls._nrows

    def __getitem__(self, index: int | slice) -> Any:
        """Return one flat row, or an array of the rows a slice covers."""
        kind = self._kind_or_raise()
        cls = type(self)
        if isinstance(index, slice):
            if not kind.slice_read:
                msg = f"{self._label} cannot be read by slice; read its fields by name"
                raise TypeError(msg)
            start, stop, step = index.indices(cls._nrows)
            source = self._source
            if source is not None and (start, stop, step) == (0, cls._nrows, 1):
                # The rows already are the array YAPSS holds, so hand out a view of it
                # rather than rebuilding one row at a time. The storage is read-only.
                return np.asarray(source)
            return _stack([self[i] for i in range(start, stop, step)])
        row = self._row_index(index)
        name, _ = cls._rows[row]
        if kind.by_row:
            return self._read_row(row, name)
        return getattr(self, name)

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Set one flat row, or the rows a slice covers."""
        kind = self._kind_or_raise()
        cls = type(self)
        if kind.read_only:
            raise AttributeError(kind.refusal(self._label, None))
        if not kind.positional:
            msg = f"{self._label} cannot be set by position; set its fields by name"
            if cls._fields:
                msg = f"{msg}, for example '{cls._fields[0]} = ...'"
            raise TypeError(msg)
        if isinstance(index, slice):
            rows = list(range(*index.indices(cls._nrows)))
            label = (
                f"rows {rows[0]}-{rows[-1]} ({', '.join(cls._rows[r][0] for r in rows)})"
                if rows
                else "an empty slice"
            )
            if kind.is_element(value):
                values: list[Any] = [value] * len(rows)
            else:
                try:
                    values = list(value)
                except TypeError:
                    msg = (
                        f"{self._label}: assigning to {label} needs {len(rows)} values; "
                        f"got a {type(value).__name__}"
                    )
                    raise TypeError(msg) from None
                if len(values) != len(rows):
                    msg = (
                        f"{self._label}: assigning to {label} needs {len(rows)} values; "
                        f"got {len(values)}"
                    )
                    raise ValueError(msg)
            for row, row_value in zip(rows, values, strict=True):
                self._set_row(row, row_value)
            return
        self._set_row(self._row_index(index), value)

    def _set_row(self, row: int, value: Any) -> None:
        kind = self._kind_or_raise()
        name, _ = type(self)._rows[row]
        self._values[row] = kind.check(value, label=self._label, name=name, npoints=self._npoints)

    # -- handing values to and from YAPSS -----------------------------------------------------

    def _row_values(self) -> list[Any]:
        """Return the stored rows in flat order. Only meaningful for a by-row kind."""
        return [self._values[row] for row in range(type(self)._nrows)]

    def _fill(self, rows: Any) -> None:
        """Read the flat rows from `rows`, without copying them. Used by YAPSS for inputs."""
        object.__setattr__(self, "_source", rows)

    # -- completeness -------------------------------------------------------------------------

    def _is_complete(self) -> bool:
        """Report whether every field has a value, without working out which do not.

        Completeness is checked on every call, so it is counted rather than walked: a by-row
        kind is complete exactly when it holds as many rows as it declares. `missing`, which is
        the expensive part, runs only to write the message when this says something is absent.
        """
        kind = type(self)._kind
        if kind is None or kind.default is not MISSING or self._source is not None:
            return True
        return len(self._values) == type(self)._nrows

    def missing(self) -> list[str]:
        """Return the names of the fields that have not been assigned.

        Returns
        -------
        list of str
            The unassigned field names, in declaration order. Always empty for a kind whose
            fields have a default.
        """
        kind = self._kind_or_raise()
        if kind.default is not MISSING:
            return []
        cls = type(self)
        return [
            name
            for name in cls._fields
            if any(cls._offsets[name] + i not in self._values for i in range(cls._meta[name].rows))
        ]

    def __repr__(self) -> str:
        """Return a representation naming the assigned fields."""
        cls = type(self)
        parts = []
        for name in cls._fields:
            try:
                parts.append(f"{name}={getattr(self, name)!r}")
            except AttributeError:
                parts.append(f"{name}=<unset>")
        return f"{cls.__name__}({', '.join(parts)})"


# --------------------------------------------------------------------------------- the roles
#
# A declaration subclasses its role. Each role declares no fields, which is what lets it serve
# as the empty declaration of that role -- the default for a phase with no path constraints is
# `Path` itself -- and it is also what lets a user subclass it, since `_check_bases` refuses
# only a base that has fields of its own.


class State(Vector):
    """Base of a state declaration: the variables a phase's dynamics govern."""

    _role = "state"


class Control(Vector):
    """Base of a control declaration: the variables a phase chooses freely at every point."""

    _role = "control"


class Path(Vector):
    """Base of a path declaration: the constraints a phase holds at every point."""

    _role = "path"


class Integral(Vector):
    """Base of an integral declaration: the quantities a phase accumulates."""

    _role = "integral"


class Parameter(Vector):
    """Base of a parameter declaration: the variables a problem chooses once, for all phases."""

    _role = "parameter"


class Discrete(Vector):
    """Base of a discrete declaration: the constraints on a problem's endpoints and parameters."""

    _role = "discrete"


ROLES: tuple[type[Vector], ...] = (State, Control, Path, Integral, Parameter, Discrete)
"""The six roles, in the order a problem is declared."""


def role_of(cls: type[Vector]) -> str | None:
    """Return the role a declaration was made with, or None if it subclasses `Vector` directly.

    Read off the class hierarchy rather than stored on the declaration, so a declaration cannot
    claim one role and descend from another.
    """
    for role in ROLES:
        if issubclass(cls, role):
            return role._role
    return None


class Maker:
    """Everything constant about making one kind of instance of one declaration.

    Creating an instance in a callback happens once per stencil point per phase, so what does
    not change between calls -- the kind-specific subclass, the label a message would use, and
    the number of time points -- is resolved once, when the solve is set up. A declaration with
    no fields has nothing to hold and nothing that can be written to it, so one instance of it
    is shared.

    Parameters
    ----------
    declaration : type[Vector]
        The vector class to make instances of.
    kind : type[Kind]
        What the elements of those instances mean.
    label : str
        What to call them in a message.
    npoints : int, optional
        The number of time points a row must cover, when that is known.
    """

    __slots__ = ("_cls", "_label", "_npoints", "_shared")

    def __init__(
        self, declaration: type[Vector], kind: type[Kind], label: str, npoints: int | None = None
    ) -> None:
        self._cls = declaration._for(kind, label, npoints)
        self._label = label
        self._npoints = npoints
        self._shared: Any = None
        if self._cls._nrows == 0:
            self._shared = self._build()

    def _build(self) -> Any:
        obj = object.__new__(self._cls)
        object.__setattr__(obj, "_values", {})
        return obj

    def make(self) -> Any:
        """Return a fresh empty instance, to be filled in.

        Returns
        -------
        Vector
            The instance. For a declaration with no fields, one shared instance.
        """
        return self._shared if self._shared is not None else self._build()

    def over(self, rows: Any) -> Any:
        """Return a read-only instance whose rows are read from `rows`.

        Parameters
        ----------
        rows : sequence
            The flat rows, which are read where they are rather than copied.

        Returns
        -------
        Vector
            The instance.
        """
        if self._shared is not None:
            return self._shared
        obj = object.__new__(self._cls)
        object.__setattr__(obj, "_values", _NO_VALUES)
        object.__setattr__(obj, "_source", rows)
        return obj


def symbolic_view(block: Any) -> Any:
    """Return an object-dtype array as an `SXArray`, and anything else unchanged.

    A block of symbols is an object-dtype array, and numpy has no loop for it but the object
    loop, which reports a stale floating-point flag as a spurious `RuntimeWarning: invalid
    value encountered in divide` (numpy issue 21416) whenever a large constant is involved.
    `SXArray` implements `__array_ufunc__` and routes every ufunc through the `yapss.math`
    table instead, so that loop never runs. 0.2.3 made that fix for the released front end
    (its changelog removed the Delta III note that described the warning); anything here that
    builds a block has to keep the view, or the warning comes back.
    """
    return block.view(SXArray) if block.dtype == object else block


def _stack(values: list[Any]) -> Any:
    """Return `values` as a read-only array, stacking rows that are arrays.

    Rows of a block field nearly always already share a shape, and then they can simply be laid
    out, which is several times cheaper than broadcasting them first. Broadcasting is still
    needed when they do not -- a row that is one constant beside rows that vary over the points.

    The array is built for the read, so nothing written into it would reach the field: a
    callback writing ``out.dynamics.r[0] = ...`` or ``arg.state.r[0] = ...`` would lose the
    write without a word. Read-only, it fails at that line, as a write into the slice of a
    stored block does.
    """
    if not values:
        return _read_only(np.empty((0,)))
    first = values[0]
    if isinstance(first, np.ndarray) and first.ndim > 0:
        shape = first.shape
        if all(isinstance(v, np.ndarray) and v.shape == shape for v in values):
            return _read_only(symbolic_view(np.array(values)))
        return _read_only(symbolic_view(np.stack(np.broadcast_arrays(*values))))
    if any(isinstance(v, np.ndarray) and v.ndim > 0 for v in values):
        return _read_only(symbolic_view(np.stack(np.broadcast_arrays(*values))))
    return _read_only(symbolic_view(np.asarray(values)))


def _read_only(array: Any) -> Any:
    """Return `array` with writing turned off."""
    array.flags.writeable = False
    return array
