"""

`Vector`, the single container of the redesigned YAPSS API.

A `Vector` subclass is a *declaration*. It names the members of a state, control, path,
integral, parameter, or discrete-constraint vector, and says how many rows each member has::

    class Rocket(yapss.Vector):
        h = yapss.field(units="ft", latex="h", doc="altitude")
        v = yapss.field(units="ft/s", latex="v", doc="velocity")
        m = yapss.field(units="slug", latex="m", doc="mass")

The user never instantiates it. YAPSS creates every instance, and each one is a *kind*-specific
subclass generated once per (declaration, kind) pair: an instance holding bounds validates its
elements as bounds, one holding a callback's output rows validates them as rows, and so on.
The declaration is reused for every aspect of the vector it names, so a member is spelled the
same way wherever it is reached.

Names address fields; integers and slices address flat rows, so a block field declared with
``field(size=3)`` occupies three consecutive positions.

"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np

from .kinds import MISSING, Kind, is_sequence

__all__ = ["Empty", "Field", "Maker", "Vector", "field"]


class Field:
    """Metadata recorded for one declared field. Created by `field`, never directly.

    Attributes
    ----------
    units : str
        The physical units of the field, as a string. YAPSS does no unit arithmetic; the
        string is used in messages and plot labels.
    latex : str
        A LaTeX fragment labelling the field in plots.
    doc : str
        A one-line description.
    size : int or None
        The number of rows in a block field, or None for a scalar member. A block field keeps
        its leading axis at every size, including one row and none.
    """

    __slots__ = ("doc", "latex", "size", "units")

    def __init__(self, *, units: str, latex: str, doc: str, size: int | None) -> None:
        self.units = units
        self.latex = latex
        self.doc = doc
        self.size = size

    def __repr__(self) -> str:
        """Return a representation naming only the metadata that was given."""
        parts = [f"{n}={getattr(self, n)!r}" for n in ("units", "latex", "doc") if getattr(self, n)]
        if self.size is not None:
            parts.append(f"size={self.size}")
        return f"field({', '.join(parts)})"

    @property
    def rows(self) -> int:
        """int: The number of flat rows the field occupies."""
        return 1 if self.size is None else self.size


def field(*, units: str = "", latex: str = "", doc: str = "", size: int | None = None) -> Any:
    """Declare one field of a `Vector`.

    Parameters
    ----------
    units : str, default ""
        The physical units of the field. YAPSS does no unit arithmetic; the string appears in
        messages and plot labels.
    latex : str, default ""
        A LaTeX fragment labelling the field in plots.
    doc : str, default ""
        A one-line description of the field.
    size : int, optional
        The number of rows, for a field that holds a block of them, such as a position vector.
        Any count from 0 up is allowed, so that a declaration built by an algorithm needs no
        special case at one row or none. Omitting it is not the same as ``size=1``: a field
        with no size is a scalar member, read as ``(npoints,)``, while a block field of one row
        is read as ``(1, npoints)``.

    Returns
    -------
    Any
        A marker recording the metadata. YAPSS replaces it when the class is defined, so it is
        never seen again.
    """
    # Typed as object so that the runtime check, which is for callers who use no
    # annotations at all, is not read as unreachable.
    strings: tuple[tuple[str, object], ...] = (("units", units), ("latex", latex), ("doc", doc))
    for name, value in strings:
        if not isinstance(value, str):
            msg = f"field({name}=) must be a string; got {value!r}"
            raise TypeError(msg)
    if size is not None:
        if not _is_integer(size):
            msg = f"field(size=) must be an integer; got {size!r}"
            raise TypeError(msg)
        size = int(size)
        if size < 0:
            msg = f"field(size=) must be 0 or more; got {size}"
            raise ValueError(msg)
    return Field(units=units, latex=latex, doc=doc, size=size)


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
        """Return the element of one row, or of the rows a slice covers."""
        return self._rows[index]

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
    _npoints: int | None = None
    _source: Any = None
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
                        f"only fields; write '{name} = yapss.field(...)'."
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
                f"annotations: write '{annotated[0]} = yapss.field(...)'."
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
            f"that holds them, for example 'phase.state.bounds.{example} = (0, 1)'."
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
    def _new(cls, kind: type[Kind], label: str, npoints: int | None = None) -> Any:
        """Create an instance holding elements of `kind`.

        Parameters
        ----------
        kind : type[Kind]
            What the elements of this instance mean.
        label : str
            What to call this instance in a message, such as ``"phase 'boost' state bounds"``.
        npoints : int, optional
            The number of time points a row must cover, when that is known.

        Returns
        -------
        Vector
            A new, empty instance of the kind-specific subclass.
        """
        obj = object.__new__(cls._for(kind, label, npoints))
        object.__setattr__(obj, "_values", {})
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

    def __getattr__(self, name: str) -> Any:
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

    def __setattr__(self, name: str, value: Any) -> None:
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
            msg = f"{self._label} is read-only; '{name}' cannot be assigned"
            raise AttributeError(msg)
        spec = cls._meta[name]
        if not kind.by_row:
            self._values[name] = self._one_or_per_row(kind, spec, name, value)
            return
        start = cls._offsets[name]
        for i, row_value in enumerate(self._split(value, spec.rows, name)):
            self._values[start + i] = kind.check(
                row_value, label=self._label, name=name, npoints=self._npoints
            )

    def _one_or_per_row(self, kind: type[Kind], spec: Field, name: str, value: Any) -> Any:
        """Return the stored value of a scalar setup field, and refuse a block one.

        A scalar field is an array with no shape, so the bare name is the only spelling it has
        and it takes one element. A block field has rows, and is written through them.
        """
        if spec.size is not None:
            example = f"{name}[:]" if spec.size != 1 else f"{name}[0]"
            msg = (
                f"{self._label} '{name}' has {spec.size} rows, so say which: "
                f"'{example} = ...' gives every row the same, and an index or a slice gives "
                f"rows their own. The bare name is refused because it cannot show which was "
                f"meant."
            )
            raise TypeError(msg)
        return kind.check(value, label=self._label, name=name, npoints=self._npoints)

    def _set_field_rows(self, name: str, index: int | slice, value: Any) -> None:
        """Assign to the rows of block field `name` that `index` covers."""
        cls = type(self)
        kind = self._kind_or_raise()
        if kind.read_only:
            msg = f"{self._label} is read-only; '{name}' cannot be assigned"
            raise AttributeError(msg)
        spec = cls._meta[name]
        count = spec.rows

        if isinstance(index, slice):
            rows = list(range(*index.indices(count)))
            if kind.is_element(value):
                values = [value] * len(rows)
            else:
                if not is_sequence(value):
                    msg = (
                        f"{self._label} '{name}'[{_show_slice(index)}] covers {len(rows)} "
                        f"rows, so it takes one element for all of them or {len(rows)} of "
                        f"them; got a {type(value).__name__}"
                    )
                    raise TypeError(msg)
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
                msg = (
                    f"{self._label} '{name}'[{index}] is one row, so it takes one element, "
                    f"not a sequence of them; got {value!r}"
                )
                raise TypeError(msg)
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
            msg = f"{self._label} is read-only"
            raise AttributeError(msg)
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


class Empty(Vector):
    """A vector with no fields, for a phase that declares no path or integral."""


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


def _stack(values: list[Any]) -> Any:
    """Return `values` as an array, stacking rows that are arrays.

    Rows of a block field nearly always already share a shape, and then they can simply be laid
    out, which is several times cheaper than broadcasting them first. Broadcasting is still
    needed when they do not -- a row that is one constant beside rows that vary over the points.
    """
    if not values:
        return np.empty((0,))
    first = values[0]
    if isinstance(first, np.ndarray) and first.ndim > 0:
        shape = first.shape
        if all(isinstance(v, np.ndarray) and v.shape == shape for v in values):
            return np.array(values)
        return np.stack(np.broadcast_arrays(*values))
    if any(isinstance(v, np.ndarray) and v.ndim > 0 for v in values):
        return np.stack(np.broadcast_arrays(*values))
    return np.asarray(values)
