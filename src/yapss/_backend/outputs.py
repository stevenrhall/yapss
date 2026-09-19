"""

Callback outputs: read-only arrays whose writes assign whole rows.

A continuous output (``arg.phase[p].dynamics``, ``integrand``, ``path``) is a stack of rows, one
per component, each with a value at every evaluation point; ``arg.discrete`` is a stack of
scalar rows, one per discrete constraint. The array a callback receives is read-only; the only
way to change it is to assign whole rows:

- ``out[:] = rows``, ``out[i:j] = rows``: a tuple or list with one value per selected row, a
  2-D array with one row per selected row, or a scalar constant for every selected row;
- ``out[i] = value`` (negative ``i`` counts from the end): a row value -- a scalar constant, an
  expression over the points, or a list with one value per point;
- ``out[i] += value`` and the other in-place operators, on a row or on the whole output, once
  the row has been assigned.

Anything else -- a write into part of a row (``out[i][k] = ...``), a column (``out[:, k] =
...``), a mask or fancy index, ``np.copyto``, or a ufunc's ``out=`` -- is refused at the line,
and so is a value whose shape fits only by broadcasting: an expression over the points assigned
to several rows, or a length-1 array over several points.

An output carries no record of which rows were written, because nothing during a solve asks.
The one question ever asked -- did the callback assign every row at the initial guess? -- is
answered once, in `yapss._backend.setup_check`, from the values themselves: the outputs are
blanked to NaN before each call under the float methods, so a row that was never assigned is
NaN, and the scan for non-finite values that runs there anyway finds it. That is why an
in-place operator on a row that has not been assigned reads as unassigned: it is.
"""

# future imports
from __future__ import annotations

# standard imports
import operator
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

# third party imports
import numpy as np

# package imports
from yapss.math.wrapper import SXW

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Callable

    # third party imports
    from numpy.typing import NDArray

__all__ = ["Output", "OutputArray", "OutputRow"]

T = TypeVar("T", bound=np.generic)

WHOLE_ROWS = (
    "assign whole rows, as out[i] = value, out[i:j] = (row, ...), or out[:] = (row, ...), "
    "where a row value is a constant, an expression over the points, or a list with one value "
    "per point"
)

IN_PLACE = {
    "__iadd__": operator.add,
    "__isub__": operator.sub,
    "__imul__": operator.mul,
    "__itruediv__": operator.truediv,
    "__ifloordiv__": operator.floordiv,
    "__imod__": operator.mod,
    "__ipow__": operator.pow,
}


def _is_scalar(value: Any) -> bool:
    """Whether a value is one number or one symbol.

    Bools count as numbers: `yapss.math.all` and `any` return numpy bools on floats and 0/1
    symbols under ``"auto"``, so refusing them would make an output depend on the method.
    """
    if isinstance(value, (bool, int, float, np.generic, SXW)):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0


def _is_row_index(index: Any) -> bool:
    return isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))


def _plain(value: Any) -> Any:
    """Replace output arrays, at any depth in tuples and lists, by plain ndarray views."""
    if isinstance(value, np.ndarray) and isinstance(value, _Protected):
        return value.view(np.ndarray)
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _show(index: Any) -> str:
    """Spell an index as it appears in source."""
    if isinstance(index, slice):
        text = ":".join("" if part is None else str(part) for part in (index.start, index.stop))
        return text if index.step is None else f"{text}:{index.step}"
    if isinstance(index, tuple):
        return ", ".join(_show(item) for item in index)
    return "..." if index is Ellipsis else repr(index)


class _Protected:
    """Array behavior shared by an output and its rows.

    Arithmetic on them yields plain arrays, and ``out=`` and ``np.copyto`` cannot write into
    them.
    """

    if TYPE_CHECKING:

        def _describe(self) -> str: ...

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        for target in kwargs.get("out", ()):
            if isinstance(target, _Protected):
                msg = f"cannot write into {target._describe()} with out=: {WHOLE_ROWS}"
                raise TypeError(msg)
        return getattr(ufunc, method)(*_plain(inputs), **kwargs)

    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        if func is np.copyto:
            target = args[0] if args else kwargs.get("dst")
            if isinstance(target, _Protected):
                msg = f"cannot copy into {target._describe()}: {WHOLE_ROWS}"
                raise TypeError(msg)
        return func(*_plain(args), **{key: _plain(item) for key, item in kwargs.items()})


class OutputArray(_Protected, np.ndarray[Any, np.dtype[T]], Generic[T]):
    """The rows of one callback output, read-only except through whole-row assignment.

    Create with `OutputArray.zeros`. A view of an output (``out[:, 0]``) is read-only and
    refuses writes; an array computed from an output (``2 * out``, ``out.copy()``) is ordinary
    data.
    """

    _storage: NDArray[T]
    _label: str
    _count: str
    _scalar_rows: bool
    _is_output: bool = False
    _view_of: str | None = None

    @classmethod
    def zeros(
        cls, shape: tuple[int, ...], dtype: type[T], label: str, count: str
    ) -> OutputArray[T]:
        """Return an output of zeros.

        Parameters
        ----------
        shape : tuple[int, ...]
            ``(rows, points)`` for a continuous output, ``(rows,)`` for discrete constraints.
        dtype : type
        label : str
            The output as the user spells it, such as ``arg.phase[0].path``.
        count : str
            The row count as the user sets it, such as ``nh = 2 in phase 0``.
        """
        storage: NDArray[T] = np.zeros(shape, dtype=dtype)
        output = cast("OutputArray[T]", storage.view(cls))
        output._storage = storage
        output._label = label
        output._count = count
        output._scalar_rows = len(shape) == 1
        output._is_output = True
        output.flags.writeable = False
        return output

    def __array_finalize__(self, obj: Any) -> None:
        """Mark a view of an output as part of it; anything else is ordinary data."""
        self._is_output = False
        shares = isinstance(obj, np.ndarray) and np.shares_memory(self, obj)
        self._view_of = (
            (obj._label if obj._is_output else obj._view_of)
            if shares and isinstance(obj, OutputArray)
            else None
        )

    def _describe(self) -> str:
        return self._label if self._is_output else (self._view_of or "an output")

    # ----------------------------------------------------------------------- reads
    def blank(self, value: Any) -> None:
        """Set every value to `value`, which a callback is expected to overwrite."""
        self._storage.fill(value)

    def __getitem__(self, index: Any) -> Any:
        """Read; a row of a continuous output is returned as a read-only `OutputRow`."""
        if self._is_output and _is_row_index(index):
            row = self._row(index)
            if self._scalar_rows:
                return self._storage[row]
            view = self._storage[row].view(OutputRow)
            view._output = self
            view._row = row
            view.flags.writeable = False
            return view
        return super().__getitem__(index)

    # ---------------------------------------------------------------------- writes
    def __setitem__(self, index: Any, value: Any) -> None:  # type: ignore[override, unused-ignore]
        """Assign whole rows."""
        if not self._is_output:
            if self._view_of is None:
                self.view(np.ndarray)[index] = value
                return
            msg = f"cannot assign part of {self._view_of}: {WHOLE_ROWS}"
            raise TypeError(msg)
        if value is self:  # the descriptor's assignment after an in-place operator
            return
        if _is_row_index(index):
            row = self._row(index)
            self._storage[row] = self._row_value(value, index)
            return
        if isinstance(index, slice):
            rows = range(*index.indices(self.shape[0]))
            storage = self._storage
            if type(value) in (tuple, list) and len(value) == len(rows):
                # the common case: one value per row, each checked without building messages
                for row, item in zip(rows, value, strict=True):
                    storage[row] = self._row_value(item, row)
            else:
                for row, row_value in zip(rows, self._rows_value(value, rows, index), strict=True):
                    storage[row] = row_value
            return
        msg = f"cannot assign {self._label}[{_show(index)}]: {WHOLE_ROWS}"
        raise TypeError(msg)

    def _row(self, index: int) -> int:
        n = self.shape[0]
        if not -n <= index < n:
            bound = "it has no rows" if n == 0 else f"the row index must be in range({n})"
            msg = f"cannot assign {self._label}[{index}]: {bound} ({self._count})"
            raise IndexError(msg)
        return int(index) % int(n)

    def _where(self, index: Any) -> str:
        return f"{self._label}[{_show(index)}]"

    def _row_value(self, value: Any, index: Any) -> Any:
        """Check one row's value; return it ready to store."""
        kind = type(value)
        if (
            kind is float
            or kind is np.float64
            or (kind is np.ndarray and value.shape == self.shape[1:])
        ):
            return value  # the common cases, first
        if _is_scalar(value):
            return value
        if self._scalar_rows:
            # a scalar slot takes one value, not an array of one, as NumPy and bounds require
            slot = self._where(index)
            row = int(index) % self.shape[0]
            got = (
                f"an array of shape {value.shape}"
                if isinstance(value, np.ndarray)
                else f"a {type(value).__name__}"
            )
            msg = (
                f"cannot assign {slot}: expected one value, got {got}; use {slot} = value[0], "
                f"or a slice such as {self._label}[{row}:{row + 1}] = value"
            )
            raise ValueError(msg)
        points = self.shape[1]
        where = self._where
        if isinstance(value, (list, tuple)):
            if len(value) == points:
                return value
            if len(value) == 1:
                msg = (
                    f"cannot assign {where(index)}: a list with one value does not fill the "
                    f"{points} points; assign the value itself for a constant row"
                )
            else:
                msg = (
                    f"cannot assign {where(index)}: expected {points} values, one per point, "
                    f"got {len(value)}"
                )
            raise ValueError(msg)
        array = np.asarray(_plain(value))
        if array.shape == (points,):
            return array
        if array.shape == (1,):
            msg = (
                f"cannot assign {where(index)}: a length-1 array does not fill the {points} "
                "points; assign a scalar for a constant row"
            )
        else:
            msg = (
                f"cannot assign {where(index)}: expected a row of {points} points, "
                f"got shape {array.shape}"
            )
        raise ValueError(msg)

    def _rows_value(self, value: Any, rows: range, index: Any) -> list[Any]:
        """Check the value for several rows; return one stored value per row."""
        where = self._where(index)
        count = len(rows)
        if _is_scalar(value):
            return [value] * count
        if isinstance(value, (list, tuple)):
            if len(value) != count:
                got = len(value)
                msg = f"cannot assign {where}: expected {count} rows, got {got} ({self._count})"
                raise ValueError(msg)
            return [self._row_value(item, row) for row, item in zip(rows, value, strict=True)]
        array = np.asarray(_plain(value))
        expected = (count,) if self._scalar_rows else (count, self.shape[1])
        if array.shape == expected:
            return list(array)
        if count == 1 and array.ndim == 1:  # one row selected: the expression is that row
            return [self._row_value(array, rows[0])]
        if count == 0:
            msg = f"cannot assign {where}: it selects no rows ({self._count})"
        elif array.ndim == 1 and not self._scalar_rows:
            msg = (
                f"cannot assign {where}: one expression over the points cannot fill {count} "
                f"rows; give one value per row, as (row0, row1, ...) ({self._count})"
            )
        else:
            msg = f"cannot assign {where}: expected shape {expected}, got {array.shape}"
        raise ValueError(msg)

    def _in_place(self, op: Callable[[Any, Any], Any], value: Any) -> OutputArray[T]:
        if not self._is_output:
            msg = f"cannot assign part of {self._describe()}: {WHOLE_ROWS}"
            raise TypeError(msg)
        rows = range(self.shape[0])
        for row, row_value in zip(rows, self._rows_value(value, rows, slice(None)), strict=True):
            self._storage[row] = op(self._storage[row], row_value)
        return self


class OutputRow(_Protected, np.ndarray[Any, np.dtype[Any]]):
    """One row of a continuous output, read-only; in-place operators assign the whole row."""

    _output: OutputArray[Any] | None = None
    _row: int = 0

    def __array_finalize__(self, obj: Any) -> None:
        """Mark a slice of a row as part of it; anything else is ordinary data."""
        self._part_of = (
            obj._describe()
            if isinstance(obj, OutputRow)
            and (obj._output is not None or obj._part_of is not None)
            and np.shares_memory(self, obj)
            else None
        )
        self._output = None

    _part_of: str | None = None

    def _describe(self) -> str:
        if self._output is not None:
            return f"{self._output._label}[{self._row}]"
        return self._part_of or "an output row"

    def __setitem__(self, index: Any, value: Any) -> None:  # type: ignore[override, unused-ignore]
        """Refuse a write into part of a row."""
        if self._output is None and self._part_of is None:
            self.view(np.ndarray)[index] = value
            return
        row = self._describe()
        msg = (
            f"cannot assign part of {row}: assign the whole row, as "
            f"{row} = [f(x[k]) for k in range(n)] or {row} = f(x)"
        )
        raise TypeError(msg)

    def _in_place(self, op: Callable[[Any, Any], Any], value: Any) -> Any:
        if self._output is None:
            if self._part_of is not None:
                msg = f"cannot assign part of {self._describe()}: {WHOLE_ROWS}"
                raise TypeError(msg)
            return op(self.view(np.ndarray), value)
        output, row = self._output, self._row
        output[row] = op(self.view(np.ndarray), _plain(value))
        return output[row]


def _in_place_method(name: str, op: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def method(self: Any, value: Any) -> Any:
        return self._in_place(op, value)

    method.__name__ = name
    return method


for _name, _op in IN_PLACE.items():
    setattr(OutputArray, _name, _in_place_method(_name, _op))
    setattr(OutputRow, _name, _in_place_method(_name, _op))


class Output(Generic[T]):
    """Descriptor for a continuous output, stored in the instance's ``_outputs`` dictionary.

    Assigning the attribute (``arg.phase[0].dynamics = rows``) assigns every row.
    """

    name: str

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Record the output name."""
        self.name = name

    def __get__(self, instance: Any, owner: type[Any]) -> OutputArray[T]:
        """Return the output array."""
        return instance._outputs[self.name]  # type: ignore[no-any-return]

    def __set__(self, instance: Any, value: Any) -> None:
        """Assign every row of the output."""
        instance._outputs[self.name][:] = value
