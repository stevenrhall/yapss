"""

A NumPy array that checks every value written into it where it is written.

Bounds and scale factors are stored as arrays and handed to the user as they are, so that
``problem.bounds.phase[0].state.lower[1] = 0.0`` and ``scale.phase[0].state *= 2`` work as
they would on any array. A plain ndarray cannot refuse such a write, so a NaN bound or a zero
scale factor written that way was reported only later, by ``validate()``. `CheckedArray`
refuses it at the user's line.

The rules, decided 2026-09-16:

- **Idiomatic writes are checked where they are made:** element, slice, and mask assignment
  (`__setitem__`), in-place operators and ufuncs with ``out=`` (`__array_ufunc__`), and
  `fill`. A value is checked for its type (a real number: no str, bool, complex, or None)
  and then by the array's `Check`, which knows what the attribute allows.
- **A refused write changes nothing.** The write is made on a copy, the copy is checked, and
  only then is the result stored.
- **Only an array that writes into the problem is checked.** A view (``a[1:]``,
  ``a.reshape(...)``) writes into the stored array and keeps the checks; a copy (``copy()``,
  ``astype``, indexing with a list or a mask, ``np.sort``, arithmetic) is the user's own and
  is a plain ndarray. A checked array prints like a plain one.
- **Unidiomatic writes are left to validate():** ``np.copyto``, ``np.put``,
  ``a.flat[...] = ``, ``a.view(np.ndarray)[...] = ``, and ``ufunc.at``. They reach the stored
  values without the checks, and ``validate()`` reports what it can. A bool written that
  way is converted by NumPy and cannot be detected afterward.
- **A copied or unpickled problem keeps its checks** (provisional, pending the design of
  `Solution`): `copy.deepcopy` and `pickle` preserve them.

Checks relating several values, such as a lower bound above its upper bound, belong to
``validate()``: the user must be free to assign ``lower`` and ``upper`` in either order.
"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, Any, Protocol, cast

# third party imports
import numpy as np

from .coercion import real_array, real_scalar

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

__all__ = ["Check", "CheckedArray", "raise_if_invalid"]


class Check(Protocol):
    """What one attribute allows, element by element.

    Implementations are module-level classes so that a checked array can be pickled.
    """

    def invalid(self, values: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Return a mask of the elements the attribute does not allow."""
        ...

    def describe(self, value: float) -> str:
        """Say what is wrong with one refused value, as the end of a sentence."""
        ...


def raise_if_invalid(
    check: Check,
    label: str,
    values: NDArray[np.float64],
    *,
    among: NDArray[np.bool_] | None = None,
    indexed: bool = True,
) -> None:
    """Raise `ValueError` naming the first element of ``values`` the check refuses.

    Parameters
    ----------
    check : Check
        What the attribute allows.
    label : str
        The attribute as the user spells it, such as ``bounds.phase[0].state.lower``.
    values : NDArray[np.float64]
        The values to check.
    among : NDArray[np.bool_], optional
        Check only these elements: the ones a write changes, so that a value already stored
        (by a write around the checks) is left for ``validate()`` to report.
    indexed : bool, default True
        Whether an index into ``values`` is an index into the attribute. It is not for a
        write through a view, whose message names the attribute alone.
    """
    bad = check.invalid(values)
    if among is not None:
        bad &= among
    if not np.any(bad):
        return
    i = int(np.flatnonzero(bad.reshape(-1))[0])
    value = float(values.reshape(-1)[i])
    where = f"{label}[{i}]" if indexed else f"an element of {label}"
    msg = f"{where} {check.describe(value)}"
    raise ValueError(msg)


def _changed(new: NDArray[np.float64], old: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Return a mask of the elements that differ, counting NaN as equal to NaN."""
    changed: NDArray[np.bool_] = ~((new == old) | (np.isnan(new) & np.isnan(old)))
    return changed


class CheckedArray(np.ndarray):
    """A float64 array whose idiomatic writes are checked, with the problem's arrays as roots.

    Create one with `CheckedArray.create`. The array created is a *root*: the stored array
    of one attribute. Views of it share its label, check, and root, and are checked; any
    other array derived from it is returned as a plain ndarray.
    """

    # slots: an attribute assigned by mistake (`lower.dummy = 1`) raises, as on a plain ndarray
    __slots__ = ("_check", "_label", "_root")

    _label: str | None
    _check: Check | None
    _root: CheckedArray | None

    @classmethod
    def create(cls, values: NDArray[np.float64], label: str, check: Check) -> CheckedArray:
        """Return a new root holding a copy of ``values``, which must already be valid."""
        array = np.array(values, dtype=np.float64).view(cls)
        array._label = label
        array._check = check
        array._root = array
        return array

    def __array_finalize__(self, obj: Any) -> None:
        """Carry the label, check, and root to an array derived from a checked one."""
        self._label = getattr(obj, "_label", None)
        self._check = getattr(obj, "_check", None)
        self._root = getattr(obj, "_root", None)

    # ------------------------------------------------------------ views and copies

    def _is_checked(self) -> bool:
        return self._root is not None and np.shares_memory(self, self._root)

    @staticmethod
    def _plain(result: Any) -> Any:
        """Return ``result`` as a plain ndarray unless it writes into a checked root."""
        if isinstance(result, CheckedArray) and not result._is_checked():
            return result.view(np.ndarray)
        if isinstance(result, tuple):
            return tuple(CheckedArray._plain(item) for item in result)
        if isinstance(result, list):
            return [CheckedArray._plain(item) for item in result]
        return result

    def __getitem__(self, key: Any) -> Any:
        return self._plain(super().__getitem__(key))

    def copy(self, order: Any = "C") -> NDArray[np.float64]:  # type: ignore[override]
        """Return a copy, as a plain ndarray: a copy is the user's own."""
        return super().copy(order=order).view(np.ndarray)

    def __copy__(self) -> Any:
        return self.copy()

    def astype(self, *args: Any, **kwargs: Any) -> Any:
        """Convert, returning a plain ndarray unless the result is this array itself."""
        return self._plain(super().astype(*args, **kwargs))

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        return self._plain(super().__array_function__(func, types, args, kwargs))

    # --------------------------------------------------------------------- writes

    def __setitem__(self, key: Any, value: Any) -> None:
        if not self._is_checked():
            super().__setitem__(key, value)
            return
        assert self._label is not None
        assert self._check is not None
        # the type first: after NumPy has written True or "1" into a float array, it is 1.0
        converted = real_array(value, self._label)
        current = self.view(np.ndarray)
        trial = current.copy()
        trial[key] = converted
        raise_if_invalid(
            self._check,
            self._label,
            trial,
            among=_changed(trial, current),
            indexed=self is self._root,
        )
        super().__setitem__(key, converted)

    def fill(self, value: Any) -> None:
        """Fill the array with one value, checked first."""
        if self._is_checked():
            assert self._label is not None
            assert self._check is not None
            trial = np.full(self.shape, real_scalar(value, self._label))
            raise_if_invalid(self._check, self._label, trial, indexed=self is self._root)
        super().fill(value)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        out: tuple[Any, ...] | None = None,
        **kwargs: Any,
    ) -> Any:
        plain = [x.view(np.ndarray) if isinstance(x, CheckedArray) else x for x in inputs]
        if out is None:
            return getattr(ufunc, method)(*plain, **kwargs)
        # An output that writes into the problem is computed on a copy, every such copy is
        # checked, and only then is anything stored: a refused operation changes nothing.
        targets = [o.view(np.ndarray) if isinstance(o, CheckedArray) else o for o in out]
        checked = [isinstance(o, CheckedArray) and o._is_checked() for o in out]
        trials = tuple(
            t.copy() if is_checked else t for t, is_checked in zip(targets, checked, strict=True)
        )
        getattr(ufunc, method)(*plain, out=trials, **kwargs)
        for o, target, trial, is_checked in zip(out, targets, trials, checked, strict=True):
            if is_checked:
                raise_if_invalid(
                    o._check,
                    o._label,
                    trial,
                    among=_changed(trial, target),
                    indexed=o is o._root,
                )
        for target, trial, is_checked in zip(targets, trials, checked, strict=True):
            if is_checked:
                target[...] = trial
        return out[0] if len(out) == 1 else out

    # ------------------------------------------------------------ copying a problem

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Any:
        """Copy a root with its checks: deep copies are how a whole problem is copied."""
        if self._root is not self or self._label is None or self._check is None:
            return self.copy()
        return CheckedArray.create(self.view(np.ndarray), self._label, self._check)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle a root with its label and check."""
        reconstruct, arguments, state = cast("tuple[Any, Any, Any]", super().__reduce__())
        root = self._root is self
        return reconstruct, arguments, (state, self._label if root else None, self._check)

    def __setstate__(self, state: Any) -> None:
        array_state, label, check = state
        super().__setstate__(array_state)
        if label is not None:
            self._label = label
            self._check = check
            self._root = self

    # ------------------------------------------------------------------- printing

    def __repr__(self) -> str:
        return repr(self.view(np.ndarray))

    def __str__(self) -> str:
        return str(self.view(np.ndarray))
