"""

Define wrapper classes that allow casadi SX objects to work with numpy arrays.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, Any, ClassVar, cast

# third party imports
import casadi as ca
import numpy as np
from casadi import SX
from numpy.lib.mixins import NDArrayOperatorsMixin

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


# def is_ufunc(name:str) -> bool:
#     """Check if the given name is a numpy ufunc."""
#     obj = getattr(np, name, None)
#     return isinstance(obj, np.ufunc)


def ufunc_num_args(name: str) -> tuple[int, int] | None:
    """Return the number of input and output arguments for a numpy ufunc."""
    obj = getattr(np, name, None)
    if isinstance(obj, np.ufunc):
        return obj.nin, obj.nout
    return None


class SXW(NDArrayOperatorsMixin):
    """Base class for wrapper classes encapsulating casadi SX objects."""

    # __eq__ (inherited from NDArrayOperatorsMixin, via __array_ufunc__ below)
    # returns an elementwise SXW, not a bool -- the same numpy.ndarray convention
    # that makes arrays unhashable. Set explicitly, like numpy does, rather than
    # leaving it implicit.
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    def __init__(self, value: float | SX | SXW):
        while isinstance(value, SXW):
            value = value._value
        self._value: SX = self.convert_value(value)

    def convert_value(self, value: float | SXW) -> SX:
        """Convert the input value to an SX object."""
        return SX(value)

    def __repr__(self) -> str:
        """Return the printable representation of the object."""
        return f"{SXW.__name__}({self._value!r})"

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: SXW | NDArray[np.object_] | float | Any,
        **kwargs: Any,
    ) -> SXW | NDArray[Any] | Any:
        """Implement the array ufunc protocol."""
        if method != "__call__":
            return NotImplemented

        # Prepare the input values for ufunc, unwrapping  SXW instances
        values = []
        for item in inputs:
            if isinstance(item, SXW):
                values.append(item._value)
            elif isinstance(item, np.ndarray) and item.dtype == object:
                values.append(
                    np.array([x._value if isinstance(x, SXW) else x for x in item], dtype=object),
                )
            else:
                values.append(item)

        # Perform the ufunc operation, casting to ensure mypy compatibility
        result = cast(Any, ufunc)(*values, **kwargs)

        # Wrap the result back in  SXW if needed
        if isinstance(result, np.ndarray) and result.dtype == object:
            return np.array([SXW(res) for res in result], dtype=object)
        if isinstance(result, list):
            return np.array([SXW(res) for res in result], dtype=object)
        return SXW(result)

    def __pos__(self) -> SXW:
        """Return the argument (unary plus)."""
        return self

    def __neg__(self) -> SXW:
        """Return the negation of the argument (unary minus)."""
        return SXW(-self._value)

    def _apply_function(self, func: Callable[[SX], SX]) -> SXW:
        return SXW(func(self._value))

    # equality and comparison operators

    def __eq__(self, other: float | SX | SXW) -> SXW:  # type: ignore[override]
        """Return whether the argument is equal to another value."""
        if isinstance(other, SXW):
            return SXW(self._value == other._value)
        return SXW(self._value == other)

    def __ne__(self, other: float | SX | SXW) -> SXW:  # type: ignore[override]
        """Return whether the argument is not equal to another value."""
        if isinstance(other, SXW):
            return SXW(self._value != other._value)
        return SXW(self._value != other)

    def __lt__(self, other: float | SX | SXW) -> SXW:
        """Return whether the argument is less than another value."""
        if isinstance(other, SXW):
            return SXW(self._value < other._value)
        return SXW(self._value < other)

    def __le__(self, other: float | SX | SXW) -> SXW:
        """Return whether the argument is less than or equal to another value."""
        if isinstance(other, SXW):
            return SXW(self._value <= other._value)
        return SXW(self._value <= other)

    def __gt__(self, other: float | SX | SXW) -> SXW:
        """Return whether the argument is greater than another value."""
        if isinstance(other, SXW):
            return SXW(self._value > other._value)
        return SXW(self._value > other)

    def __ge__(self, other: float | SX | SXW) -> SXW:
        """Return whether the argument is greater than or equal to another value."""
        if isinstance(other, SXW):
            return SXW(self._value >= other._value)
        return SXW(self._value >= other)

    def __getattr__(self, name: str) -> Callable[..., SXW]:
        """Intercept calls from numpy ufuncs."""
        num_args = ufunc_num_args(name)
        if num_args and num_args[0] == 1:

            def method(*_args: Any, **_kwargs: Any) -> SXW:
                return self._apply_function(getattr(np, name))

            return method
        msg = f"'{type(self).__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)

    # convert degrees and radians

    def deg2rad(self) -> SXW:
        """Convert degrees to radians."""
        return SXW(self._value * np.pi / 180)

    def rad2deg(self) -> SXW:
        """Convert radians to degrees."""
        return SXW(self._value * 180 / np.pi)

    def radians(self) -> SXW:
        """Convert degrees to radians."""
        return self.deg2rad()

    def degrees(self) -> SXW:
        """Convert radians to degrees."""
        return self.rad2deg()

    # exponential and logarithmic functions

    def exp2(self) -> SXW:
        """Return 2 raised to the power of the argument."""
        return SXW(2**self._value)

    def log2(self) -> SXW:
        """Return the base 2 logarithm of the argument."""
        return SXW(np.log(self._value) / np.log(2))

    # power functions

    def square(self) -> SXW:
        """Return the square of the argument."""
        return self._apply_function(ca.square)

    def cbrt(self) -> SXW:
        """Return the cube root of the argument."""
        value = self._value
        return SXW(ca.sign(value) * ca.fabs(value) ** (1 / 3))

    def reciprocal(self) -> SXW:
        """Return the reciprocal of the argument."""
        return SXW(1 / self._value)

    # conjugate, needed so that std and var work properly

    def conjugate(self) -> SXW:
        """Return the complex conjugate of the argument.

        Just returns the argument, since casadi does not support complex numbers.
        """
        return self

    # miscellaneous functions

    def __abs__(self) -> SXW:
        """Return the absolute value of the argument."""
        value = self._value
        return SXW(ca.sign(value) * value)

    def __floor__(self) -> SXW:
        """Return the float of the argument."""
        return self._apply_function(ca.floor)

    def __ceil__(self) -> SXW:
        """Return the ceiling of the argument."""
        return self._apply_function(ca.ceil)

    def __trunc__(self) -> SXW:
        """Return the argument truncated toward zero."""
        value = self._value
        return SXW(ca.sign(value) * ca.floor(ca.fabs(value)))


class SXArray(np.ndarray[Any, np.dtype[Any]]):
    """Object-dtype ndarray of :class:`SXW` that keeps comparisons symbolic.

    numpy's object-dtype comparison loop coerces each elementwise result to ``bool``.
    :class:`SXW` defines no ``__bool__``, so Python's default makes every comparison
    ``True``, and a gated expression such as ``(abs(y) <= 1) * x`` silently loses its
    mask under the ``"auto"`` derivative method while working correctly under the
    finite-difference methods.

    ``yapss.math`` cannot intercept this: the operator form goes straight to
    :class:`numpy.ndarray`, never through a module-level function. Overriding the
    comparison operators on the array type is the only place it can be caught. CasADi
    represents all of these exactly, so both derivative paths agree once it is.

    Used for the symbolic state, control, and time arrays built in
    ``yapss._private.auto``; the numeric path keeps plain float arrays, whose
    comparisons already behave correctly.
    """

    # ensure the subclass wins reflected operations against plain ndarrays
    __array_priority__ = 20.0

    def _elementwise(self, other: Any, casadi_func: Callable[..., SX]) -> SXArray:
        """Apply a two-argument casadi function elementwise, broadcasting `other`."""
        left, right = np.broadcast_arrays(
            np.asarray(self, dtype=object),
            np.asarray(other, dtype=object),
        )
        result = np.empty(left.shape, dtype=object)
        for index in np.ndindex(result.shape):
            # annotated Any because mypy types ndarray tuple-indexing as ndarray, and so
            # rules out the isinstance narrowing below; the elements really are SXW
            a: Any = left[index]
            b: Any = right[index]
            result[index] = SXW(
                casadi_func(
                    a._value if isinstance(a, SXW) else a,
                    b._value if isinstance(b, SXW) else b,
                ),
            )
        return result.view(SXArray)

    def _elementwise_unary(self, casadi_func: Callable[[SX], SX]) -> SXArray:
        """Apply a one-argument casadi function elementwise."""
        result = np.empty(self.shape, dtype=object)
        for index in np.ndindex(result.shape):
            a: Any = self[index]
            result[index] = SXW(casadi_func(a._value if isinstance(a, SXW) else a))
        return result.view(SXArray)

    # comparison operators

    def __eq__(self, other: Any) -> SXArray:  # type: ignore[override]
        """Return the elementwise symbolic equality."""
        return self._elementwise(other, ca.eq)

    def __ne__(self, other: Any) -> SXArray:  # type: ignore[override]
        """Return the elementwise symbolic inequality."""
        return self._elementwise(other, ca.ne)

    def __lt__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic less-than."""
        return self._elementwise(other, ca.lt)

    def __le__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic less-than-or-equal."""
        return self._elementwise(other, ca.le)

    def __gt__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic greater-than."""
        return self._elementwise(other, ca.gt)

    def __ge__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic greater-than-or-equal."""
        return self._elementwise(other, ca.ge)

    # mask combination: `&`, `|`, and `~`, since `and`, `or`, and `not` cannot be
    # overloaded in Python

    def __and__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical and."""
        return self._elementwise(other, ca.logic_and)

    def __rand__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical and."""
        return self._elementwise(other, ca.logic_and)

    def __or__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical or."""
        return self._elementwise(other, ca.logic_or)

    def __ror__(self, other: Any) -> SXArray:
        """Return the elementwise symbolic logical or."""
        return self._elementwise(other, ca.logic_or)

    def __invert__(self) -> SXArray:
        """Return the elementwise symbolic logical not."""
        return self._elementwise_unary(ca.logic_not)

    # __eq__ is overridden above, so __hash__ must be restated, as for SXW
    __hash__: ClassVar[None] = None


def sx_array(values: Any) -> SXArray:
    """Build an :class:`SXArray` from a sequence of :class:`SXW` values."""
    return np.array(values, dtype=object).view(SXArray)
