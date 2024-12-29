"""

Define wrapper classes that allow casadi SX objects to work with numpy arrays.

"""

# future imports
from __future__ import annotations

# standard imports
from typing import TYPE_CHECKING, Any, Callable, cast

# third party imports
import casadi as ca
import numpy as np
from casadi import SX
from numpy.lib.mixins import NDArrayOperatorsMixin

if TYPE_CHECKING:
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
        """Return the exponential of the argument."""
        return SXW(2 ** np.log(self._value))

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
        return SXW(ca.sign(value) * ca.abs(value) ** (1 / 3))

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
