"""

What the transcription is handed to call: the user's functions, and their auxiliary data.

`UserFunctions` is a field of `ProblemSpec`, so it is the shape both front ends fill in --
the 0.3.0 `Problem` assigns to it directly, and the redesigned API builds one in
`yapss._next.compile`. `Callback` is the descriptor that guards each slot, and `Auxdata` is
the namespace a 0.3.0 problem carries through to its callbacks.

"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from .types_ import Protected, set_private

if TYPE_CHECKING:

    from .input_args import (
        ContinuousFunction,
        ContinuousHessianFunction,
        ContinuousJacobianFunction,
        DiscreteFunction,
        DiscreteHessianFunction,
        DiscreteJacobianFunction,
        ObjectiveFunction,
        ObjectiveGradientFunction,
        ObjectiveHessianFunction,
    )

F = TypeVar("F")


class Auxdata(SimpleNamespace):
    """Auxiliary problem data, which can be anything."""


class Callback(Generic[F]):
    """A `UserFunctions` slot: a callable taking exactly one argument, or None."""

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        """Record the attribute name."""
        self.name = name

    def __get__(self, instance: UserFunctions | None, owner: type) -> F | None:
        """Return the callback, or None if it has not been set."""
        if instance is None:
            return self  # type: ignore[return-value]
        return cast("F | None", instance.__dict__.get("_" + self.name))

    def __set__(self, instance: UserFunctions, value: F | None) -> None:
        """Set the callback after checking it can be called with one argument."""
        if value is not None:
            msg = f"Value of '{self.name}' must be a callable object with one argument, or None."
            if not callable(value):
                raise TypeError(msg)
            # the question is whether it can be *called* with one argument, not how many
            # parameters it has: extra parameters with defaults, and *args/**kwargs, are
            # all fine. `bind` answers exactly that.
            signature = inspect.signature(value)
            try:
                signature.bind(None)
            except TypeError:
                msg = (
                    f"Value of '{self.name}' must be a callable object with one argument, "
                    f"or None; {getattr(value, '__name__', value)}{signature} cannot be "
                    f"called with one."
                )
                raise TypeError(msg) from None
        set_private(instance, "_" + self.name, value)

    def __delete__(self, instance: UserFunctions) -> None:
        """Refuse deletion, saying how to unset a callback."""
        msg = f"cannot delete 'UserFunctions' attribute '{self.name}'; set to None instead"
        raise AttributeError(msg)


class UserFunctions(Protected):
    """Container for the user-defined callback functions and their derivatives.

    The `functions` attribute of a `Problem` instance is an instance of the `UserFunctions`
    class, which stores the user-defined callback functions and their derivatives. Every
    optimal control problem must have at least an objective function. Most problems will have
    one or more phases with dynamics, path constraints, and/or integrands, and these problems
    require at least a `continuous` callback function. Problems with discrete constraints
    require at least a `discrete` callback function.

    For problems that use automatic differentiation, or differentiation by finite differences,
    no further callbacks are required. For problems that use user-supplied derivatives,
    additional callback functions are required. The `objective_gradient` callback is required
    for problems that use user-supplied gradients, and the `objective_hessian` callback is
    required for problems that use user-supplied Hessians. The `continuous_jacobian`,
    `continuous_hessian`, `discrete_jacobian`, and `discrete_hessian` callbacks are required
    as appropriate for problems that use user-supplied derivatives.

    Attributes
    ----------
    objective : ObjectiveFunction | None
    continuous : ContinuousFunction | None
    discrete : DiscreteFunction | None
    objective_gradient : ObjectiveGradientFunction | None
    continuous_jacobian : ContinuousJacobianFunction | None
    discrete_jacobian : DiscreteJacobianFunction | None
    objective_hessian : ObjectiveHessianFunction | None
    continuous_hessian : ContinuousHessianFunction | None
    discrete_hessian : DiscreteHessianFunction | None
    """

    objective: Callback[ObjectiveFunction] = Callback()
    objective_gradient: Callback[ObjectiveGradientFunction] = Callback()
    objective_hessian: Callback[ObjectiveHessianFunction] = Callback()
    continuous: Callback[ContinuousFunction] = Callback()
    continuous_jacobian: Callback[ContinuousJacobianFunction] = Callback()
    continuous_hessian: Callback[ContinuousHessianFunction] = Callback()
    discrete: Callback[DiscreteFunction] = Callback()
    discrete_jacobian: Callback[DiscreteJacobianFunction] = Callback()
    discrete_hessian: Callback[DiscreteHessianFunction] = Callback()


# The point above which `LargeSegmentWarning` suggests splitting a segment. It is a
# judgement, not a cliff: nothing fails at 26 points. The figure is set well above
# published practice and well below where the cost becomes painful.
#
# Published hp-adaptive methods cap the degree per interval far lower: the method is
# parameterized as hp-Method(Nmin, Nmax) with "a user-specified upper limit Nmax >= 2 ...
# to prevent the polynomial degree from growing unreasonably large", and GPOPS-II's
# examples use ph-(4, 10) -- a maximum of 10 (Darby, Hager and Rao, "An hp-adaptive
# pseudospectral method for solving optimal control problems", Optimal Control
# Applications and Methods 32, 2011; Patterson and Rao, "GPOPS-II", ACM TOMS 41, 2014).
# Conditioning is the milder constraint: the first-derivative differentiation matrix
# conditions as O(N^2), so N = 100 costs about four digits, which double precision
# absorbs.
#
# What bites in YAPSS is the mesh setup. `quadrature.py` computes the nodes with mpmath,
# memoized per (method, count), and the cost grows quadratically: measured on an M-series
# Mac, LGL takes 0.02 s at 10 points, 0.03 s at 15, 0.08 s at 25, 0.32 s at 50, 1.2 s at
# 100, and 21 s at 400.
