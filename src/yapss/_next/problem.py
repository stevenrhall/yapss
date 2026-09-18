"""

The problem: declarations in, a solution out.

Everything about the problem lives on it -- its phases, its parameters and discrete
constraints, its objective, and the settings that say how it is to be solved -- and
``solve()`` takes no arguments. Each solve first takes a snapshot, so editing a problem
afterwards never alters what an earlier solution recorded.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yapss._private.ipopt_options import IpoptOptions

from .compile import solve_problem
from .containers import Container, is_callable, is_string, is_subclass
from .declare import Phases
from .kinds import Bounds, ScalarGuess, Scale
from .spec import snapshot, validate_problem
from .vector import Empty, Vector

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Problem"]

METHODS = ("lgl", "lgr", "lg")
DERIVATIVE_METHODS = ("auto", "central-difference", "central-difference-full", "user")
ORDERS = ("first", "second")
SENSES = ("minimize", "maximize")
CATCH_KEYBOARD_INTERRUPT = True
"""Whether a keyboard interrupt stops the solve cleanly, by default."""


def _one_of(value: Any, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        msg = f"{label} must be one of {', '.join(allowed)}; got {value!r}"
        raise ValueError(msg)
    return str(value)


class ObjectiveAspects(Container):
    """The objective: whether it is minimized or maximized, and how large it typically is."""

    _settable = ("sense", "scale")

    def __init__(self) -> None:
        self._label = "objective"
        self._hold("sense", "minimize")
        self._hold("scale", 1.0)

    def _check(self, name: str, value: Any) -> Any:
        if name == "scale":
            return Scale.check(value, label="objective", name="scale", npoints=None)
        return _one_of(value, SENSES, "objective.sense")


class DiscreteAspects(Container):
    """The discrete constraints: their bounds and their scales."""

    _held = ("bounds", "scale")


class ParameterAspects(Container):
    """The problem's parameters: their bounds, their guesses, and their scales."""

    _held = ("bounds", "guess", "scale")


class Derivatives(Container):
    """How derivatives are computed."""

    _settable = ("method", "order")

    def __init__(self) -> None:
        self._label = "derivatives"
        self._hold("method", "auto")
        self._hold("order", "second")

    def _check(self, name: str, value: Any) -> Any:
        if name == "method":
            return _one_of(value, DERIVATIVE_METHODS, "derivatives.method")
        return _one_of(value, ORDERS, "derivatives.order")


class Problem(Container):
    """An optimal control problem.

    Parameters
    ----------
    name : str
        A name for the problem, used in messages and printed output.
    phases : type[Phases]
        The class declaring the problem's phases.
    discrete : type[Vector], optional
        The class naming the problem's discrete constraint groups.
    parameter : type[Vector], optional
        The class naming the problem's parameters.
    """

    _held = ("phases", "objective", "discrete", "parameter", "derivatives", "ipopt_options")
    _settable = ("method", "catch_keyboard_interrupt")

    def __init__(
        self,
        name: str,
        *,
        phases: type[Phases],
        discrete: type[Vector] = Empty,
        parameter: type[Vector] = Empty,
    ) -> None:
        if not is_string(name):
            msg = f"the problem name must be a string; got {name!r}"
            raise TypeError(msg)
        if not is_subclass(phases, Phases):
            msg = (
                "Problem(phases=) takes a phase declaration, written as "
                f"'class Phases(yapss.Phases)'; got {phases!r}"
            )
            raise TypeError(msg)
        for label, value in (("discrete", discrete), ("parameter", parameter)):
            if not is_subclass(value, Vector):
                msg = (
                    f"Problem({label}=) takes a vector class declared with "
                    f"'class X(yapss.Vector)'; got {value!r}"
                )
                raise TypeError(msg)

        self._name = name
        self._label = "problem"
        self._objective_function: Callable[..., Any] | None = None
        self._discrete_function: Callable[..., Any] | None = None
        self._discrete_class = discrete
        self._parameter_class = parameter

        self._hold("phases", phases())
        self._hold("objective", ObjectiveAspects())
        self._hold("derivatives", Derivatives())
        self._hold("ipopt_options", IpoptOptions())
        self._hold("method", "lgl")
        self._hold("catch_keyboard_interrupt", CATCH_KEYBOARD_INTERRUPT)

        discrete_aspects = DiscreteAspects()
        discrete_aspects._label = "problem discrete"
        discrete_aspects._hold("bounds", discrete._new(Bounds, "discrete bounds"))
        discrete_aspects._hold("scale", discrete._new(Scale, "discrete scale"))
        self._hold("discrete", discrete_aspects)

        parameter_aspects = ParameterAspects()
        parameter_aspects._label = "problem parameter"
        parameter_aspects._hold("bounds", parameter._new(Bounds, "parameter bounds"))
        parameter_aspects._hold("guess", parameter._new(ScalarGuess, "parameter guess"))
        parameter_aspects._hold("scale", parameter._new(Scale, "parameter scale"))
        self._hold("parameter", parameter_aspects)

    @property
    def name(self) -> str:
        """str: The problem's name."""
        return self._name

    def _check(self, name: str, value: Any) -> Any:
        if name == "method":
            return _one_of(value, METHODS, "problem.method")
        if not isinstance(value, bool):
            msg = f"problem.catch_keyboard_interrupt must be True or False; got {value!r}"
            raise TypeError(msg)
        return value

    # -- registration -------------------------------------------------------------------------

    def objective_function(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the objective callback, as a decorator or as a call.

        The callback takes the endpoint argument and *returns* the objective, which is one
        expression, so there is no output object to fill.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator.
        replace : bool, default False
            Replace a callback already registered.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("objective", function, replace=replace)

    def discrete_function(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the discrete constraint callback, as a decorator or as a call.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator.
        replace : bool, default False
            Replace a callback already registered.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._register("discrete", function, replace=replace)

    def _register(self, which: str, function: Callable[..., Any] | None, *, replace: bool) -> Any:
        attribute = f"_{which}_function"

        def register(callback: Callable[..., Any]) -> Callable[..., Any]:
            if not is_callable(callback):
                msg = f"the {which} callback must be callable; got {callback!r}"
                raise TypeError(msg)
            current = getattr(self, attribute)
            if current is not None and not replace:
                existing = getattr(current, "__qualname__", repr(current))
                msg = (
                    f"the problem already has the {which} callback '{existing}'; pass "
                    f"replace=True to replace it"
                )
                raise ValueError(msg)
            object.__setattr__(self, attribute, callback)
            return callback

        return register if function is None else register(function)

    # -- solving ------------------------------------------------------------------------------

    def validate(self) -> None:
        """Check everything `solve` checks before Ipopt starts, without solving.

        Raises
        ------
        ValueError
            If the problem is incomplete: a missing callback, a missing time guess, or a
            constraint that was declared and never bounded.
        """
        validate_problem(self)

    def solve(self) -> Any:
        """Solve the problem.

        Returns
        -------
        Solution
            The solution, holding every quantity under the names the problem declared.
        """
        self.validate()
        return solve_problem(snapshot(self))

    def __repr__(self) -> str:
        """Return a short representation naming the problem and its phases."""
        names = ", ".join(phase.name for phase in self.phases)
        return f"<Problem {self._name!r} phases=({names})>"
