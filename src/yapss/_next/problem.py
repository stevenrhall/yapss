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
from yapss._private.solution import warn_if_not_converged

from .compile import solve_problem
from .containers import Container, HasRegistry, Registry, is_callable, is_string, is_subclass
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse a registration written here, which is a likely slip. See `_not_a_decorator`."""
        del args, kwargs
        raise TypeError(_not_a_decorator("objective", "the objective"))


def _not_a_decorator(which: str, phrase: str) -> str:
    """Return the message for a callback registered on an aspect rather than on `register`.

    ``problem.objective`` and ``problem.discrete`` hold the settings of those things -- the
    sense, the scale, the bounds -- and the callbacks are registered next door. Decorating with
    the aspect is the natural slip, so it is answered rather than left to read as "object is
    not callable".
    """
    return (
        f"problem.{which} holds the settings of {phrase}, not the callback. Register the "
        f"callback with '@problem.register.{which}'."
    )


class DiscreteAspects(Container):
    """The discrete constraints: their bounds and their scales."""

    _held = ("bounds", "scale")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse a registration written here, which is a likely slip. See `_not_a_decorator`."""
        del args, kwargs
        raise TypeError(_not_a_decorator("discrete", "the discrete constraints"))


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


def _check_parameters(parameter: type[Vector], phases: Any) -> None:
    """Refuse a parameter whose name is also a variable of some phase.

    A phase's states, controls and independent variable are one namespace, and the parameters
    join it: a derivative names a variable from it without saying which vector it came from
    (spec 5.7). The phase declaration cannot check this half, because the parameters are the
    problem's and arrive here; this is where they meet.

    The message names the phase, which is what distinguishes this from the half `phase()`
    checks -- there the two classes are the actionable thing, here it is which phase the
    parameter collided in.
    """
    if not parameter._fields:
        return
    names = set(parameter._fields)
    for phase in phases:
        declaration = phase._declaration
        variables = {
            **dict.fromkeys(declaration.state._fields, "a state"),
            **dict.fromkeys(declaration.control._fields, "a control"),
            declaration.independent: "its independent variable",
        }
        for shared in sorted(names & set(variables)):
            msg = (
                f"Problem(parameter={parameter.__name__}) declares {shared!r}, which phase "
                f"'{phase.name}' also has as {variables[shared]}. Parameters are the "
                f"problem's, so their names must differ from every phase's variables; they "
                f"are one namespace."
            )
            raise ValueError(msg)


class ProblemRegistry(Registry):
    """The problem's callbacks. Reached as ``problem.register``."""

    _registrations = (
        "objective",
        "discrete",
        "objective_gradient",
        "objective_hessian",
        "discrete_jacobian",
        "discrete_hessian",
    )
    _label = "problem callbacks"

    def __init__(self, problem: Problem) -> None:
        self._problem = problem

    def objective(
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
        return self._problem._register("objective", function, replace=replace)

    def discrete(
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
        return self._problem._register("discrete", function, replace=replace)

    def objective_gradient(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the objective's gradient, as a decorator or as a call.

        Required under ``derivatives.method = "user"``. The callback takes the endpoint
        argument and a `gradient` to fill: ``gradient[ph].final.time = 1.0``. What it writes
        is the sparsity structure, so a name it does not write is a derivative that is zero
        everywhere, and the same names must be written on every call.

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
        return self._problem._register("objective_gradient", function, replace=replace)

    def objective_hessian(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the objective's Hessian, as a decorator or as a call.

        Required under ``derivatives.method = "user"`` at ``derivatives.order = "second"``,
        *including* when every entry of it is zero: a callback that writes nothing says so,
        and leaving it out would be indistinguishable from forgetting it. Forgetting it is
        not caught by the answer, because a wrong Hessian still leaves the same KKT point --
        it costs iterations instead, which is the one failure worth refusing in a feature
        whose purpose is speed.

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
        return self._problem._register("objective_hessian", function, replace=replace)

    def discrete_jacobian(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the discrete constraints' Jacobian, as a decorator or as a call.

        Required under ``derivatives.method = "user"`` when the problem declares discrete
        constraints. The callback takes the endpoint argument and a `jacobian` whose entries
        name the constraint group and the variable:
        ``jacobian.discrete.link[ph].final.h = -1.0``.

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
        return self._problem._register("discrete_jacobian", function, replace=replace)

    def discrete_hessian(
        self, function: Callable[..., Any] | None = None, /, *, replace: bool = False
    ) -> Any:
        """Register the discrete constraints' Hessian, as a decorator or as a call.

        Required under ``derivatives.method = "user"`` at ``derivatives.order = "second"``
        when the problem declares discrete constraints, *including* when every entry is zero:
        linkage constraints are linear, and an empty callback is how that is said.

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
        return self._problem._register("discrete_hessian", function, replace=replace)


class Problem(HasRegistry):
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

    _held = (
        "phases",
        "objective",
        "discrete",
        "parameter",
        "derivatives",
        "ipopt_options",
        "register",
    )
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
        self._objective_gradient_function: Callable[..., Any] | None = None
        self._objective_hessian_function: Callable[..., Any] | None = None
        self._discrete_jacobian_function: Callable[..., Any] | None = None
        self._discrete_hessian_function: Callable[..., Any] | None = None
        self._discrete_class = discrete
        self._parameter_class = parameter

        self._hold("phases", phases())
        _check_parameters(parameter, self.phases)
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
        self._hold("register", ProblemRegistry(self))

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

        Warns
        -----
        IpoptConvergenceWarning
            If Ipopt reported a status other than 0 (optimal), 1 (acceptable level) or
            6 (feasible point for a square problem). A `Solution` is returned for every
            status; an unconverged solve is valid input that deserves attention rather than
            a contract violation, which is the rule of CLAUDE.md's conventions.
        """
        self.validate()
        solution = solve_problem(snapshot(self))
        # The warning belongs at the public boundary, not inside the solve, so that its
        # stacklevel points at the caller's own `solve()`; a mesh-refinement loop written
        # against this API calls it once per pass and should hear about each one.
        # stacklevel=3: warn -> warn_if_not_converged -> this method -> user code.
        warn_if_not_converged(solution._legacy, stacklevel=3)
        return solution

    def __repr__(self) -> str:
        """Return a short representation naming the problem and its phases."""
        names = ", ".join(phase.name for phase in self.phases)
        return f"<Problem {self._name!r} phases=({names})>"
