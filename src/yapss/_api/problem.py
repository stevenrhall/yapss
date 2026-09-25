"""

The problem: declarations in, a solution out.

Everything about the problem lives on it -- its phases, its parameters and discrete
constraints, its objective, and the settings that say how it is to be solved -- and
``solve()`` takes no arguments. Each solve first takes a snapshot, so editing a problem
afterwards never alters what an earlier solution recorded.

"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Generic, Literal, overload

# See `_api.declare`: PEP 696 defaults, which `typing.TypeVar` cannot carry below 3.13.
from typing_extensions import TypeVar

from yapss._backend.ipopt_options import IpoptOptions
from yapss._backend.solution import warn_if_not_converged

from .compile import solve_problem
from .containers import (
    CallbackT,
    Container,
    FillerT,
    HasRegistry,
    Registry,
    is_callable,
    is_string,
    is_subclass,
)
from .declare import Phases, declared_role
from .fields import Fields
from .kinds import Bounds, ScalarGuess, Scale
from .old_api import old_api_message
from .spec import snapshot, validate_problem
from .vector import Discrete, Parameter, Vector

if TYPE_CHECKING:
    from collections.abc import Callable

    from .solution import Solution

__all__ = ["Problem"]

# Each default is `Any`, not the role. A bare `yapss.Problem` -- the annotation every example's
# `setup` returns, and a helper taking any problem writes -- cannot say what was declared, so it
# answers any name, and so does the solution it returns; a problem built from its declarations
# is typed by them, and a misspelling is reported there. The base class cannot do both: a
# permissive reader on `Phases` would be inherited by every declaration and blind the checker to
# every phase name. A keyword left out of the constructor is typed as its role by the
# overloads of `Problem.__init__`, so a problem declaring no parameters reports reading one.
PH_co = TypeVar("PH_co", bound=Phases, default=Any, covariant=True)
"""The class declaring the problem's phases."""
D_co = TypeVar("D_co", bound=Discrete, default=Any, covariant=True)
"""The class declaring the problem's discrete constraint groups."""
PR_co = TypeVar("PR_co", bound=Parameter, default=Any, covariant=True)
"""The class declaring the problem's parameters."""

SPECTRAL_METHODS = ("lgl", "lgr", "lg")
DERIVATIVE_METHODS = ("auto", "central-difference", "central-difference-full")

_NO_USER_METHOD = (
    "derivatives.method = 'user' is not offered: derivatives written by hand were feasible "
    "only for problems small enough that 'auto' differentiates them instantly. Use 'auto'; "
    "for a model that cannot be traced, use 'central-difference', or "
    "'central-difference-full' where its sparsity cannot be detected."
)
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

    if TYPE_CHECKING:
        sense: Literal["minimize", "maximize"]
        scale: float

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
    """How derivatives are computed.

    Attributes
    ----------
    method : {"auto", "central-difference", "central-difference-full"}
        How derivatives are computed: automatic differentiation with CasADi (the default), or
        central differences, with or without sparsity detection.
    order : {"first", "second"}
        The order of derivatives Ipopt is given. Under ``"first"``, Ipopt approximates the
        Hessian of the Lagrangian itself. The default is ``"second"``.
    """

    _settable = ("method", "order")

    if TYPE_CHECKING:
        method: Literal["auto", "central-difference", "central-difference-full"]
        order: Literal["first", "second"]

    def __init__(self) -> None:
        self._label = "derivatives"
        self._hold("method", "auto")
        self._hold("order", "second")

    def _check(self, name: str, value: Any) -> Any:
        if name == "method":
            # 0.3.0 offered "user", so it is the value a user porting a problem will try.
            if value == "user":
                raise ValueError(_NO_USER_METHOD)
            return _one_of(value, DERIVATIVE_METHODS, "derivatives.method")
        return _one_of(value, ORDERS, "derivatives.order")


def _check_name(name: object) -> None:
    """Refuse a name that is not a string, in the constructor and on assignment alike."""
    if not is_string(name):
        msg = f"the problem name must be a string; got {name!r}"
        raise TypeError(msg)


def _check_parameters(parameter: type[Vector], phases: Any) -> None:
    """Refuse a parameter whose name is also a variable of some phase.

    A phase's states, controls and independent variable are one namespace, and the parameters
    join it: a derivative names a variable from that namespace without saying which vector it
    came from, so a name belonging to two of them would name two columns of the phase's
    Jacobian. The phase declaration cannot check this half, because the parameters are the
    problem's and arrive here; this is where they meet.

    The message names the phase, which is what distinguishes this from the half a phase's
    declaration checks -- there the two classes are the actionable thing, here it is which phase the
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


def _check_decision_variables(phases: Any, parameter: type[Vector]) -> None:
    """Refuse a problem in no variables at all, which is the one combination with no meaning.

    Every count in this API may be zero, and the transcription holds at the bottom of each
    range on its own: a problem may declare no phases, and a phase that declares nothing still
    contributes its own initial and final time. The one combination that has no meaning is all
    of them at once -- with neither a phase nor a parameter the nonlinear program has no
    variables, and that is refused because *Ipopt* does not keep the principle, not because
    YAPSS cannot state the problem.

    Refused here rather than in `validate` because it is decidable here and incurable after: a
    declaration is a class, so neither count can change once the problem is built, and the line
    that has to be fixed is the one that called `Problem`. Leaving it to `validate` also had it
    reported as incompleteness, which it is not -- nothing is missing from a constant objective
    over no variables; there is simply nothing to choose.

    The test is a proxy for the fact. What is wrong is that there are no decision variables;
    what is *checked* is that nothing was declared that would make one, and the two sides of
    that are counted differently on purpose. A phase counts by existing, because it contributes
    its initial and final time whatever else it declares, and a fixed time is still a variable,
    bounded above and below by the same number. Parameters count by *rows*, not by fields: a
    block field may have no rows, so a declaration can be non-empty and contribute nothing.

    Should fixed variables ever be eliminated instead, this check stays right and an
    equivalence written into the message would have become a lie -- so the message states the
    fact and then advises, rather than explaining how the check works. `mseipopt` refuses an
    empty NLP as well, which is the backstop if the two ever come apart.
    """
    if not phases and not parameter._nrows:
        msg = (
            "the problem has no decision variables, so there is nothing to choose:\n"
            "Ipopt requires at least one variable, so declare a phase or a parameter"
        )
        raise ValueError(msg)


class ProblemRegistry(Registry):
    """The problem's callbacks. Reached as ``problem.register``."""

    _registrations = ("objective", "discrete")
    _label = "problem callbacks"

    def __init__(self, problem: Problem[Any, Any, Any]) -> None:
        self._problem = problem

    @overload
    def objective(self, function: CallbackT, /) -> CallbackT: ...
    @overload
    def objective(self, function: None = None, /) -> Callable[[CallbackT], CallbackT]: ...
    def objective(self, function: Callable[..., Any] | None = None, /) -> Any:
        """Register the objective callback, as a decorator or as a call.

        The callback takes the endpoint argument and *returns* the objective, which is one
        expression, so there is no output object to fill.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator. Registering a second one
            replaces the first, as setting any other value twice does.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._problem._register("objective", function)

    @overload
    def discrete(self, function: FillerT, /) -> FillerT: ...
    @overload
    def discrete(self, function: None = None, /) -> Callable[[FillerT], FillerT]: ...
    def discrete(self, function: Callable[..., Any] | None = None, /) -> Any:
        """Register the discrete constraint callback, as a decorator or as a call.

        Parameters
        ----------
        function : callable, optional
            The callback. Omit it to use the result as a decorator. Registering a second one
            replaces the first, as setting any other value twice does.

        Returns
        -------
        Any
            The callback, or a decorator that registers one.
        """
        return self._problem._register("discrete", function)


class Problem(HasRegistry, Generic[PH_co, D_co, PR_co]):
    """An optimal control problem.

    Parameters
    ----------
    name : str
        A name for the problem, used in messages and printed output.
    phases : type[Phases], optional
        The class declaring the problem's phases; none, if omitted.
    discrete : type[Discrete], optional
        The class naming the problem's discrete constraint groups; none, if omitted.
    parameter : type[Parameter], optional
        The class naming the problem's parameters; none, if omitted.

    Attributes
    ----------
    name : str
        The problem's name. It may be changed; each solve records the name in force.
    phases
        The phases, by the names the ``phases`` class declared: ``problem.phases.<name>``.
    objective
        The objective's ``sense`` (``"minimize"``, the default, or ``"maximize"``) and ``scale``
        (positive, default 1.0).
    discrete, parameter
        The discrete constraints and parameters, by the names their classes declared, each
        with its settings: ``problem.discrete.<name>.bounds``, ``problem.parameter.<name>.guess``.
    derivatives
        How derivatives are computed: ``method`` and ``order``.
    ipopt_options
        Options passed to Ipopt, set by name: ``problem.ipopt_options.max_iter = 500``.
    register
        Where the objective and discrete callbacks are registered:
        ``@problem.register.objective``.
    spectral_method : {"lgl", "lgr", "lg"}
        The collocation points used in every phase. Default ``"lgl"``.
    catch_keyboard_interrupt : bool
        Whether Ctrl-C during a solve stops Ipopt at its next iterate and returns that iterate
        as a `Solution`, with status 5 and an `IpoptConvergenceWarning`, so that a long or
        stalled solve can be stopped and inspected rather than lost. Default True. With False,
        Ctrl-C raises `KeyboardInterrupt` as usual. Takes effect only when solving on the main
        thread, the only thread Python lets install a signal handler.
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
    _settable = ("name", "spectral_method", "catch_keyboard_interrupt")

    if TYPE_CHECKING:
        # The three parameters are the classes the problem was declared with, so a type
        # checker follows `problem.phases.<name>` into the phase and on to the fields of its
        # state and control without an annotation being written anywhere. Each has a default,
        # which is what keeps the bare `yapss.Problem` -- the spelling every example's `setup`
        # returns -- both legal under `disallow_any_generics` and meaningful.
        phases: PH_co
        # Typed as the declarations, so a setting is reached field first and checked: see
        # `Phase`. At runtime each is a `fields.Fields` over the aspect container.
        discrete: D_co
        parameter: PR_co
        objective: ObjectiveAspects
        derivatives: Derivatives
        ipopt_options: IpoptOptions
        register: ProblemRegistry
        name: str
        spectral_method: Literal["lgl", "lgr", "lg"]
        catch_keyboard_interrupt: bool

    # Hidden from type checkers, so the overloads below remain the signature they check. Python
    # passes the constructor's arguments to __new__ before __init__, so code written for YAPSS
    # 0.3 or earlier -- whose Problem required `nx` in every release -- is recognized here and
    # told what happened, rather than being refused as an unexpected keyword.
    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Problem[Any, Any, Any]:  # noqa: ARG004
            if "nx" in kwargs:
                raise TypeError(old_api_message("Problem(name=..., nx=...)"))
            return super().__new__(cls)

    # One overload per combination of the three keywords, all of them optional. A keyword left
    # out pins its parameter to the role, which declares no fields -- not to the class default,
    # `Any`, which is for the bare annotation. Every count may be zero, phases included; a
    # problem with nothing to choose at all is refused by `validate`, not here.
    @overload
    def __init__(
        self: Problem[PH_co, D_co, PR_co],
        name: str,
        *,
        phases: type[PH_co],
        discrete: type[D_co],
        parameter: type[PR_co],
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[PH_co, D_co, Parameter],
        name: str,
        *,
        phases: type[PH_co],
        discrete: type[D_co],
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[PH_co, Discrete, PR_co],
        name: str,
        *,
        phases: type[PH_co],
        parameter: type[PR_co],
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[PH_co, Discrete, Parameter], name: str, *, phases: type[PH_co]
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[Phases, D_co, PR_co],
        name: str,
        *,
        discrete: type[D_co],
        parameter: type[PR_co],
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[Phases, D_co, Parameter], name: str, *, discrete: type[D_co]
    ) -> None: ...

    @overload
    def __init__(
        self: Problem[Phases, Discrete, PR_co], name: str, *, parameter: type[PR_co]
    ) -> None: ...

    @overload
    def __init__(self: Problem[Phases, Discrete, Parameter], name: str) -> None: ...

    def __init__(
        self,
        name: str,
        *,
        # See `_api.declare.phase`: the defaults are declared on the type parameters, and a
        # checker measures the default value against the parameter type regardless.
        phases: type[PH_co] = Phases,  # type: ignore[assignment]
        discrete: type[D_co] = Discrete,  # type: ignore[assignment]
        parameter: type[PR_co] = Parameter,  # type: ignore[assignment]
    ) -> None:
        _check_name(name)
        if not is_subclass(phases, Phases):
            msg = (
                "Problem(phases=) takes a phase declaration, written as "
                f"'class Phases(yapss.Phases)'; got {phases!r}"
            )
            raise TypeError(msg)
        roles: dict[str, type[Vector]] = {"discrete": Discrete, "parameter": Parameter}
        declared_role(discrete, "Problem", "discrete", roles)
        declared_role(parameter, "Problem", "parameter", roles)

        self._label = "problem"
        self._objective_function: Callable[..., Any] | None = None
        self._discrete_function: Callable[..., Any] | None = None
        self._discrete_class = discrete
        self._parameter_class = parameter

        self._hold("phases", phases())
        _check_parameters(parameter, self.phases)
        _check_decision_variables(self.phases, parameter)
        self._hold("objective", ObjectiveAspects())
        self._hold("derivatives", Derivatives())
        self._hold("ipopt_options", IpoptOptions())
        self._hold("name", name)
        self._hold("spectral_method", "lgl")
        self._hold("catch_keyboard_interrupt", CATCH_KEYBOARD_INTERRUPT)

        discrete_aspects = DiscreteAspects()
        discrete_aspects._label = "problem discrete"
        discrete_aspects._hold("bounds", discrete._new(Bounds, "discrete bounds", aspect="bounds"))
        discrete_aspects._hold("scale", discrete._new(Scale, "discrete scale", aspect="scale"))
        self._hold("discrete", Fields(discrete_aspects, discrete, discrete_aspects._label))

        parameter_aspects = ParameterAspects()
        parameter_aspects._label = "problem parameter"
        parameter_aspects._hold(
            "bounds", parameter._new(Bounds, "parameter bounds", aspect="bounds")
        )
        parameter_aspects._hold(
            "guess", parameter._new(ScalarGuess, "parameter guess", aspect="guess")
        )
        parameter_aspects._hold("scale", parameter._new(Scale, "parameter scale", aspect="scale"))
        self._hold("parameter", Fields(parameter_aspects, parameter, parameter_aspects._label))
        self._hold("register", ProblemRegistry(self))

    def _check(self, name: str, value: Any) -> Any:
        if name == "name":
            # A name is a value the user means, not something YAPSS computes from the rest, so
            # it is theirs to change -- and a sweep that re-solves one problem wants to say
            # which variant each solution came from. `snapshot` reads it at each solve, so the
            # solutions of one problem carry the names it had when each was solved.
            _check_name(value)
            return value
        if name == "spectral_method":
            return _one_of(value, SPECTRAL_METHODS, "problem.spectral_method")
        if not isinstance(value, bool):
            msg = f"problem.catch_keyboard_interrupt must be True or False; got {value!r}"
            raise TypeError(msg)
        return value

    # -- registration -------------------------------------------------------------------------

    def _register(self, which: str, function: Callable[..., Any] | None) -> Any:
        # Registering is setting a value, and the last one wins, as it does for every other
        # setting. Refusing a second registration would make the registry the one setter in
        # the API that refuses to be set twice -- and it would refuse the two things users
        # actually do: re-run a notebook cell after editing the callback, and re-solve one
        # problem with a different objective. Whether a second registration was meant can
        # only be inferred, never seen from here, so nothing is raised or warned about.
        attribute = f"_{which}_function"

        def register(callback: Callable[..., Any]) -> Callable[..., Any]:
            if not is_callable(callback):
                msg = f"the {which} callback must be callable; got {callback!r}"
                raise TypeError(msg)
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

    def solve(self) -> Solution[D_co, PR_co]:
        """Solve the problem.

        Returns
        -------
        Solution
            The solution, holding every quantity under the names the problem declared.

        Raises
        ------
        ValueError
            If the problem is incomplete (see `validate`), or if Ipopt stopped without an
            iterate because of the problem or its callbacks: too few degrees of freedom
            (status -10), inconsistent bounds (-11), a refused option (-12), or a NaN or
            Inf from a callback (-13).
        MemoryError
            If Ipopt ran out of memory (-102).
        RuntimeError
            For a failure inside Ipopt (-100, -101, -199), or a status this version of
            YAPSS does not recognize.

        Warns
        -----
        IpoptConvergenceWarning
            If Ipopt stopped at an iterate but reported a status other than 0 (optimal),
            1 (acceptable level) or 6 (feasible point for a square problem). A `Solution`
            is returned for each of these; an unconverged solve is valid input that
            deserves attention.
        """
        self.validate()
        solution, record = solve_problem(snapshot(self))
        # The warning belongs at the public boundary, not inside the solve, so that its
        # stacklevel points at the caller's own `solve()`; a mesh-refinement loop written
        # against this API calls it once per pass and should hear about each one.
        # stacklevel=3: warn -> warn_if_not_converged -> this method -> user code.
        warn_if_not_converged(record, stacklevel=3)
        return solution

    def __repr__(self) -> str:
        """Return a short representation naming the problem and its phases."""
        names = ", ".join(phase.name for phase in self.phases)
        return f"<Problem {self.name!r} phases=({names})>"


# The hidden __new__ takes any arguments, and inspect.signature consults a user-defined __new__
# before __init__, so without this the signature a notebook's help or `help()` shows would be
# `(*args, **kwargs)`. Restore __init__'s, which is the constructor's true signature.
if not TYPE_CHECKING:
    _init = inspect.signature(Problem.__init__)
    Problem.__signature__ = _init.replace(parameters=list(_init.parameters.values())[1:])
    del _init
