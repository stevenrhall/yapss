"""What a problem is, and what refuses to become one.

Everything here happens before a solve: what `Problem(...)` accepts, what a phase declaration
accepts, how callbacks are registered, and the completeness check that runs when a solve is
asked for. A message about the *shape* of the problem belongs here; a message about a value
put into that shape belongs to the page for that aspect.
"""

from __future__ import annotations

import pytest

import yapss

from ._api import Control, Discrete, Parameter, Phases, State, problem, raises, solvable

# --------------------------------------------------------------- what Problem() accepts


def test_name_must_be_a_string() -> None:
    """The problem's name is a string, and a number is a mistake rather than a label."""
    with raises(TypeError, "the problem name must be a string", at="yapss.Problem("):
        yapss.Problem(42, phases=Phases)  # type: ignore[arg-type]


def test_phases_must_be_a_phases_subclass() -> None:
    """`phases=` takes the class, not an instance of it and not a list."""
    with raises(TypeError, "class Phases(yapss.Phases)", at="yapss.Problem("):
        yapss.Problem("p", phases=[])  # type: ignore[arg-type]


def test_discrete_must_be_a_discrete_declaration() -> None:
    """`discrete=` takes a subclass of `yapss.Discrete`."""
    with raises(TypeError, "takes a subclass of", "yapss.Discrete", at="yapss.Problem("):
        yapss.Problem("p", phases=Phases, discrete=object)  # type: ignore[arg-type]


def test_parameter_must_be_a_parameter_declaration() -> None:
    """`parameter=` takes a subclass of `yapss.Parameter`, and the message names the keyword."""
    with raises(TypeError, "parameter=", "yapss.Parameter", at="yapss.Problem("):
        yapss.Problem("p", phases=Phases, parameter=object)  # type: ignore[arg-type]


def test_a_declaration_is_refused_in_the_wrong_role() -> None:
    """A state handed to a phase as its control is refused where it was handed over.

    Without the check the problem would build and solve, mislabelled throughout, and nothing
    would ever raise: the names a callback writes would simply be in the wrong places.
    """
    with raises(
        TypeError, "takes a subclass of yapss.Control", "subclasses yapss.State", at="yapss.phase("
    ):
        yapss.phase(state=State, control=State)  # type: ignore[type-var]


def test_a_swapped_argument_is_named() -> None:
    """The likeliest way to get a role wrong is to swap two arguments, so that is what is said.

    Changing the class's base would also silence the error, and would be the wrong fix.
    """
    with raises(TypeError, "Did you mean parameter=", at="yapss.Problem("):
        yapss.Problem("p", phases=Phases, discrete=Parameter)  # type: ignore[arg-type]


def test_a_parameter_may_not_share_a_name_with_a_phase_variable() -> None:
    """Parameters and a phase's variables are one namespace, so the names must differ."""

    class Shared(yapss.Parameter):
        x = yapss.scalar()

    with raises(
        ValueError,
        "declares 'x'",
        "one namespace",
        at="yapss.Problem(",
    ):
        yapss.Problem("p", phases=Phases, parameter=Shared)


def test_the_message_names_the_phase_the_parameter_collided_in() -> None:
    """Which phase the name collided in is the actionable part of the message."""

    class Other(yapss.Parameter):
        u = yapss.scalar()

    class TwoPhases(yapss.Phases):
        early = yapss.phase(state=State, control=Control)

    with raises(ValueError, "phase 'early'", at="yapss.Problem("):
        yapss.Problem("p", phases=TwoPhases, parameter=Other)


def test_catch_keyboard_interrupt_is_a_flag() -> None:
    """A setting that is either on or off refuses anything else."""
    p = problem()
    with raises(TypeError, "must be True or False", at="catch_keyboard_interrupt"):
        p.catch_keyboard_interrupt = "yes"  # type: ignore[assignment]


# ------------------------------------------------------------------- settings on a problem


def test_the_spectral_method_is_one_of_three() -> None:
    """A misspelled method is refused at the assignment, listing what is allowed."""
    p = problem()
    with raises(ValueError, "must be one of", "lgl", at="p.method"):
        p.method = "radau"  # type: ignore[assignment]


def test_the_sense_is_minimize_or_maximize() -> None:
    """The objective is minimized or maximized, and nothing else."""
    p = problem()
    with raises(ValueError, "must be one of", "maximize", at="sense"):
        p.objective.sense = "smallest"  # type: ignore[assignment]


# --------------------------------------------------------------- registering callbacks


def test_the_settings_object_is_not_a_decorator() -> None:
    """`@problem.objective` is the settings, not the registry, and says so."""
    p = problem()
    with raises(TypeError, "@problem.register.objective", at="@p.objective"):

        @p.objective
        def objective(arg):
            return 0.0


def test_the_discrete_settings_object_is_not_a_decorator() -> None:
    """The same for the discrete constraints' settings."""
    p = problem()
    with raises(TypeError, "@problem.register.discrete", at="@p.discrete"):

        @p.discrete
        def discrete(arg, out):
            return out


def test_a_callback_must_be_callable() -> None:
    """Registering something that cannot be called is refused where it is registered."""
    p = problem()
    with raises(TypeError, "must be callable", at="register.objective"):
        p.register.objective(3)  # type: ignore[arg-type]


def test_registering_twice_is_refused_unless_replacement_is_asked_for() -> None:
    """Two callbacks for one job is a mistake; replacing one is said out loud."""
    p = problem()

    @p.register.objective
    def objective(arg):
        return 0.0

    with raises(ValueError, "already has the objective callback", "replace=True", at="register"):

        @p.register.objective
        def other(arg):
            return 1.0


def test_replace_true_replaces_the_callback() -> None:
    """The escape hatch works, which is what makes the refusal above a reasonable one."""
    p = solvable()

    @p.register.objective(replace=True)
    def objective(arg):
        return arg[p.phases.slide].final.time

    assert p.solve().converged


# ------------------------------------------------------- the problem must be complete


def test_a_problem_with_no_objective_is_incomplete() -> None:
    """A problem states what is to be made smallest; without it there is nothing to solve."""

    class OnlyPhase(yapss.Phases):
        only = yapss.phase(state=State, control=Control)

    p = yapss.Problem("p", phases=OnlyPhase)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]
        return out

    ph.time.guess = (0.0, 1.0)
    with raises(ValueError, "the problem is incomplete", "no objective callback", at="validate"):
        p.validate()


def test_a_phase_with_no_continuous_callback_is_incomplete() -> None:
    """A phase without dynamics is not a phase, and the message names which one."""
    p = problem()

    @p.register.objective
    def objective(arg):
        return arg[p.phases.first].final.x

    for ph in p.phases:
        ph.time.guess = (0.0, 1.0)
    with raises(
        ValueError,
        "phase 'first' has no continuous callback",
        at="validate",
    ):
        p.validate()


def test_declared_discrete_constraints_need_a_callback() -> None:
    """Declaring a constraint and never computing it is a mistake, not an option."""

    class OnlyPhase(yapss.Phases):
        only = yapss.phase(state=State, control=Control)

    p = yapss.Problem("p", phases=OnlyPhase, discrete=Discrete)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]
        return out

    @p.register.objective
    def objective(arg):
        return arg[ph].final.x

    ph.time.guess = (0.0, 1.0)
    p.discrete.d.bounds = (0.0, 0.0)
    p.discrete.e.bounds[:] = (0.0, 0.0)
    with raises(ValueError, "no discrete callback", at="validate"):
        p.validate()


def test_a_problem_with_no_decision_variables_is_refused() -> None:
    """Every count may be zero, but not all of them at once: that is the absence of a problem."""

    class NoPhases(yapss.Phases):
        pass

    p = yapss.Problem("p", phases=NoPhases)

    @p.register.objective
    def objective(arg):
        return 0.0

    with raises(ValueError, "no decision variables", "declare a phase", at="validate"):
        p.validate()


def test_the_complaints_are_reported_together() -> None:
    """One message lists everything incomplete, so a user fixes them in one pass."""
    p = problem()
    with pytest.raises(ValueError) as info:
        p.validate()
    message = str(info.value)
    assert message.count("\n") >= 2, message
    assert "no objective callback" in message
    assert "phase 'first' has no continuous callback" in message


# ------------------------------------------------------------ declaring vectors and phases


def test_a_vector_takes_a_whole_number_of_components() -> None:
    """A vector field's size is a count of components."""
    with raises(TypeError, "vector(size) takes a whole number of components", at="yapss.vector"):
        yapss.vector("two")  # type: ignore[arg-type]


def test_a_vector_size_is_not_negative() -> None:
    """Zero components is allowed -- every count may be zero -- but a negative count is not."""
    with raises(ValueError, "vector(size) must be 0 or more", at="yapss.vector"):
        yapss.vector(-1)


def test_a_declaration_holds_only_fields() -> None:
    """Anything else in the class body is a mistake, and the message shows the form."""
    with raises(TypeError, "is not a field", "yapss.scalar()", at="class Wrong"):

        class Wrong(yapss.State):
            x = 1.0


def test_a_field_is_declared_without_an_annotation() -> None:
    """An annotation makes it a class variable rather than a field, so it is caught."""
    with raises(TypeError, "is annotated", "without annotations", at="class Annotated"):

        class Annotated(yapss.State):
            x: float = yapss.scalar()


def test_a_declaration_may_not_be_inherited_from() -> None:
    """Fields are declared where they are used, so a hierarchy cannot fragment them."""
    with raises(TypeError, "cannot inherit from the vector declaration", at="class Child"):

        class Child(State):
            z = yapss.scalar()


def test_a_phase_takes_role_declarations() -> None:
    """Each role is a declaration, and the message names the role that was not one."""
    with raises(TypeError, "phase(state=) takes a subclass of yapss.State", at="yapss.phase"):
        yapss.phase(state=object)


def test_a_phase_has_one_namespace() -> None:
    """A state and a control may not share a name: a callback reaches both from one `arg`."""

    class Same(yapss.Control):
        x = yapss.scalar()

    with raises(ValueError, "both declare 'x'", "one namespace", at="yapss.phase"):
        yapss.phase(state=State, control=Same)


def test_the_independent_variable_shares_that_namespace() -> None:
    """Naming it after a state is the same collision, and the message says how to rename."""
    with raises(ValueError, "one namespace", at="yapss.phase"):
        yapss.phase(state=State, control=Control, x=yapss.scalar())


def test_a_phase_takes_one_independent_variable() -> None:
    """Two would leave no way to say which a guess or a bound is about."""
    with raises(
        (TypeError, ValueError),
        "one independent variable",
        at="yapss.phase",
    ):
        yapss.phase(state=State, control=Control, r=yapss.scalar(), s=yapss.scalar())


def test_the_independent_variable_is_a_scalar() -> None:
    """It is one value, so it is declared as a scalar and never as a vector."""
    with raises(TypeError, "is one value", "yapss.scalar()", at="yapss.phase"):
        yapss.phase(state=State, control=Control, r=yapss.vector(3))


def test_a_misspelled_role_is_refused_with_a_suggestion() -> None:
    """`phase(states=...)` is a typo, not a new role and not an independent variable."""
    with raises(TypeError, "unexpected keyword 'controls'", at="yapss.phase"):
        yapss.phase(state=State, controls=Control)


def test_a_phases_declaration_holds_only_phases() -> None:
    """The same rule as a vector's, for the class that declares phases."""
    with raises(TypeError, "is not a phase", "yapss.phase(", at="class Wrong"):

        class Wrong(yapss.Phases):
            first = 1.0


def test_a_phase_cannot_be_assigned_after_declaration() -> None:
    """Phases are declared, so the set of them is fixed once the class is written."""
    p = problem()
    with raises(AttributeError, "cannot be assigned; phases are declared", at="p.phases.first"):
        p.phases.first = None


def test_a_registration_is_called_not_assigned() -> None:
    """`register.objective = f` looks reasonable and is not how it is done, so it says how."""
    p = problem()
    with raises(
        AttributeError,
        "is not assigned",
        "register.objective(callback)",
        at="p.register.objective =",
    ):
        p.register.objective = lambda arg: 0.0


def test_a_phases_declaration_may_not_be_inherited_from() -> None:
    """The same rule as a vector's: the phases are declared where they are used."""
    with raises(TypeError, "cannot inherit from the phase declaration", at="class Child"):

        class Child(Phases):
            third = yapss.phase(state=State, control=Control)


def test_a_phase_is_declared_without_an_annotation() -> None:
    """An annotation would make it a class variable rather than a phase."""
    with raises(TypeError, "is annotated", "without annotations", at="class Annotated"):

        class Annotated(yapss.Phases):
            first: object = yapss.phase(state=State, control=Control)


def test_a_phase_callback_must_be_callable() -> None:
    """The phase's own registry checks what it is handed, as the problem's does."""
    p = problem()
    with raises(TypeError, "callback must be callable", at="register.continuous"):
        p.phases.first.register.continuous(3)


def test_a_phase_callback_is_registered_once() -> None:
    """Two continuous callbacks for one phase is a mistake; replacing is said out loud."""
    p = problem()
    ph = p.phases.first

    @ph.register.continuous
    def continuous(arg, out):
        return out

    with raises(ValueError, "already has the continuous callback", "replace=True", at="register"):

        @ph.register.continuous
        def other(arg, out):
            return out
