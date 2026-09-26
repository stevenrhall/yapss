"""What a problem is, and what refuses to become one.

Everything here happens before a solve: what `Problem(...)` accepts, what a phase declaration
accepts, how callbacks are registered, and the completeness check that runs when a solve is
asked for. A message about the *shape* of the problem belongs here; a message about a value
put into that shape belongs to the page for that aspect.
"""

from __future__ import annotations

import numpy as np
import pytest

import yapss

from ._api import Control, Discrete, Parameter, Phases, State, problem, raises, solvable

# --------------------------------------------------------------- what Problem() accepts


def test_name_must_be_a_string() -> None:
    """The problem's name is a string, and a number is a mistake rather than a label."""
    with raises(TypeError, "the problem name must be a string", at="yapss.Problem("):
        yapss.Problem(42, phases=Phases)  # type: ignore[arg-type]


def test_the_name_may_be_changed() -> None:
    """A name is a value the user means, so it is theirs to set, like any other setting.

    It is read at each solve, so a problem re-solved as a sweep gives each solution the name
    it carried when that solve ran -- which is how a solution says which variant it is.
    """
    p = solvable()
    p.name = "renamed"
    assert p.name == "renamed"
    assert repr(p).startswith("<Problem 'renamed'")
    assert p.solve().name == "renamed"


def test_the_name_is_a_string_wherever_it_is_set() -> None:
    """The same refusal on assignment as in the constructor, in the same words."""
    p = problem()
    with raises(TypeError, "the problem name must be a string", at="p.name"):
        p.name = 42  # type: ignore[assignment]


def test_a_declaration_cannot_be_replaced() -> None:
    """The shape is said once: `phases=`, `discrete=` and `parameter=` fix it at construction.

    A name is a value; a declaration is a shape, and replacing one would silently discard every
    bound, guess and scale already set under it. The message points at what *is* settable, and
    for phases that is two levels down -- a phase is not a field of `phases`.
    """
    p = solvable()
    with raises(AttributeError, "phases cannot be replaced", "phases.slide.state.x.bounds"):
        p.phases = None  # type: ignore[assignment]
    with raises(AttributeError, "parameter cannot be replaced"):
        p.parameter = None  # type: ignore[assignment]


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


def test_a_problem_with_no_decision_variables_is_refused() -> None:
    """Every count may be zero, but not all of them at once: that is the absence of a problem.

    Refused where the problem is built, not where it is solved: the counts come from classes
    and no later statement can change them, so the line that called `Problem` is the line to
    fix. It is not incompleteness -- nothing is missing from a constant objective over no
    variables -- so it is not reported as one.
    """
    with raises(ValueError, "no decision variables", "declare a phase", at="yapss.Problem("):
        yapss.Problem("p")


def test_a_declaration_is_refused_in_the_wrong_role() -> None:
    """A state annotated as a phase's control is refused where it was annotated.

    Without the check the problem would build and solve, mislabelled throughout, and nothing
    would ever raise: the names a callback writes would simply be in the wrong places.
    """
    with raises(
        TypeError,
        "Wrong.control is annotated State",
        "subclasses yapss.State",
        "takes a subclass of yapss.Control",
        at="class Wrong",
    ):

        class Wrong(yapss.Phase):
            state: State
            control: State  # type: ignore[assignment]
            time: yapss.Independent


def test_a_swapped_annotation_is_named() -> None:
    """A control annotated as the state is most likely a swap, so the message says which."""
    with raises(TypeError, "Did you mean 'control: Control'?", at="class Swapped"):

        class Swapped(yapss.Phase):
            state: Control  # type: ignore[assignment]
            time: yapss.Independent


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

    class Early(yapss.Phase):
        state: State
        control: Control
        time: yapss.Independent

    class TwoPhases(yapss.Phases):
        early: Early

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
    with raises(ValueError, "must be one of", "lgl", at="p.spectral_method"):
        p.spectral_method = "radau"  # type: ignore[assignment]


def test_the_sense_is_minimize_or_maximize() -> None:
    """The objective is minimized or maximized, and nothing else."""
    p = problem()
    with raises(ValueError, "must be one of", "maximize", at="sense"):
        p.objective.sense = "smallest"  # type: ignore[assignment]


@pytest.mark.parametrize(
    ("owner", "setting", "valid"),
    [
        ("problem", "spectral_method", "lg"),
        ("objective", "sense", "maximize"),
        ("derivatives", "method", "central-difference"),
        ("derivatives", "order", "first"),
    ],
)
def test_a_choice_is_a_string_not_an_array_holding_one(
    owner: str, setting: str, valid: str
) -> None:
    """An array of one allowed name passes a membership test, so the type is checked first.

    Stored, its string form -- "['maximize']" -- would match nothing downstream, and a sense
    that matches nothing is not "maximize": the solve would minimize without a word.
    """
    p = problem()
    target = p if owner == "problem" else getattr(p, owner)
    with raises(TypeError, f"{setting} is a string", "got array", at="setattr"):
        setattr(target, setting, np.array([valid]))


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
            pass


def test_a_callback_must_be_callable() -> None:
    """Registering something that cannot be called is refused where it is registered."""
    p = problem()
    with raises(TypeError, "must be callable", at="register.objective"):
        p.register.objective(3)  # type: ignore[arg-type]


def test_registering_twice_replaces() -> None:
    """Registering is setting a value, and every other setting in this API takes a second one.

    Refused until 2026-09-23, which made the notebook's basic gesture -- edit a cell, run it
    again -- an error, and made re-solving one problem against a different objective an error
    too. Whether a second registration was meant cannot be seen from here, only inferred.
    """
    p = problem()

    @p.register.objective
    def objective(arg):
        return 0.0

    @p.register.objective
    def other(arg):
        return 1.0

    assert p._objective_function is other


def test_a_replaced_callback_is_the_one_that_solves() -> None:
    """Not just stored: the second registration is what the solve runs."""
    p = solvable()

    @p.register.objective
    def objective(arg):
        return arg[p.phases.slide].final.time

    assert p.solve().converged


# ------------------------------------------------------- the problem must be complete


def test_a_problem_with_no_objective_is_incomplete() -> None:
    """A problem states what is to be made smallest; without it there is nothing to solve."""

    class Only(yapss.Phase):
        state: State
        control: Control
        time: yapss.Independent

    class OnlyPhase(yapss.Phases):
        only: Only

    p = yapss.Problem("p", phases=OnlyPhase)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]

    ph.time.guess = (0.0, 1.0)
    with raises(
        ValueError, "the problem is not ready to solve", "no objective callback", at="validate"
    ):
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

    class Only(yapss.Phase):
        state: State
        control: Control
        time: yapss.Independent

    class OnlyPhase(yapss.Phases):
        only: Only

    p = yapss.Problem("p", phases=OnlyPhase, discrete=Discrete)
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]

    @p.register.objective
    def objective(arg):
        return arg[ph].final.x

    ph.time.guess = (0.0, 1.0)
    p.discrete.d.bounds = (0.0, 0.0)
    p.discrete.e.bounds[:] = (0.0, 0.0)
    with raises(ValueError, "no discrete callback", at="validate"):
        p.validate()


def test_phases_may_be_omitted_entirely() -> None:
    """Zero is a count: a problem with no phases declares none, rather than an empty class."""

    class Parameters(yapss.Parameter):
        x = yapss.scalar()

    p = yapss.Problem("p", parameter=Parameters)
    assert list(p.phases) == []

    @p.register.objective
    def objective(arg):
        return arg.parameter.x**2

    p.parameter.x.guess = 1.0
    p.validate()


def test_the_complaints_are_reported_together() -> None:
    """One message lists everything incomplete, so a user fixes them in one pass."""
    p = problem()
    with pytest.raises(ValueError) as info:
        p.validate()
    message = str(info.value)
    assert message.count("\n") >= 2, message
    assert "(1) " in message and "(2) " in message, message
    assert "no objective callback" in message
    assert "phase 'first' has no continuous callback" in message


def test_one_complaint_is_not_numbered() -> None:
    """The numbering is there to separate complaints, so one of them does not get a '(1)'."""

    class Design(yapss.Parameter):
        a = yapss.scalar()

    p = yapss.Problem("p", parameter=Design)
    with pytest.raises(ValueError) as info:
        p.validate()
    message = str(info.value)
    assert message.endswith("the problem has no objective callback"), message
    assert "(1)" not in message


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
    """Each role is a declaration, and the message names the slot that was not given one."""
    with raises(TypeError, "Wrong.state is annotated", "yapss.State", at="class Wrong"):

        class Wrong(yapss.Phase):
            state: object  # type: ignore[assignment]
            time: yapss.Independent


def test_a_phase_has_a_state() -> None:
    """The other roles may be omitted, and a phase with none of them still has a state."""
    with raises(TypeError, "Bare declares no state", "state: ", at="class Bare"):

        class Bare(yapss.Phase):
            control: Control
            time: yapss.Independent


def test_a_phase_has_one_namespace() -> None:
    """A state and a control may not share a name: a callback reaches both from one `arg`."""

    class Same(yapss.Control):
        x = yapss.scalar()

    with raises(ValueError, "both declare 'x'", "one namespace", at="class Clash"):

        class Clash(yapss.Phase):
            state: State
            control: Same
            time: yapss.Independent


def test_the_independent_variable_shares_that_namespace() -> None:
    """Naming it after a state is the same collision, and the message says how to rename."""
    with raises(ValueError, "one namespace", "yapss.Independent", at="class Clash"):

        class Clash(yapss.Phase):
            state: State
            control: Control
            x: yapss.Independent


def test_a_phase_names_its_independent_variable() -> None:
    """It is not defaulted to ``time``: which name fits is the user's problem's to say.

    A phase always has exactly one, so nothing is saved by guessing at its name, and a shape
    that names it is a shape a reader -- and a type checker -- can see whole.
    """
    with raises(TypeError, "does not name its independent variable", at="class Untimed"):

        class Untimed(yapss.Phase):
            state: State
            control: Control


def test_a_phase_takes_one_independent_variable() -> None:
    """Two would leave no way to say which a guess or a bound is about."""
    with raises(TypeError, "one independent variable", "'r', 's'", at="class Twice"):

        class Twice(yapss.Phase):
            state: State
            r: yapss.Independent
            s: yapss.Independent


def test_the_independent_variable_is_not_a_phase_attribute() -> None:
    """A phase already has a ``mesh``, so a variable of that name could not be reached."""
    with raises(TypeError, "already a phase's own attribute", at="class Meshed"):

        class Meshed(yapss.Phase):
            state: State
            mesh: yapss.Independent  # type: ignore[assignment]


def test_a_misspelled_role_is_refused_with_a_suggestion() -> None:
    """`controls: Control` is a typo, not a new role and not an independent variable."""
    with raises(
        TypeError, "Typo.controls is not one of", "Did you mean 'control'?", at="class Typo"
    ):

        class Typo(yapss.Phase):
            state: State
            controls: Control
            time: yapss.Independent


def test_a_role_under_an_unrelated_name_is_named() -> None:
    """With nothing close to suggest, the vector's own role says which slot was meant."""
    with raises(TypeError, "Control is a yapss.Control: did you mean 'control'?", at="class Odd"):

        class Odd(yapss.Phase):
            state: State
            thrust: Control
            time: yapss.Independent


def test_anything_else_annotated_on_a_phase_is_refused() -> None:
    """An annotation that is neither a role nor the independent variable says what both are."""
    with raises(TypeError, "is annotated float", "yapss.Independent", at="class Odd"):

        class Odd(yapss.Phase):
            state: State
            r: float
            time: yapss.Independent


def test_a_phase_is_annotated_not_assigned() -> None:
    """``state = State`` for ``state: State`` would leave the slot empty, so it is refused."""
    with raises(TypeError, "Old.state is assigned", "annotated, not assigned", at="class Old"):

        class Old(yapss.Phase):
            state = State


def test_a_phase_s_shape_may_not_be_inherited_from() -> None:
    """A shape is declared whole where it is used, as a vector's fields are."""

    class Shape(yapss.Phase):
        state: State
        time: yapss.Independent

    with raises(TypeError, "cannot inherit from Shape", at="class Child"):

        class Child(Shape):
            control: Control


def test_the_base_phase_is_not_a_phase() -> None:
    """``yapss.Phase`` is a shape to subclass, so naming a phase with it is refused."""
    with raises(TypeError, "is not a phase's shape", at="class Bare"):

        class Bare(yapss.Phases):
            only: yapss.Phase


def test_a_phases_declaration_holds_only_phases() -> None:
    """Each phase is annotated with its shape, and anything else is refused, naming it."""
    with raises(
        TypeError, "Wrong.first is annotated State", "subclass of yapss.Phase", at="class Wrong"
    ):

        class Wrong(yapss.Phases):
            first: State


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
    with raises(TypeError, "cannot inherit from Phases", at="class Child"):

        class Child(Phases):
            pass


def test_a_phase_is_annotated_not_assigned_among_phases() -> None:
    """An assignment among the phases is refused, and the message gives the annotation."""
    with raises(
        TypeError, "Assigned.first is assigned", "'first: <a yapss.Phase", at="class Assigned"
    ):

        class Assigned(yapss.Phases):
            first = State


def test_a_phase_callback_must_be_callable() -> None:
    """The phase's own registry checks what it is handed, as the problem's does."""
    p = problem()
    with raises(TypeError, "callback must be callable", at="register.continuous"):
        p.phases.first.register.continuous(3)


def test_a_phase_callback_registered_twice_replaces() -> None:
    """A phase's callback is a setting like the problem's; see `test_registering_twice_replaces`."""
    p = problem()
    ph = p.phases.first

    @ph.register.continuous
    def continuous(arg, out):
        pass

    @ph.register.continuous
    def other(arg, out):
        pass

    assert ph._continuous is other


@pytest.mark.parametrize("name", ["", "   "])
def test_the_name_is_not_blank(name: str) -> None:
    """Messages and the solution's repr call the problem by its name."""
    with raises(ValueError, "must not be blank", at="yapss.Problem"):
        yapss.Problem(name, phases=Phases)
    p = problem()
    with raises(ValueError, "must not be blank", at="p.name"):
        p.name = name


def test_a_callback_of_the_wrong_arity_is_refused_where_it_is_registered() -> None:
    """The solve would otherwise fail inside YAPSS's call, in Python's own words."""
    p = problem()
    with raises(TypeError, "continuous callback for phase 'first'", "(arg, out)", at="register"):
        p.phases.first.register.continuous(lambda arg: None)
    with raises(TypeError, "objective callback", "(arg)", at="register"):
        p.register.objective(lambda arg, out: 0.0)


def test_a_callback_may_take_extra_parameters_with_defaults() -> None:
    """Only what YAPSS passes has to be accepted; a defaulted extra is the user's business."""
    p = problem()
    p.phases.first.register.continuous(lambda arg, out, gain=1.0: None)
    p.register.objective(lambda arg, scale=1.0: 0.0)
