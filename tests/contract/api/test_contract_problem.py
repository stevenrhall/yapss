"""What a problem is, and what refuses to become one.

Everything here happens before a solve: what a problem class accepts, what a phase declaration
accepts, how callbacks are registered, and the completeness check that runs when a solve is
asked for. A message about the *shape* of the problem belongs here; a message about a value
put into that shape belongs to the page for that aspect.
"""

from __future__ import annotations

import copy
import pickle
import threading
from typing import Any

import numpy as np
import pytest

import yapss

from ._api import Control, Discrete, Parameter, Phases, State, problem, raises, solvable

# ------------------------------------------------------ what a problem class accepts


def test_name_must_be_a_string() -> None:
    """The problem's name is a string, and a number is a mistake rather than a label."""

    class P(yapss.Problem):
        phases: Phases

    with raises(TypeError, "the problem name must be a string", at="P(42)"):
        P(42)  # type: ignore[arg-type]


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


def test_the_comment_is_a_string_and_empty_by_default() -> None:
    """Free text recorded with each solution, such as a description of the variant solved."""
    ocp = problem()
    assert ocp.comment == ""
    ocp.comment = "the heavier variant"
    assert ocp.comment == "the heavier variant"


def test_a_comment_that_is_not_a_string_is_refused() -> None:
    ocp = problem()
    with raises(TypeError, "comment is a string", at="ocp.comment ="):
        ocp.comment = 3


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
    """`phases` is annotated with a `yapss.Phases` subclass, and nothing else."""
    with raises(TypeError, "P.phases is annotated", "class Phases(yapss.Phases)", at="class P"):

        class P(yapss.Problem):
            phases: list[int]


def test_discrete_must_be_a_discrete_declaration() -> None:
    """`discrete` is annotated with a subclass of `yapss.Discrete`."""
    with raises(TypeError, "takes a subclass of", "yapss.Discrete", at="class P"):

        class P(yapss.Problem):
            phases: Phases
            discrete: object


def test_parameter_must_be_a_parameter_declaration() -> None:
    """`parameter` is annotated with a subclass of `yapss.Parameter`; the message names it."""
    with raises(TypeError, "P.parameter", "yapss.Parameter", at="class P"):

        class P(yapss.Problem):
            phases: Phases
            parameter: object


def test_a_problem_with_no_decision_variables_is_refused() -> None:
    """Every count may be zero, but not all of them at once: that is the absence of a problem.

    Refused where the problem is built, not where it is solved: the counts come from classes
    and no later statement can change them, so the line that called `Problem` is the line to
    fix. It is not incompleteness -- nothing is missing from a constant objective over no
    variables -- so it is not reported as one.
    """
    with raises(ValueError, "no decision variables", "Declare a phase", at="class Empty"):

        class Empty(yapss.Problem):
            """Nothing to choose."""


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
            control: State


def test_a_swapped_annotation_is_named() -> None:
    """A control annotated as the state is most likely a swap, so the message says which."""
    with raises(TypeError, "Did you mean 'control: Control'?", at="class Swapped"):

        class Swapped(yapss.Phase):
            state: Control


def test_a_declaration_with_no_role_is_refused() -> None:
    """A subclass of the base the six roles share declares fields but no role, so it is refused
    as any vector class, naming the role to declare. The base is not public, but it is reachable
    as ``yapss.State.__bases__[0]``.
    """
    base = yapss.State.__bases__[0]

    class Roleless(base):  # type: ignore[misc, valid-type]
        x = yapss.scalar()

    with raises(
        TypeError,
        "takes a subclass of yapss.State, but Roleless has no role",
        "Declare it as 'class Roleless(yapss.State)'",
        at="class NoRole",
    ):

        class NoRole(yapss.Phase):
            state: Roleless


def test_a_swapped_argument_is_named() -> None:
    """The likeliest way to get a role wrong is to swap two arguments, so that is what is said.

    Changing the class's base would also silence the error, and would be the wrong fix.
    """
    with raises(TypeError, "Did you mean 'parameter: Parameter'?", at="class P"):

        class P(yapss.Problem):
            phases: Phases
            discrete: Parameter


def test_a_parameter_may_not_share_a_name_with_a_phase_variable() -> None:
    """Parameters and a phase's variables are one namespace, so the names must differ."""

    class Shared(yapss.Parameter):
        x = yapss.scalar()

    with raises(ValueError, "declares 'x'", "one namespace", at="class P"):

        class P(yapss.Problem):
            phases: Phases
            parameter: Shared


def test_the_message_names_the_phase_the_parameter_collided_in() -> None:
    """Which phase the name collided in is the actionable part of the message."""

    class Other(yapss.Parameter):
        u = yapss.scalar()

    class Early(yapss.Phase):
        state: State
        control: Control

    class TwoPhases(yapss.Phases):
        early: Early

    with raises(ValueError, "phases.early", at="class P"):

        class P(yapss.Problem):
            phases: TwoPhases
            parameter: Other


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
    with raises(TypeError, "@<problem>.register.objective", at="@p.objective"):

        @p.objective
        def objective(arg):
            return 0.0


def test_the_discrete_settings_object_is_not_a_decorator() -> None:
    """The same for the discrete constraints' settings."""
    p = problem()
    with raises(TypeError, "@<problem>.register.discrete", at="@p.discrete"):

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
        return arg[p.phases.slide].final_time

    assert p.solve().converged


# ------------------------------------------------------- the problem must be complete


def test_a_problem_with_no_objective_is_incomplete() -> None:
    """A problem states what is to be made smallest; without it there is nothing to solve."""

    class Only(yapss.Phase):
        state: State
        control: Control

    class OnlyPhase(yapss.Phases):
        only: Only

    class P(yapss.Problem):
        phases: OnlyPhase

    p = P("p")
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
        return arg[p.phases.first].final_state.x

    for ph in p.phases:
        ph.time.guess = (0.0, 1.0)
    with raises(
        ValueError,
        "phases.first has no continuous callback",
        at="validate",
    ):
        p.validate()


def test_declared_discrete_constraints_need_a_callback() -> None:
    """Declaring a constraint and never computing it is a mistake, not an option."""

    class Only(yapss.Phase):
        state: State
        control: Control

    class OnlyPhase(yapss.Phases):
        only: Only

    class P(yapss.Problem):
        phases: OnlyPhase
        discrete: Discrete

    p = P("p")
    ph = p.phases.only

    @ph.register.continuous
    def continuous(arg, out):
        out.dynamics.x = 0.0
        out.dynamics.y = [0.0, 0.0]

    @p.register.objective
    def objective(arg):
        return arg[ph].final_state.x

    ph.time.guess = (0.0, 1.0)
    p.discrete.d.bounds = (0.0, 0.0)
    p.discrete.e.bounds[:] = (0.0, 0.0)
    with raises(ValueError, "no discrete callback", at="validate"):
        p.validate()


def test_phases_may_be_omitted_entirely() -> None:
    """Zero is a count: a problem with no phases declares none, rather than an empty class."""

    class Parameters(yapss.Parameter):
        x = yapss.scalar()

    class P(yapss.Problem):
        parameter: Parameters

    p = P("p")
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
    assert "phases.first has no continuous callback" in message


def test_one_complaint_is_not_numbered() -> None:
    """The numbering is there to separate complaints, so one of them does not get a '(1)'."""

    class Design(yapss.Parameter):
        a = yapss.scalar()

    class P(yapss.Problem):
        parameter: Design

    p = P("p")
    with pytest.raises(ValueError) as info:
        p.validate()
    message = str(info.value)
    assert message.endswith("'@<problem>.register.objective'"), message
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


def test_a_phase_has_a_state() -> None:
    """The other roles may be omitted, and a phase with none of them still has a state."""
    with raises(TypeError, "Bare declares no state", "state: ", at="class Bare"):

        class Bare(yapss.Phase):
            control: Control


def test_a_phase_has_one_namespace() -> None:
    """A state and a control may not share a name: a callback reaches both from one `arg`."""

    class Same(yapss.Control):
        x = yapss.scalar()

    with raises(ValueError, "both declare 'x'", "one namespace", at="class Clash"):

        class Clash(yapss.Phase):
            state: State
            control: Same


def test_a_misspelled_role_is_refused_with_a_suggestion() -> None:
    """`controls: Control` is a typo, not a new role and not an independent variable."""
    with raises(
        TypeError, "Typo.controls is not one of", "Did you mean 'control'?", at="class Typo"
    ):

        class Typo(yapss.Phase):
            state: State
            controls: Control


def test_a_role_under_an_unrelated_name_is_named() -> None:
    """With nothing close to suggest, the vector's own role says which slot was meant."""
    with raises(TypeError, "Control is a yapss.Control: did you mean 'control'?", at="class Odd"):

        class Odd(yapss.Phase):
            state: State
            thrust: Control


def test_a_phase_is_annotated_not_assigned() -> None:
    """``state = State`` for ``state: State`` would leave the slot empty, so it is refused."""
    with raises(TypeError, "Old.state is assigned", "annotated, not assigned", at="class Old"):

        class Old(yapss.Phase):
            state = State


def test_a_phase_s_shape_may_not_be_inherited_from() -> None:
    """A shape is declared whole where it is used, as a vector's fields are."""

    class Shape(yapss.Phase):
        state: State

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
    with raises(
        AttributeError, "cannot be assigned; a problem's phases are declared", at="p.phases.first"
    ):
        p.phases.first = None


def test_a_registration_is_called_not_assigned() -> None:
    """`register.objective = f` looks reasonable and is not how it is done, so it says how."""
    p = problem()
    with raises(
        AttributeError,
        "is not assigned",
        "'@<problem>.register.objective'",
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
    with raises(TypeError, "must be callable", at="register.continuous"):
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

    class P(yapss.Problem):
        phases: Phases

    with raises(ValueError, "must not be blank", at="P(name)"):
        P(name)
    p = problem()
    with raises(ValueError, "must not be blank", at="p.name"):
        p.name = name


def test_a_callback_of_the_wrong_arity_is_refused_where_it_is_registered() -> None:
    """The solve would otherwise fail inside YAPSS's call, in Python's own words."""
    p = problem()
    with raises(
        TypeError, "continuous callback of phases.first", "line", "(arg, out)", at="register"
    ):
        p.phases.first.register.continuous(lambda arg: None)
    with raises(TypeError, "objective callback", "(arg)", at="register"):
        p.register.objective(lambda arg, out: 0.0)


def test_a_callback_may_take_extra_parameters_with_defaults() -> None:
    """Only what YAPSS passes has to be accepted; a defaulted extra is the user's business."""
    p = problem()
    p.phases.first.register.continuous(lambda arg, out, gain=1.0: None)
    p.register.objective(lambda arg, scale=1.0: 0.0)


@pytest.mark.parametrize(
    ("target", "example"),
    [
        ("objective", "'<problem>.objective.sense = ...'"),
        ("derivatives", "'<problem>.derivatives.method = ...'"),
        ("ipopt_options", "'<problem>.ipopt_options.max_iter = ...'"),
        ("register", "'@<problem>.register.objective'"),
    ],
)
def test_replacing_a_held_object_names_a_real_example(target: str, example: str) -> None:
    """The advice shows a line the user could write, not a placeholder."""
    p = problem()
    with raises(AttributeError, "cannot be replaced", example, at="setattr"):
        setattr(p, target, None)


def test_every_held_object_can_be_named_in_the_advice() -> None:
    """Each object a problem or a phase holds says how it is changed, so the advice for
    replacing it is a line the user could write, whatever kind of object it is."""
    p = problem()
    todo: list[object] = [p, p.phases.first]
    while todo:
        owner = todo.pop()
        for name in getattr(type(owner), "_held", ()):  # only containers hold objects
            advice = owner._advice(name)  # type: ignore[attr-defined]
            assert "cannot be replaced" in advice
            assert "<" not in advice.replace("<problem>.", ""), advice
            todo.append(object.__getattribute__(owner, name))


def test_maximizing_gives_the_negated_optimum_of_minimizing_the_negation() -> None:
    """Sense flips what is optimized, not how: the two optima agree up to sign."""
    minimize = solvable()
    minimize.ipopt_options.print_level = 0
    ph = minimize.phases.slide
    minimize.register.objective(lambda arg: arg[ph].final_time)
    maximize = solvable()
    maximize.ipopt_options.print_level = 0
    qh = maximize.phases.slide
    maximize.register.objective(lambda arg: -arg[qh].final_time)
    maximize.objective.sense = "maximize"
    low, high = minimize.solve().objective, maximize.solve().objective
    assert abs(low + high) < 1e-6 * abs(low)


def test_a_problem_solves_from_a_worker_thread() -> None:
    """Ctrl-C handling needs the main thread; solving elsewhere still works."""
    results: list[float] = []

    def run() -> None:
        p = solvable()
        p.ipopt_options.print_level = 0
        results.append(p.solve().objective)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()
    assert len(results) == 1


def test_too_few_degrees_of_freedom_raises() -> None:
    """Ipopt stops without an iterate (status -10), so there is no solution to return."""

    class K(yapss.Parameter):
        k = yapss.scalar()

    class Twice(yapss.Discrete):
        a = yapss.scalar()
        b = yapss.scalar()

    class P(yapss.Problem):
        discrete: Twice
        parameter: K

    p = P("overdetermined")

    def discrete(arg, out):
        out.discrete.a = arg.parameter.k
        out.discrete.b = 2 * arg.parameter.k

    p.register.objective(lambda arg: arg.parameter.k)
    p.register.discrete(discrete)
    p.discrete.a.bounds = (1.0, 1.0)
    p.discrete.b.bounds = (3.0, 3.0)
    p.ipopt_options.print_level = 0
    with raises(ValueError, "Status -10", "too few degrees of freedom", at="solve"):
        p.solve()


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_a_problem_is_not_copied(copier: Any) -> None:
    """A copy would share the callbacks, which refer to this problem; build it again instead."""
    p = problem()
    with raises(
        TypeError, "a problem cannot be copied", "call the function that builds", at="copier"
    ):
        copier(p)


@pytest.mark.parametrize("copier", [copy.deepcopy, lambda s: pickle.loads(pickle.dumps(s))])
def test_a_solution_copies_and_pickles_completely(copier: Any) -> None:
    """A solution is data only: its copy answers the original problem's handles and names."""
    p = solvable()
    p.ipopt_options.print_level = 0
    solution = p.solve()
    copied = copier(solution)
    ph = p.phases.slide
    assert copied.objective == solution.objective
    assert np.array_equal(copied.phases[ph].state.x, solution.phases[ph].state.x)
    assert np.array_equal(copied.phases["slide"].time, solution.phases["slide"].time)


# ------------------------------------------------------------------ a phase by position or name


def test_a_phase_is_reached_by_name_as_by_attribute() -> None:
    """A name that arrives as data reaches the same handle the attribute does."""
    p = problem()
    assert p.phases["first"] is p.phases.first
    assert p.phases[0] is p.phases.first
    assert p.phases[np.int64(1)] is p.phases.second


def test_an_unknown_phase_name_is_suggested() -> None:
    """A misspelled name is told what is there, as the solution's name lookup is."""
    p = problem()
    with raises(KeyError, "no phase named 'frist'", "Did you mean 'first'", at='p.phases["frist"]'):
        p.phases["frist"]  # noqa: B018


def test_a_phase_position_out_of_range_names_the_count() -> None:
    """A position past the last phase says how many there are."""
    p = problem()
    with raises(IndexError, "the problem has 2 phases", "no phase 5", at="p.phases[5]"):
        p.phases[5]  # noqa: B018


@pytest.mark.parametrize("key", [1.0, True, slice(0, 1)])
def test_a_phase_is_reached_by_position_or_name_only(key: object) -> None:
    """Anything else is refused in YAPSS's words, not a list's."""
    p = problem()
    with raises(TypeError, "by its position or its name", at="p.phases[key]"):
        p.phases[key]  # type: ignore[index]


# ------------------------------------------------------------ what a problem class holds


def test_a_problem_class_is_not_subclassed_again() -> None:
    """A variant's inherited declarations would not carry the setup its instance was given."""

    class First(yapss.Problem):
        phases: Phases

    with raises(TypeError, "cannot inherit from First", "Subclass yapss.Problem", at="class"):

        class Second(First):
            pass


def test_a_problem_class_holds_its_declarations_only() -> None:
    """Every other name on a problem is YAPSS's, so a method of the user's would collide."""
    with raises(TypeError, "is defined in a problem class", at="class"):

        class WithMethod(yapss.Problem):
            phases: Phases

            def helper(self) -> None:
                pass


def test_the_base_problem_class_is_not_instantiated() -> None:
    with raises(TypeError, "is subclassed, not instantiated", at="yapss.Problem("):
        yapss.Problem("bare")


def test_an_annotation_naming_nothing_is_refused_naming_the_class() -> None:
    """A declaration names classes declared before it; the message says which class."""
    with raises(NameError, "Late: an annotation names 'Undeclared'", at="class"):

        class Late(yapss.Phase):
            state: Undeclared  # type: ignore[name-defined]  # noqa: F821


def test_a_vector_s_settings_are_not_called() -> None:
    """Only the aspects that answer a call -- the objective's and the discrete's -- take one."""
    ph = problem().phases.first
    with raises(TypeError, "holds settings and is not callable", at="ph.state("):
        ph.state()
