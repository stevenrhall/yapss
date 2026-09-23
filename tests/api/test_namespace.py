"""A phase's states, controls, independent variable and the problem's parameters are one space.

That is what the columns of a phase's Jacobian are, and a derivative names a variable from it
without saying which vector it came from, so a name belonging to two of them would name two
columns. The namespace is assembled in two steps and each step checks what it adds: a phase's
shape has the classes in front of it, and `Problem` is where the parameters arrive.
"""

import pytest

import yapss
from yapss._api.fields import aspects_of


class State(yapss.State):
    h = yapss.scalar()
    v = yapss.scalar()


class Control(yapss.Control):
    u = yapss.scalar()


class Only(yapss.Phase):
    state: State
    control: Control
    time: yapss.Independent


class Phases(yapss.Phases):
    only: Only


def made(role, *names):
    """Return a declaration of `role` with the given scalar field names."""
    return type("Made", (role,), {name: yapss.scalar() for name in names})


def shape(**annotations):
    """Return a phase's shape with the given annotations, as a class body would declare it.

    A shape names its independent variable, so `time` is supplied unless the caller names one.
    """
    if not any(value is yapss.Independent for value in annotations.values()):
        annotations["time"] = yapss.Independent
    return type("Shape", (yapss.Phase,), {"__annotations__": annotations})


# --- what a phase's shape checks ----------------------------------------------------------


def test_a_state_and_a_control_may_not_share_a_name():
    with pytest.raises(ValueError, match=r"both declare 'v'.*one namespace"):
        shape(state=State, control=made(yapss.Control, "v"))


def test_a_state_may_not_be_called_time():
    with pytest.raises(ValueError, match=r"its state Made declares 'time'.*which you named"):
        shape(state=made(yapss.State, "time"), time=yapss.Independent)


def test_a_state_may_not_share_the_independent_variable_s_name():
    with pytest.raises(ValueError, match=r"its state Made declares 'r'.*which you named"):
        shape(state=made(yapss.State, "r"), r=yapss.Independent)


def test_a_control_may_not_share_the_independent_variable_s_name():
    with pytest.raises(ValueError, match="its control Made declares 'r'"):
        shape(state=State, control=made(yapss.Control, "r"), r=yapss.Independent)


def test_path_and_integral_names_are_outside_the_namespace():
    # they are outputs, so they appear on the other side of a derivative
    shape(
        state=State, control=Control, path=made(yapss.Path, "h"), integral=made(yapss.Integral, "u")
    )


def test_a_name_may_be_a_control_in_one_phase_and_a_state_in_another():
    # which matters when a quantity is commanded during one phase and coasts through the next
    burn = shape(state=State, control=made(yapss.Control, "thrust"))
    coast = shape(state=made(yapss.State, "thrust"), control=Control)
    mixed = type("Mixed", (yapss.Phases,), {"__annotations__": {"burn": burn, "coast": coast}})
    assert [phase.name for phase in mixed()] == ["burn", "coast"]


# --- what Problem() checks ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "role"),
    [("h", "a state"), ("u", "a control"), ("time", "its independent variable")],
)
def test_a_parameter_may_not_share_a_phase_variable_s_name(name, role):
    with pytest.raises(ValueError, match=rf"declares '{name}'.*phase 'only' also has as {role}"):
        yapss.Problem("p", phases=Phases, parameter=made(yapss.Parameter, name))


def test_a_parameter_that_collides_with_no_phase_is_accepted():
    problem = yapss.Problem("p", phases=Phases, parameter=made(yapss.Parameter, "wind"))
    assert aspects_of(problem.parameter).bounds._fields == ("wind",)


def test_a_discrete_name_may_match_a_phase_variable():
    # discrete constraints are outputs, like path and integral names
    problem = yapss.Problem("p", phases=Phases, discrete=made(yapss.Discrete, "h"))
    assert aspects_of(problem.discrete).bounds._fields == ("h",)
