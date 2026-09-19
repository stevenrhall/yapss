"""A phase's states, controls, independent variable and the problem's parameters are one space.

That is what the columns of a phase's Jacobian are, and a derivative names a variable from it
without saying which vector it came from, so a name belonging to two of them would name two
columns. The namespace is assembled in two steps and each step checks what it adds: `phase()`
has the classes in front of it, and `Problem` is where the parameters arrive.
"""

import pytest

import yapss


class State(yapss.Vector):
    h = yapss.field()
    v = yapss.field()


class Control(yapss.Vector):
    u = yapss.field()


class Phases(yapss.Phases):
    only = yapss.phase(state=State, control=Control)


def vector(*names):
    """Return a vector declaration with the given field names."""
    return type("Made", (yapss.Vector,), {name: yapss.field() for name in names})


# --- what phase() checks ------------------------------------------------------------------


def test_a_state_and_a_control_may_not_share_a_name():
    with pytest.raises(ValueError, match=r"both declare 'v'.*one namespace"):
        yapss.phase(state=State, control=vector("v"))


def test_a_state_may_not_be_called_time():
    with pytest.raises(ValueError, match="declares 'time' as a state"):
        yapss.phase(state=vector("time"))


def test_a_state_may_not_share_the_independent_variable_s_name():
    with pytest.raises(ValueError, match=r"declares 'r' as a state.*which you named"):
        yapss.phase(state=vector("r"), r=yapss.field())


def test_a_control_may_not_share_the_independent_variable_s_name():
    with pytest.raises(ValueError, match="declares 'r' as a control"):
        yapss.phase(state=State, control=vector("r"), r=yapss.field())


def test_path_and_integral_names_are_outside_the_namespace():
    # they are outputs, so they appear on the other side of a derivative
    yapss.phase(state=State, control=Control, path=vector("h"), integral=vector("u"))


def test_a_name_may_be_a_control_in_one_phase_and_a_state_in_another():
    # which matters when a quantity is commanded during one phase and coasts through the next
    class Mixed(yapss.Phases):
        burn = yapss.phase(state=State, control=vector("thrust"))
        coast = yapss.phase(state=vector("thrust"), control=Control)

    assert [phase.name for phase in Mixed()] == ["burn", "coast"]


# --- what Problem() checks ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "role"),
    [("h", "a state"), ("u", "a control"), ("time", "its independent variable")],
)
def test_a_parameter_may_not_share_a_phase_variable_s_name(name, role):
    with pytest.raises(ValueError, match=rf"declares '{name}'.*phase 'only' also has as {role}"):
        yapss.Problem("p", phases=Phases, parameter=vector(name))


def test_a_parameter_that_collides_with_no_phase_is_accepted():
    problem = yapss.Problem("p", phases=Phases, parameter=vector("wind"))
    assert problem.parameter.bounds._fields == ("wind",)


def test_a_discrete_name_may_match_a_phase_variable():
    # discrete constraints are outputs, like path and integral names
    problem = yapss.Problem("p", phases=Phases, discrete=vector("h"))
    assert problem.discrete.bounds._fields == ("h",)
