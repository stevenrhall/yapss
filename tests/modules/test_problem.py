"""

Test the yapss._private.problem module.

"""

# standard library imports
import re

# third party imports
import numpy as np
import pytest

# package imports
from yapss._legacy import Problem
from yapss._legacy.examples import brachistochrone_minimal, dynamic_soaring, rosenbrock
from yapss._private.problem import ScalePhase


def test_derivatives_options():
    ocp = rosenbrock.setup()

    # misspelled method option
    ocp.derivatives.method = "auto"
    msg = (
        "The value 'misspelled' is not allowed for 'method'. Allowed values are "
        "('auto', 'central-difference', 'central-difference-full', 'user')."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.derivatives.method = "misspelled"

    # misspelled order option
    ocp.derivatives.order = "first"
    msg = "The value 'frst' is not allowed for 'order'. Allowed values are ('first', 'second')."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.derivatives.order = "frst"


def test_scale_setup():
    ocp = dynamic_soaring.setup()

    # valid scale
    ocp.scale.parameter = [1]

    # scale must be a sequence
    msg = "Scale 'parameter' must have length 1"
    with pytest.raises(ValueError, match=msg):
        ocp.scale.parameter = 1
    msg = "Scale 'state' in phase 0 must have length 6"
    with pytest.raises(ValueError, match=msg):
        ocp.scale.phase[0].state = 1

    # scale must be a sequence of length 1
    msg = "Scale 'parameter' must have length 1"
    with pytest.raises(ValueError, match=msg):
        ocp.scale.parameter = [1, 2]

    # scale must be a sequence of length 6
    ocp.scale.phase[0].state = [1, 2, 3, 4, 5, 6]
    msg = "Scale 'state' in phase 0 must have length 6"
    with pytest.raises(ValueError, match=msg):
        ocp.scale.phase[0].state = [1, 2, 3, 4, 5]


def test_name_keyword():
    msg = "Value of keyword 'name' must be a nonempty string."
    with pytest.raises(ValueError, match=msg):
        Problem(name="", nx=[1])
    with pytest.raises(TypeError, match=msg):
        # noinspection PyTypeChecker
        Problem(name=1, nx=[1])


# noinspection PyTypeChecker
def test_keywords():
    # valid nx, including a phase with zero states (see test_auto.test_no_dynamics
    # for a fully solved example of this edge case)
    Problem(name="test", nx=[1, 2, 3])
    Problem(name="test", nx=[1, 0, 3])
    # nx must be a sequence of nonnegative integers
    with pytest.raises(TypeError, match="nx must be a sequence of integers"):
        Problem(name="test", nx=1)
    with pytest.raises(ValueError, match=re.escape("nx[1] must be at least 0")):
        Problem(name="test", nx=[1, -2, 3])

    # test keyword ns
    Problem(name="test", nx=[1, 2, 3], ns=1)
    with pytest.raises(TypeError, match="ns must be an integer"):
        Problem(name="test", nx=[1, 2, 3], ns="one")
    with pytest.raises(ValueError, match="ns must be a nonnegative integer"):
        Problem(name="test", nx=[1, 2, 3], ns=-1)

    # test keyword nd
    Problem(name="test", nx=[1, 2, 3], nd=1)

    with pytest.raises(TypeError, match="nd must be an integer"):
        Problem(name="test", nx=[1, 2, 3], nd="one")
    with pytest.raises(ValueError, match="nd must be a nonnegative integer"):
        Problem(name="test", nx=[1, 2, 3], nd=-1)


def test_fraction():
    ocp = dynamic_soaring.setup()
    msg = re.escape("mesh.phase[0].fraction must be a real number")
    with pytest.raises(TypeError, match=msg):
        ocp.mesh.phase[0].fraction = ["a", "b", "c"]
    msg = re.escape("mesh.phase[0].fraction must sum to 1, but sums to 1.5")
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].fraction = (0.5, 0.5, 0.5)


def test_collocation_points():
    ocp = dynamic_soaring.setup()
    msg = re.escape("mesh.phase[0].collocation_points must be a sequence of integers")
    with pytest.raises(TypeError, match=msg):
        ocp.mesh.phase[0].collocation_points = 0.5
    # a float count is the wrong type; a count below the floor is the wrong value
    msg = re.escape("mesh.phase[0].collocation_points[0] must be an integer, got a float")
    with pytest.raises(TypeError, match=msg):
        ocp.mesh.phase[0].collocation_points = (0.5, 0.5, 0.5)
    msg = re.escape("mesh.phase[0].collocation_points[2] must be at least 2")
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].collocation_points = (10, 10, 0)
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].collocation_points = (10, 10, 1)
    ocp.mesh.phase[0].collocation_points = (10, 10, 2)


def test_mesh_validates():
    msg = re.escape("mesh.phase[0].collocation_points has 10 segments")
    ocp = dynamic_soaring.setup()
    ocp.mesh.phase[0].collocation_points = 10 * [4]
    ocp.mesh.phase[0].fraction = 4 * [0.25]
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.validate()


def _scale_problem() -> Problem:
    """Return a problem with at least one entry in every scale array."""
    return Problem(name="scale", nx=[2, 1], nu=[1, 1], nq=[1, 1], nh=[1, 1], ns=1, nd=1)


_SCALE_ARRAYS = [
    (lambda s: s.phase[0].state, "scale.phase[0].state"),
    (lambda s: s.phase[1].control, "scale.phase[1].control"),
    (lambda s: s.phase[0].integral, "scale.phase[0].integral"),
    (lambda s: s.phase[1].dynamics, "scale.phase[1].dynamics"),
    (lambda s: s.phase[0].path, "scale.phase[0].path"),
    (lambda s: s.discrete, "scale.discrete"),
    (lambda s: s.parameter, "scale.parameter"),
]


@pytest.mark.parametrize(("get_array", "name"), _SCALE_ARRAYS)
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_validate_catches_element_assignment(get_array, name, bad):
    """A write around the array's checks still reaches `Scale.validate`, which reports it."""
    ocp = _scale_problem()
    ocp.scale.validate()
    array = get_array(ocp.scale)
    array.view(np.ndarray)[-1] = bad
    index = len(array) - 1
    with pytest.raises(ValueError, match=re.escape(f"{name}[{index}] must be finite and positive")):
        ocp.scale.validate()


def test_scale_validate_catches_slice_assignment():
    ocp = _scale_problem()
    ocp.scale.phase[0].state.view(np.ndarray)[:] = [1.0, 0.0]
    msg = "scale.phase[0].state[1] must be finite and positive, got 0.0."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.scale.validate()


def test_problem_validate_checks_scale():
    """`Problem.validate`, which `solve` runs first, reports a zero scale set around the checks.

    Before this check, the zero reached Ipopt as an infinite scaling factor and the solve
    crashed the process with no Python traceback. This test calls `validate` rather than
    `solve` so that a regression fails the test instead of killing the test process.
    """
    ocp = brachistochrone_minimal.setup()
    ocp.validate()
    ocp.scale.phase[0].dynamics.view(np.ndarray)[0] = 0.0
    msg = "scale.phase[0].dynamics[0] must be finite and positive, got 0.0."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.validate()


def test_scale_array():
    """Test ScaleArray descriptor class."""
    msg = "attribute 'state' can be accessed on instance objects only."
    with pytest.raises(AttributeError, match=msg):
        state = ScalePhase.state


ocp = Problem(
    name="Test_Property",
    nx=[1, 2, 3],
    nu=[2, 3, 4],
    nq=[3, 4, 5],
    nh=[4, 5, 6],
    ns=3,
    nd=4,
)


def test_nx():
    assert ocp.nx == (1, 2, 3)
    with pytest.raises(AttributeError):
        ocp.nx = None


def test_nu():
    assert ocp.nu == (2, 3, 4)
    with pytest.raises(AttributeError):
        ocp.nu = None


def test_nq():
    assert ocp.nq == (3, 4, 5)
    with pytest.raises(AttributeError):
        ocp.nq = None


def test_nh():
    assert ocp.nh == (4, 5, 6)
    with pytest.raises(AttributeError):
        ocp.nh = None


def test_ns():
    assert ocp.ns == 3
    with pytest.raises(AttributeError):
        ocp.ns = None


def test_nd():
    assert ocp.nd == 4
    with pytest.raises(AttributeError):
        ocp.nd = None


def test_name_is_string() -> None:
    with pytest.raises(TypeError):
        Problem(name=10, nx=[2])  # type: ignore


def test_nx_is_list() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx="ten")  # type: ignore


def test_nu_is_list() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nu=2)  # type: ignore


def test_nq_is_list() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nq=2)  # type: ignore


def test_nh_is_list() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nh=3)  # type: ignore


def test_nu_is_length_np() -> None:
    with pytest.raises(ValueError, match="Length of 'nu' must be the same as length of 'nx'."):
        Problem(name="test", nx=[1, 2], nu=[2])
    Problem(name="test", nx=[1, 2], nu=[2, 2])


def test_nq_is_length_np() -> None:
    with pytest.raises(ValueError, match="Length of 'nq' must be the same as length of 'nx'."):
        Problem(name="test", nx=[1, 2], nq=[2])
    Problem(name="test", nx=[1, 2], nq=[2, 2])


def test_nh_is_length_np() -> None:
    with pytest.raises(ValueError, match="Length of 'nh' must be the same as length of 'nx'."):
        Problem(name="test", nx=[1, 2], nh=[2])
    Problem(name="test", nx=[1, 2], nh=[2, 3])


def test_nu_is_list_of_integers() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nu=[2, 2.0])  # type: ignore


def test_nq_is_list_of_integers() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nq=[2, "a"])  # type: ignore


def test_nh_is_list_of_integers() -> None:
    with pytest.raises(TypeError):
        Problem(name="test", nx=[1, 2], nh=[None, 2])  # type: ignore


def test_nu_is_list_of_nonnegative_integers() -> None:
    match = re.escape("nu[")
    with pytest.raises(ValueError, match=match):
        Problem(name="test", nx=[1, 2], nu=[2, -1])
    Problem(name="test", nx=[1, 2], nu=[0, 3])


def test_nq_is_list_of_nonnegative_integers() -> None:
    match = re.escape("nq[")
    with pytest.raises(ValueError, match=match):
        Problem(name="test", nx=[1, 2], nq=[-2, -1])
    Problem(name="test", nx=[1, 2], nq=[0, 0])


def test_nh_is_list_of_nonnegative_integers() -> None:
    match = re.escape("nh[")
    with pytest.raises(ValueError, match=match):
        Problem(name="test", nx=[1, 2], nh=[-2, -1])
    Problem(name="test", nx=[1, 2], nh=[1, 0])


def test_of_getters() -> None:
    ocp = Problem(name="test", nx=[1, 2], nu=[3, 4], nq=[5, 6], nh=[7, 8], nd=9, ns=10)
    assert ocp.nx == (1, 2)
    assert ocp.nu == (3, 4)
    assert ocp.nq == (5, 6)
    assert ocp.nh == (7, 8)
    assert ocp.nd == 9
    assert ocp.ns == 10
