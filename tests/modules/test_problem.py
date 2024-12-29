"""

Test the yapss._private.problem module.

"""

# standard library imports
import re

# third party imports
import pytest

# package imports
from yapss import Problem
from yapss._private.problem import ScalePhase
from yapss.examples import dynamic_soaring, rosenbrock


def test_derivatives_options():
    ocp = rosenbrock.setup()

    # misspelled method option
    ocp.derivatives.method = "auto"
    msg = (
        "The value 'misspelled' is not allowed. Allowed values are in "
        "('auto', 'central-difference', 'central-difference-full', 'user')"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.derivatives.method = "misspelled"

    # misspelled order option
    ocp.derivatives.order = "first"
    msg = "The value 'frst' is not allowed. Allowed values are in ('first', 'second')"
    with pytest.raises(ValueError, match=re.escape(msg)):
        ocp.derivatives.order = "frst"


def test_scale_setup():
    ocp = dynamic_soaring.setup()

    # valid scale
    ocp.scale.parameter = [1]

    # scale must be a sequence
    msg = "Scale 'parameter' must be an array of length 1."
    with pytest.raises(ValueError, match=msg):
        ocp.scale.parameter = 1
    msg = "Scale 'state' in phase 0 must be an array of length 6."
    with pytest.raises(ValueError, match=msg):
        ocp.scale.phase[0].state = 1

    # scale must be a sequence of length 1
    msg = "Scale 'parameter' must be an array of length 1."
    with pytest.raises(ValueError, match=msg):
        ocp.scale.parameter = [1, 2]

    # scale must be a sequence of length 6
    ocp.scale.phase[0].state = [1, 2, 3, 4, 5, 6]
    msg = "Scale 'state' in phase 0 must be an array of length 6."
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
    # valid nx
    Problem(name="test", nx=[1, 2, 3])
    # nx must be a sequence
    # msg = "Keyword 'nx' must be a tuple or list of nonnegative integers, or None."
    msg = "Keyword 'nx' must be a tuple or list of positive integers."
    with pytest.raises(TypeError, match=msg):
        Problem(name="test", nx=1)
    with pytest.raises(ValueError, match=msg):
        Problem(name="test", nx=[1, -2, 3])

    # test keyword ns
    msg = "Argument 'ns' must a nonnegative integer or None."
    Problem(name="test", nx=[1, 2, 3], ns=1)
    with pytest.raises(TypeError, match=msg):
        Problem(name="test", nx=[1, 2, 3], ns="one")
    with pytest.raises(ValueError, match=msg):
        Problem(name="test", nx=[1, 2, 3], ns=-1)

    # test keyword nd
    Problem(name="test", nx=[1, 2, 3], nd=1)
    msg = "Argument 'nd' must a nonnegative integer or None."
    with pytest.raises(TypeError, match=msg):
        Problem(name="test", nx=[1, 2, 3], nd="one")
    with pytest.raises(ValueError, match=msg):
        Problem(name="test", nx=[1, 2, 3], nd=-1)


def test_fraction():
    ocp = dynamic_soaring.setup()
    msg = "fraction must be a sequence of floats"
    with pytest.raises(TypeError, match=msg):
        ocp.mesh.phase[0].fraction = ["a", "b", "c"]
    msg = "Sum of mesh fractions must be close to 1.0. Sum is 1.5"
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].fraction = (0.5, 0.5, 0.5)


def test_collocation_points():
    ocp = dynamic_soaring.setup()
    msg = "collocation_points must be a sequence of positive integers, not 0.5"
    with pytest.raises(TypeError, match=msg):
        ocp.mesh.phase[0].collocation_points = 0.5
    msg = "collocation_points must be a sequence of positive integers"
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].collocation_points = (0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.phase[0].collocation_points = (10, 10, 0)


def test_mesh_validates():
    msg = re.escape("mesh.phase[0].col_points and mesh.phase[0].fraction must be the same length")
    ocp = dynamic_soaring.setup()
    ocp.mesh.phase[0].collocation_points = 10 * [4]
    ocp.mesh.phase[0].fraction = 4 * [0.25]
    with pytest.raises(ValueError, match=msg):
        ocp.mesh.validate()


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
    match = "Keyword 'nu' must be a tuple or list of nonnegative integers, or None."
    with pytest.raises(ValueError, match=match):
        Problem(name="test", nx=[1, 2], nu=[2, -1])
    Problem(name="test", nx=[1, 2], nu=[0, 3])


def test_nq_is_list_of_nonnegative_integers() -> None:
    match = "Keyword 'nq' must be a tuple or list of nonnegative integers, or None."
    with pytest.raises(ValueError, match=match):
        Problem(name="test", nx=[1, 2], nq=[-2, -1])
    Problem(name="test", nx=[1, 2], nq=[0, 0])


def test_nh_is_list_of_nonnegative_integers() -> None:
    match = "Keyword 'nh' must be a tuple or list of nonnegative integers, or None."
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
