"""Contract: `Problem` construction and configuration.

What a user may do
    - Construct with keyword arguments only: `name`, `nx` (a list or tuple of per-phase
      state counts), and optionally `nu`, `nq`, `nh` (same length as `nx`, default zeros)
      and `ns`, `nd` (default 0). A problem may have no phases, or phases with no states.
    - Read the counts and the number of phases back.
    - Set `spectral_method`, `sense`, `derivatives.method`, `derivatives.order`, and
      `catch_keyboard_interrupt` to any of their allowed values.
    - Put anything in `auxdata`.
    - Set a phase's mesh as `collocation_points` (integers >= 2) and `fraction` (positive
      reals summing to 1, rescaled to exactly 1 if within 0.01), including a single segment.

What a user may get wrong
    - A positional argument, a misspelled keyword (with a suggestion), a missing or empty
      name, a negative count, per-phase counts of the wrong length: raise at construction.
    - Reassigning a count or a top-level attribute (`bounds`, `guess`, `scale`,
      `functions`, `mesh`, `derivatives`, `ipopt_options`, `auxdata`, `name`), or a
      misspelled attribute on `Problem`, `Derivatives`, `UserFunctions`, `MeshPhase`:
      `AttributeError` at the assignment.
    - A value outside the allowed set: `ValueError` at the assignment.
    - `collocation_points` below 2 or not integers: `ValueError` at the assignment.
      A `fraction` that does not sum to 1: `ValueError` at the assignment.
      Mismatched lengths: `ValueError` from `validate()`.
"""

from __future__ import annotations

import numpy as np
import pytest

from yapss import Problem

from ._contract import callback_problem, not_yet, raises

TOP_LEVEL = [
    "nx",
    "nu",
    "nq",
    "nh",
    "ns",
    "nd",
    "np",
    "name",
    "bounds",
    "guess",
    "scale",
    "functions",
    "mesh",
    "derivatives",
    "ipopt_options",
    "auxdata",
]
CHOICES = {
    "spectral_method": ("lgr", "lg", "lgl"),
    "sense": ("minimize", "maximize"),
    "catch_keyboard_interrupt": (True, False),
}
DERIVATIVE_CHOICES = {
    "method": ("auto", "user", "central-difference", "central-difference-full"),
    "order": ("first", "second"),
}


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
def test_construction_with_counts(container):
    ocp = Problem(
        name="counts",
        nx=container([2, 1]),
        nu=container([1, 0]),
        nq=container([0, 2]),
        nh=container([1, 1]),
        ns=3,
        nd=4,
    )
    assert (ocp.nx, ocp.nu, ocp.nq, ocp.nh) == ((2, 1), (1, 0), (0, 2), (1, 1))
    assert (ocp.ns, ocp.nd, ocp.np) == (3, 4, 2)


def test_optional_counts_default_to_zero():
    ocp = Problem(name="defaults", nx=[2, 1])
    assert (ocp.nu, ocp.nq, ocp.nh, ocp.ns, ocp.nd) == ((0, 0), (0, 0), (0, 0), 0, 0)


@pytest.mark.parametrize("nx", [[], [0]], ids=["no phases", "phase with no states"])
def test_degenerate_problems_can_be_constructed(nx):
    ocp = Problem(name="degenerate", nx=nx, ns=1)
    assert ocp.np == len(nx)


@pytest.mark.parametrize("attribute", CHOICES)
def test_settings_accept_every_allowed_value(attribute):
    ocp = callback_problem()
    for value in CHOICES[attribute]:
        setattr(ocp, attribute, value)
        assert getattr(ocp, attribute) == value


@pytest.mark.parametrize("attribute", DERIVATIVE_CHOICES)
def test_derivative_settings_accept_every_allowed_value(attribute):
    ocp = callback_problem()
    for value in DERIVATIVE_CHOICES[attribute]:
        setattr(ocp.derivatives, attribute, value)
        assert getattr(ocp.derivatives, attribute) == value


def test_auxdata_accepts_anything():
    ocp = callback_problem()
    ocp.auxdata.g0 = 9.81
    ocp.auxdata.table = {"a": [1, 2, 3]}
    assert ocp.auxdata.g0 == 9.81


def test_mesh_defaults_to_ten_uniform_segments_of_ten_points():
    phase = Problem(name="mesh", nx=[1]).mesh.phase[0]
    assert tuple(phase.collocation_points) == 10 * (10,)
    assert tuple(phase.fraction) == pytest.approx(10 * (0.1,))


@pytest.mark.parametrize(
    ("collocation_points", "fraction"),
    [
        ([4, 6], [0.3, 0.7]),
        ((4, 6), (0.3, 0.7)),
        ([6], [1.0]),
        ([4, 4], np.array([0.5, 0.5])),
        ([4, 4], [np.float64(0.5), np.float64(0.5)]),
    ],
    ids=["lists", "tuples", "single segment", "ndarray fraction", "numpy float64 fraction"],
)
def test_mesh_accepts_documented_forms(collocation_points, fraction):
    ocp = callback_problem()
    phase = ocp.mesh.phase[0]
    phase.collocation_points = collocation_points
    phase.fraction = fraction
    ocp.mesh.validate()
    assert list(phase.collocation_points) == list(collocation_points)
    assert sum(phase.fraction) == pytest.approx(1.0)
    assert ocp.solve().nlp_info.ipopt_status == 0


def test_fraction_off_by_rounding_is_rescaled_to_sum_to_one():
    """Only rounding-level error is promised to be absorbed (see the E8 tolerance xfail)."""
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [4, 4]
    phase.fraction = [0.5, 0.5 - 1e-12]
    assert sum(phase.fraction) == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------- what a user may get wrong


def test_positional_arguments_are_rejected():
    with raises(TypeError, "positional"):
        Problem("name", [1])  # type: ignore[misc]


def test_misspelled_keyword_is_rejected():
    # The "Did you mean 'nh'?" suggestion comes from the interpreter (Python 3.13 and
    # later), not from YAPSS, so only the rejection itself is part of the contract.
    with raises(TypeError, "nhh"):
        Problem(name="typo", nx=[1], nhh=[1])  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("kwargs", "exc", "fragment"),
    [
        ({"nx": [1]}, TypeError, "name"),
        ({"name": "", "nx": [1]}, ValueError, "name"),
        ({"name": "c", "nx": [-1]}, ValueError, "nx"),
        ({"name": "c", "nx": [1, 1], "nu": [1]}, ValueError, "nu"),
        ({"name": "c", "nx": [2.0]}, TypeError, "nx"),
    ],
    ids=["missing name", "empty name", "negative count", "count length", "float count"],
)
def test_invalid_construction_raises(kwargs, exc, fragment):
    with raises(exc, fragment):
        Problem(**kwargs)


@pytest.mark.parametrize("attribute", TOP_LEVEL)
def test_counts_and_top_level_attributes_cannot_be_reassigned(attribute):
    ocp = callback_problem()
    with raises(AttributeError, attribute, at="setattr"):
        setattr(ocp, attribute, None)


@pytest.mark.parametrize(
    ("owner", "typo"),
    [
        (lambda p: p, "spectral_metod"),
        (lambda p: p.derivatives, "methd"),
        (lambda p: p.functions, "continous"),
        (lambda p: p.mesh.phase[0], "fracton"),
    ],
    ids=["Problem", "Derivatives", "UserFunctions", "MeshPhase"],
)
def test_misspelled_attribute_raises_at_the_assignment(owner, typo):
    ocp = callback_problem()
    with raises(AttributeError, typo, at="setattr"):
        setattr(owner(ocp), typo, None)


@pytest.mark.parametrize(
    ("owner", "attribute", "value"),
    [
        (lambda p: p, "spectral_method", "LGR"),
        (lambda p: p, "sense", "max"),
        (lambda p: p.derivatives, "method", "Auto"),
        (lambda p: p.derivatives, "order", "third"),
    ],
    ids=["spectral_method", "sense", "derivatives.method", "derivatives.order"],
)
def test_value_outside_the_allowed_set_raises_at_the_assignment(owner, attribute, value):
    ocp = callback_problem()
    with raises(ValueError, repr(value), "not allowed", at="setattr"):
        setattr(owner(ocp), attribute, value)


@pytest.mark.parametrize("value", [[1], [4.0], [4, 1]], ids=["below 2", "float", "one below 2"])
def test_invalid_collocation_points_raise_at_the_assignment(value):
    phase = callback_problem().mesh.phase[0]
    with raises(ValueError, "collocation_points", at="collocation_points ="):
        phase.collocation_points = value


def test_fraction_not_summing_to_one_raises_at_the_assignment():
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [4, 4]
    with raises(ValueError, "Sum of mesh fractions", at="fraction ="):
        phase.fraction = [0.5, 0.4]


def test_mismatched_mesh_lengths_are_reported_by_validate():
    ocp = callback_problem()
    ocp.mesh.phase[0].collocation_points = [4, 4]  # the fraction has three segments
    with raises(ValueError, "mesh.phase[0]"):
        ocp.mesh.validate()


# ----------------------------------------------------------------- not yet met


@not_yet("W3", "counts accept NumPy integers and other integer sequences")
@pytest.mark.parametrize(
    "nx", [[np.int64(2)], np.array([2]), range(2)], ids=["numpy int", "ndarray", "range"]
)
def test_counts_accept_integer_like_values(nx):
    assert len(Problem(name="counts", nx=nx).nx) == len(nx)


@not_yet("W3", "a bool count is rejected at construction with a message showing the value")
@pytest.mark.parametrize("kwargs", [{"nx": [True]}, {"nx": [1], "ns": True}], ids=["nx", "ns"])
def test_bool_count_is_rejected_helpfully(kwargs):
    with raises(TypeError, "True"):
        Problem(name="bool", **kwargs)


@not_yet("E7a", "ns and nd no longer accept None")
@pytest.mark.parametrize("name", ["ns", "nd"])
def test_none_count_is_rejected(name):
    with raises(TypeError, name):
        Problem(name="none", nx=[1], **{name: None})


@not_yet("W3 LimitOptions", "an invalid value message names the attribute")
def test_invalid_setting_message_names_the_attribute():
    ocp = callback_problem()
    with raises(ValueError, "spectral_method"):
        ocp.spectral_method = "LGR"


@not_yet("W3 LimitOptions", "an array or a non-bool is not an allowed value")
@pytest.mark.parametrize(
    ("attribute", "value"),
    [("spectral_method", np.array(["lg"])), ("catch_keyboard_interrupt", 1)],
    ids=["ndarray", "int for a bool"],
)
def test_setting_rejects_values_of_the_wrong_type(attribute, value):
    ocp = callback_problem()
    with raises((TypeError, ValueError), attribute):
        setattr(ocp, attribute, value)


@not_yet("W3 LimitOptions", "class access to a setting returns the descriptor")
def test_class_access_to_a_setting():
    assert Problem.sense is not None


@not_yet("E9", "a private backing name cannot be used to bypass validation")
def test_private_backing_names_are_not_writable():
    ocp = callback_problem()
    with raises(AttributeError, "_spectral_method", at="_spectral_method ="):
        ocp._spectral_method = "LGX"


@not_yet("E9", "a misspelled attribute message suggests the intended name")
def test_misspelled_attribute_message_suggests_the_name():
    ocp = callback_problem()
    with raises(AttributeError, "spectral_method"):
        ocp.spectral_metod = "lg"


@not_yet("W3 / E9", "Mesh rejects misspelled attributes and mesh.phase cannot be reassigned")
@pytest.mark.parametrize("attribute", ["phses", "phase"])
def test_mesh_is_protected(attribute):
    ocp = callback_problem()
    with raises(AttributeError, attribute, at="setattr"):
        setattr(ocp.mesh, attribute, ())


@not_yet("W3", "deleting a real callback says to set it to None; a bogus name is unknown")
def test_function_deletion_messages():
    ocp = callback_problem()
    with raises(AttributeError, "set to None"):
        del ocp.functions.continuous
    with raises(AttributeError) as info:
        del ocp.functions.bogus
    assert "set to None" not in str(info.value)


@not_yet("W3", "Problem has a repr naming the problem")
def test_problem_repr_names_the_problem():
    assert "callbacks" in repr(callback_problem())


@not_yet("W3", "validate() reports every problem at once, not only the first")
def test_validate_reports_every_problem():
    ocp = Problem(name="incomplete", nx=[1])
    with pytest.raises(ValueError) as info:
        ocp.validate()
    message = str(info.value)
    assert "guess.phase[0].time" in message
    assert "functions.objective" in message


@not_yet("E8 fixes", "fraction accepts any real number type")
def test_fraction_accepts_numpy_float32():
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [4, 4]
    phase.fraction = [np.float32(0.5), np.float32(0.5)]


@not_yet("E8 fixes", "collocation_points accepts NumPy integers")
def test_collocation_points_accept_numpy_integers():
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [np.int64(4)]


@not_yet("E8 fixes", "a non-positive fraction raises ValueError naming the phase")
def test_negative_fraction_raises_value_error_naming_the_phase():
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [4, 4]
    with raises(ValueError, "mesh.phase[0].fraction", at="fraction ="):
        phase.fraction = [1.5, -0.5]


@not_yet("E8 fixes", "an empty collocation_points is rejected")
def test_empty_collocation_points_is_rejected():
    phase = callback_problem().mesh.phase[0]
    with raises(ValueError, "collocation_points", at="collocation_points ="):
        phase.collocation_points = []


@not_yet("E8 fixes", "the rescaling tolerance only absorbs rounding error (about 1e-9)")
def test_fraction_off_by_more_than_rounding_raises():
    phase = callback_problem().mesh.phase[0]
    phase.collocation_points = [4, 4]
    with raises(ValueError, "fraction", at="fraction ="):
        phase.fraction = [0.5, 0.495]


@not_yet("E8 fixes", "the length-mismatch message uses the public name collocation_points")
def test_mismatched_lengths_message_uses_public_names():
    ocp = callback_problem()
    ocp.mesh.phase[0].collocation_points = [4, 4]  # the fraction has three segments
    with raises(ValueError, "collocation_points"):
        ocp.mesh.validate()


@not_yet("E8 fixes", "more than about 100 collocation points in a segment warns")
def test_very_large_segment_warns():
    phase = callback_problem().mesh.phase[0]
    with pytest.warns(Warning, match="collocation"):
        phase.collocation_points = [1000]


@not_yet("F5 target", "callbacks can be registered with a decorator")
def test_callback_decorator_registration():
    ocp = callback_problem()

    @ocp.register.objective
    def objective(arg):
        arg.objective = arg.phase[0].final_time

    assert ocp.functions.objective is objective
