"""What a mesh is, and what a mesh that cannot be collocated on says."""

from __future__ import annotations

import yapss

from ._api import problem, raises

# ------------------------------------------------------------------- setting a phase's mesh


def test_a_phase_takes_a_mesh_object() -> None:
    """The uniform mesh is the common case, and is built by naming its two counts."""
    ph = problem().phases.first
    ph.mesh = yapss.Mesh.uniform(segments=4, points=6)
    assert len(ph.mesh.segments) == 4


def test_a_mesh_is_not_a_number() -> None:
    """A count alone does not say which of the two it is, and the message shows both."""
    ph = problem().phases.first
    with raises(TypeError, "must be a Mesh", "yapss.Mesh.uniform(", at="ph.mesh"):
        ph.mesh = 5


def test_a_mesh_needs_at_least_one_segment() -> None:
    """Zero segments is no mesh at all."""
    with raises(ValueError, "must be a positive integer", at="Mesh.uniform"):
        yapss.Mesh.uniform(segments=0, points=5)


def test_a_segment_needs_at_least_two_collocation_points() -> None:
    """One point cannot carry a polynomial, and the message names the segment."""
    with raises(ValueError, "at least 2 collocation points", at="Mesh.uniform"):
        yapss.Mesh.uniform(segments=2, points=1)


def test_a_segment_is_a_fraction_and_a_count() -> None:
    """Given by hand, a segment is a pair, and the message names which one was not."""
    with raises(TypeError, "is not a (fraction, points) pair", at="yapss.Mesh"):
        yapss.Mesh([(0.5,)])


def test_segments_may_be_given_by_hand() -> None:
    """The general form: each segment's share of the phase and its own number of points."""
    mesh = yapss.Mesh([(0.25, 4), (0.75, 8)])
    assert len(mesh.segments) == 2


def test_segments_are_pairs_not_a_number() -> None:
    """A mesh is a sequence of segments; one number is not one."""
    with raises(TypeError, "Mesh segments are (fraction, points) pairs", at="yapss.Mesh"):
        yapss.Mesh(5)


def test_a_mesh_needs_at_least_one_segment() -> None:
    """An empty mesh has nowhere to collocate."""
    with raises(ValueError, "at least one segment", at="yapss.Mesh"):
        yapss.Mesh([])


def test_the_number_of_points_is_an_integer() -> None:
    """A count of collocation points, so a fraction of one is a mistake."""
    with raises(TypeError, "collocation points must be an integer", at="yapss.Mesh"):
        yapss.Mesh([(1.0, 4.5)])


def test_a_fraction_is_positive() -> None:
    """A segment of no extent is not a segment."""
    with raises(ValueError, "the fraction must be positive", at="yapss.Mesh"):
        yapss.Mesh([(0.0, 4), (1.0, 4)])


def test_the_fractions_sum_to_one() -> None:
    """They divide the phase, so they account for all of it, and the message does the sum."""
    with raises(ValueError, "must sum to 1", at="yapss.Mesh"):
        yapss.Mesh([(0.25, 4), (0.25, 4)])
