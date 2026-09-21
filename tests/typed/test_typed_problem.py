"""The strictly typed problem runs: its annotations are evaluated, and it solves."""

import pytest

from yapss.examples import brachistochrone_minimal

from .typed_problem import setup


def test_the_typed_problem_solves_to_the_untyped_answer() -> None:
    """The same brachistochrone, with the landing a discrete constraint and gravity a parameter."""
    solution = setup().solve()
    untyped = brachistochrone_minimal.setup()
    untyped.ipopt_options.print_level = 0
    assert solution.converged
    assert solution.objective == pytest.approx(untyped.solve().objective, rel=1e-8)
