"""The strictly typed problem runs: its annotations are evaluated, and it solves."""

import numpy as np
import pytest

from yapss.examples import brachistochrone_minimal

from .typed_problem import setup, solve_and_report


def test_the_typed_problem_solves_to_the_untyped_answer() -> None:
    """The same brachistochrone, with the landing a discrete constraint and gravity a parameter."""
    solution = setup().solve()
    untyped = brachistochrone_minimal.setup()
    untyped.ipopt_options.print_level = 0
    assert solution.converged
    assert solution.objective == pytest.approx(untyped.solve().objective, rel=1e-8)


def test_the_typed_solution_reads() -> None:
    """Every read in `report` runs: the annotations say what is there, and it is."""
    values = solve_and_report()
    assert values["landing"] == pytest.approx(1.0)
    np.testing.assert_array_equal(values["costate"], values["same costate"])
    assert values["iterations"] > 0
