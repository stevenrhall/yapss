"""

Test that all example main() functions run without error.

"""

# ruff: noqa: D103 (missing docstring)
# third party imports
import matplotlib.pyplot as plt
import pytest

# package imports
from yapss.examples import (
    brachistochrone,
    brachistochrone_minimal,
    delta_iii_ascent,
    dynamic_soaring,
    goddard_problem_1_phase,
    goddard_problem_3_phase,
    hs071,
    isoperimetric,
    minimum_time_to_climb,
    newton,
    orbit_raising,
    rosenbrock,
)

# run all example main() functions

examples = [
    brachistochrone,
    brachistochrone_minimal,
    delta_iii_ascent,
    dynamic_soaring,
    goddard_problem_1_phase,
    goddard_problem_3_phase,
    hs071,
    isoperimetric,
    minimum_time_to_climb,
    newton,
    orbit_raising,
    rosenbrock,
]


@pytest.mark.parametrize("example", examples)
def test_main_function(example) -> None:
    print("###############", example)
    print("###############", type(example))
    example.main()
    plt.show(block=True)
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__])
