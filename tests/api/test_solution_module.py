"""The public `yapss.solution` module names the classes a solution is made of."""

import pytest

import yapss
import yapss.solution
from yapss.examples.goddard_problem_3_phase import setup


@pytest.fixture(scope="module")
def solved():
    problem = setup()
    problem.ipopt_options.print_level = 0
    return problem, problem.solve()


def test_each_class_is_the_type_of_what_it_names(solved):
    """A class exported for annotations is the class of the attribute it annotates."""
    problem, solution = solved
    ps = solution.phases[problem.phases.boost]
    reads = {
        "Solution": solution,
        "PhaseSolution": ps,
        "ProblemMultiplier": solution.multiplier,
        "PhaseMultiplier": ps.multiplier,
        "NLPRecord": solution.nlp,
        "NLPIndex": solution.nlp.index,
        "ProblemVariableIndex": solution.nlp.index.variable,
        "ProblemConstraintIndex": solution.nlp.index.constraint,
        "Jacobian": solution.nlp.jac_g,
        "NLPScale": solution.nlp.scale,
        "Convergence": solution.nlp.convergence,
        "PhaseNLP": ps.nlp,
        "PhaseIndex": ps.nlp.index,
        "VariableIndex": ps.nlp.index.variable,
        "ConstraintIndex": ps.nlp.index.constraint,
        "PhasePoint": ps.nlp.point,
        "Settings": solution.settings,
        "SettingsGroup": solution.settings.phases,
        "Callback": solution.settings.callbacks.objective,
        "Run": solution.run,
        "Seconds": solution.run.seconds,
    }
    assert sorted(reads) == sorted(yapss.solution.__all__)
    for name, value in reads.items():
        assert type(value) is getattr(yapss.solution, name), name


def test_solution_and_phase_solution_are_the_top_level_classes():
    assert yapss.solution.Solution is yapss.Solution
    assert yapss.solution.PhaseSolution is yapss.PhaseSolution


def test_the_time_settings_class_is_not_exported():
    """`ph.time` is reached, never named: nothing in a problem's declaration annotates it."""
    assert not hasattr(yapss, "Independent")
    assert "Independent" not in yapss.__all__


def test_a_settings_container_prints_its_label(solved):
    """A container prints the label its messages use, not the default object repr."""
    problem, _ = solved
    assert repr(problem.phases.boost.time) == "<phases.boost.time>"
    assert repr(problem.derivatives) == "<derivatives>"
    assert repr(problem.objective) == "<objective>"
    assert repr(problem.register) == "<register>"
