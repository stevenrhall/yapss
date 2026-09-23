"""Every count may be zero, and the one combination that cannot.

Zero is a count: a problem may declare no phases, a phase no states and no control, a vector
no fields. Nothing about the transcription changes shape at the bottom of any of those ranges
on its own, and these tests hold it there. The exception is all of them at once -- with
neither a phase nor a parameter the nonlinear program has no variables, which is the absence
of a problem rather than a degenerate one, and is refused with the same message whichever
derivative method is asked for.

A phase contributes its initial and final time whatever else it declares, and a fixed time is
a variable bounded above and below by the same number, so "a phase with no variables in it" is
not a thing that can be written. `test_a_phase_that_declares_nothing_still_has_its_own_times`
is what holds that, and it is what makes the check in `validate` equivalent to the fact.
"""

import pytest

import yapss

METHODS = ("auto", "central-difference", "central-difference-full")


def test_a_phase_that_declares_nothing_still_has_its_own_times():
    """A phase contributes an initial and a final time even with no state and no control."""

    class P(yapss.Phase):
        state: yapss.State

    class Phases(yapss.Phases):
        p: P

    problem = yapss.Problem("bare phase", phases=Phases)
    ph = problem.phases.p

    @ph.register.continuous
    def dynamics(arg, out):
        return out

    @problem.register.objective
    def objective(arg):
        return arg[ph].final.time

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 2.0)
    ph.time.guess = (0.0, 1.0)
    problem.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(1.0)


def test_the_endpoints_alone_make_an_optimization_problem():
    """A phase that declares nothing is still a nonlinear program in `t0` and `tf`.

    The stronger form of the test above: minimizing the final time lands on a bound, which a
    clipped value would do as well, while this lands in the *interior* of both ranges. Only a
    pair of genuine decision variables, optimized over, can do that -- which is the premise
    the zero-variable check in `validate` rests on.
    """

    class Interval(yapss.Phase):
        state: yapss.State

    class Phases(yapss.Phases):
        interval: Interval

    problem = yapss.Problem("endpoints alone", phases=Phases)
    ph = problem.phases.interval

    @ph.register.continuous
    def nothing_happens(arg, out):
        return out

    @problem.register.objective
    def objective(arg):
        return (arg[ph].initial.time - 0.3) ** 2 + (arg[ph].final.time - 1.7) ** 2

    ph.time.initial = (0.0, 1.0)
    ph.time.final = (1.0, 2.0)
    ph.time.guess = (0.0, 1.5)
    problem.ipopt_options.print_level = 0

    solution = problem.solve()
    assert solution.objective == pytest.approx(0.0, abs=1e-16)
    assert solution[ph].initial.time == pytest.approx(0.3)
    assert solution[ph].final.time == pytest.approx(1.7)


def test_a_parameter_of_no_rows_is_not_a_variable():
    """A declaration can be non-empty and still contribute nothing.

    `vector(0)` is a vector field with no components, which is legal so that a declaration built
    by an algorithm needs no special case. Counting fields rather than rows would let this
    reach the derivative setup, where it fails inside CasADi.
    """

    class Design(yapss.Parameter):
        a = yapss.vector(0)

    class Phases(yapss.Phases):
        pass

    problem = yapss.Problem("a parameter of no rows", phases=Phases, parameter=Design)

    @problem.register.objective
    def objective(arg):
        return 0.0

    assert Design._fields == ("a",)
    assert Design._nrows == 0
    problem.ipopt_options.print_level = 0
    with pytest.raises(ValueError, match="no decision variables"):
        problem.solve()


def test_a_problem_of_parameters_alone_solves():
    """No phases at all is an ordinary nonlinear program, which is how hs071 is written."""

    class Design(yapss.Parameter):
        a = yapss.scalar()

    class Phases(yapss.Phases):
        pass

    problem = yapss.Problem("parameters only", phases=Phases, parameter=Design)

    @problem.register.objective
    def objective(arg):
        return (arg.parameter.a - 2.0) ** 2

    problem.parameter.a.bounds = (-10.0, 10.0)
    problem.parameter.a.guess = 0.0
    problem.ipopt_options.print_level = 0
    assert problem.solve().objective == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_a_problem_with_no_variables_is_refused_the_same_way(method):
    """Neither a phase nor a parameter leaves nothing to solve for.

    The message must come from `validate`, before any derivative setup: under ``"auto"`` the
    objective would otherwise trace to a constant and fail inside CasADi, naming a C++ header
    and a type the user never wrote.
    """

    class Phases(yapss.Phases):
        pass

    problem = yapss.Problem("nothing at all", phases=Phases)

    @problem.register.objective
    def objective(arg):
        return 0.0

    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    with pytest.raises(ValueError, match="no decision variables") as info:
        problem.solve()
    assert "declare a phase or a parameter" in str(info.value)
