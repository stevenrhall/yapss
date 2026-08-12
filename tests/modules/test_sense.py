"""

Test `problem.sense`: attribute validation, the `scale.objective` positivity it
requires, the Ipopt options YAPSS now reserves for itself, and the KKT-level claim
the whole feature rests on -- that flipping `sense` flips the reported
multiplier/costate sign to match, rather than the reported objective value.

"""

from __future__ import annotations

import pytest

import yapss
from yapss.examples import goddard_problem_1_phase, goddard_problem_3_phase, orbit_raising


def _tiny_problem() -> yapss.Problem:
    """A `Problem` just big enough to construct; never solved in these tests."""
    return yapss.Problem(name="t", nx=[0], nu=[0])


# --- attribute validation --------------------------------------------------------


class TestSenseAttribute:
    def test_default_is_minimize(self) -> None:
        assert _tiny_problem().sense == "minimize"

    def test_maximize_is_settable(self) -> None:
        problem = _tiny_problem()
        problem.sense = "maximize"
        assert problem.sense == "maximize"

    def test_invalid_value_raises(self) -> None:
        problem = _tiny_problem()
        with pytest.raises(ValueError, match="not allowed"):
            problem.sense = "bogus"


class TestScaleObjectivePositivity:
    def test_default_is_one(self) -> None:
        assert _tiny_problem().scale.objective == 1.0

    def test_positive_value_is_settable(self) -> None:
        problem = _tiny_problem()
        problem.scale.objective = 8000.0
        assert problem.scale.objective == 8000.0

    @pytest.mark.parametrize("value", [0.0, -1.0, -8000.0])
    def test_nonpositive_value_raises(self, value: float) -> None:
        problem = _tiny_problem()
        with pytest.raises(ValueError, match="must be positive"):
            problem.scale.objective = value


# --- reserved Ipopt options -------------------------------------------------------


class TestReservedIpoptOptions:
    """YAPSS manages these directly; setting them must fail on assignment.

    `nlp_scaling_method` is hardcoded to `"user-scaling"` in `solver.py`, and under
    `user-scaling` Ipopt never consults the `obj_scaling_factor` *option* at all --
    only `set_problem_scaling`'s values matter. Before this reservation existed,
    setting `obj_scaling_factor` was therefore accepted and silently did nothing.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "nlp_scaling_method",
            "obj_scaling_factor",
            "hessian_approximation",
            "warm_start_init_point",
        ],
    )
    def test_raises_on_assignment(self, name: str) -> None:
        problem = _tiny_problem()
        with pytest.raises(ValueError, match="managed by YAPSS"):
            setattr(problem.ipopt_options, name, "anything")

    def test_obj_scaling_factor_does_not_silently_compose(self) -> None:
        """Regression guard for the specific pre-`sense` footgun.

        It would be worse for `obj_scaling_factor` to now silently compose with
        `sense` than to keep silently doing nothing -- a user who set it would have
        no way to tell whether it took effect. It must fail loudly instead.
        """
        problem = _tiny_problem()
        with pytest.raises(ValueError):
            problem.ipopt_options.obj_scaling_factor = 2.0

    def test_ordinary_option_is_unaffected(self) -> None:
        problem = _tiny_problem()
        problem.ipopt_options.max_iter = 500
        assert problem.ipopt_options.get_options()["max_iter"] == 500


# --- numerical check: multiplier/costate sign tracks sense, not the objective -----


def _orbit_raising_with_manual_negation() -> yapss.Problem:
    """The pre-`sense` (wrong) way to maximize: negate, sense left at the default.

    Mathematically the identical NLP to `orbit_raising.setup()` -- same feasible
    region, same optimum -- so it exists only to compare multiplier signs against.
    """
    problem = orbit_raising.setup()
    problem.sense = "minimize"

    def objective(arg: yapss.ObjectiveArg) -> None:
        arg.objective = -arg.phase[0].final_state[0]

    problem.functions.objective = objective
    return problem


def _quiet(problem: yapss.Problem) -> yapss.Problem:
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.sb = "yes"
    return problem


@pytest.fixture(scope="module")
def solutions() -> tuple[yapss.Solution, yapss.Solution]:
    """Solve both formulations once and share the result across the comparisons below."""
    sense_solution = _quiet(orbit_raising.setup()).solve()
    negated_solution = _quiet(_orbit_raising_with_manual_negation()).solve()
    return sense_solution, negated_solution


class TestMultiplierSignTracksSense:
    """`orbit_raising.setup()` uses `sense = "maximize"`; compare it against the
    manual-negation convention it replaced.

    Both formulations describe the identical NLP, so the primal solution must
    agree exactly. (An earlier version of this test used the isoperimetric
    problem for this comparison; dropped because its optimal curve is a circle
    free to start at any point along its own circumference -- a continuous
    symmetry with no reason for two independent solves to land on the same
    point of, even though both are equally optimal. `orbit_raising` has no such
    symmetry: its initial state is fully pinned and its dynamics are not
    rotationally invariant.) The KKT claim under test is mu = dJ/dc for
    whichever quantity each formulation treats as "the objective":
    `sense="maximize"` reports the true (positive) final radius, so its
    multipliers must be the negative of the manual-negation formulation's,
    which reports the negative final radius.
    """

    def test_primal_solution_agrees(self, solutions: tuple[yapss.Solution, yapss.Solution]) -> None:
        sense_solution, negated_solution = solutions
        assert sense_solution.phase[0].state == pytest.approx(
            negated_solution.phase[0].state,
            abs=1e-6,
        )

    def test_objective_is_negated_between_formulations(
        self, solutions: tuple[yapss.Solution, yapss.Solution]
    ) -> None:
        sense_solution, negated_solution = solutions
        assert sense_solution.objective == pytest.approx(-negated_solution.objective, rel=1e-6)

    def test_discrete_multiplier_sign_is_flipped(
        self, solutions: tuple[yapss.Solution, yapss.Solution]
    ) -> None:
        sense_solution, negated_solution = solutions
        assert sense_solution.discrete_multiplier == pytest.approx(
            -negated_solution.discrete_multiplier,
            abs=1e-6,
        )

    def test_path_multiplier_sign_is_flipped(
        self, solutions: tuple[yapss.Solution, yapss.Solution]
    ) -> None:
        sense_solution, negated_solution = solutions
        assert sense_solution.phase[0].path_multiplier == pytest.approx(
            -negated_solution.phase[0].path_multiplier,
            abs=1e-6,
        )

    def test_costate_sign_is_flipped(
        self, solutions: tuple[yapss.Solution, yapss.Solution]
    ) -> None:
        sense_solution, negated_solution = solutions
        assert sense_solution.phase[0].costate == pytest.approx(
            -negated_solution.phase[0].costate,
            abs=1e-6,
        )


# --- regression: objective values unchanged by the goddard migration --------------


class TestGoddardObjectiveRegression:
    """`goddard_problem_1_phase`/`goddard_problem_3_phase` moved from
    `scale.objective = -1` to `sense = "maximize"` in the same release that added
    `sense`. `obj_scale` is mathematically identical either way -- sign and
    magnitude are just factored apart -- so the optimal objective value must be
    unchanged. These are the values the examples produced before migration.

    `test_one_phase`'s value was later updated again, for an unrelated reason:
    `goddard_problem_1_phase.setup()`'s initial guess was changed (altitude ramping
    0 -> hmax, thrust ramping Tm -> 0, to match the notebook) after the sense
    migration. The one-phase problem has a known singular arc with a noisy,
    non-unique interior solution, so a different guess lands on a slightly
    different nearby point -- confirmed by `test_three_phase`, whose guess was not
    touched and whose value is still exactly the pre-migration one.
    """

    def test_one_phase(self) -> None:
        problem = goddard_problem_1_phase.setup()
        problem.ipopt_options.tol = 1e-8
        solution = _quiet(problem).solve()
        assert solution.objective == pytest.approx(18565.096736988486, rel=1e-6)

    def test_three_phase(self) -> None:
        problem = goddard_problem_3_phase.setup()
        problem.ipopt_options.tol = 1e-8
        solution = _quiet(problem).solve()
        assert solution.objective == pytest.approx(18550.871863824497, rel=1e-6)
