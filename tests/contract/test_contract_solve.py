"""Contract: `Problem.solve()`, Ipopt options, status, and warnings.

What a user may do
    - Solve, re-solve the same problem, and solve from a worker thread.
    - Maximize with `problem.sense = "maximize"`.
    - Set any Ipopt option to a value of its kind (NumPy integers included; an int for a
      Number option), and unset one by assigning `None`.
    - Rely on `solve()` returning a `Solution` for a run that did not converge, with an
      `IpoptConvergenceWarning` pointing at the user's `solve()` line, on every such solve.
    - Filter that warning with "ignore" or turn it into an error with "error".

What a user may get wrong
    - An Ipopt option value of the wrong kind: `TypeError` at the assignment.
    - Setting an option YAPSS manages (e.g. `obj_scaling_factor`): `ValueError` at the
      assignment, naming the YAPSS setting to use instead.
"""

from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

import yapss
from yapss._legacy import Problem
from yapss._legacy.examples import brachistochrone_minimal

from ._contract import callback_problem, default_objective, not_yet, raises

# Reference optimum of `brachistochrone_minimal`, and the tolerance its example test holds it
# to on every CI runner (tests/examples/test_brachistochrone_minimal.py).
BRACHISTOCHRONE_J = 0.312480130
BRACHISTOCHRONE_REL = 1e-8


def unconverged():
    ocp = callback_problem()
    ocp.ipopt_options.max_iter = 1
    return ocp


def problem_inputs(ocp):
    """Everything a user sets on a problem, as plain values that compare with ``==``.

    Callbacks and `auxdata` entries are recorded by identity: what matters is that solve
    replaced nothing, and a user object need not define equality.
    """

    def value(x):
        return np.asarray(x).tolist() if isinstance(x, np.ndarray) else x

    def lower_upper(b):
        return value(b.lower), value(b.upper)

    bound_names = (
        "initial_time", "final_time", "duration", "initial_state", "final_state",
        "state", "control", "integral", "path",
    )  # fmt: skip
    return {
        "settings": (
            ocp.name, ocp.spectral_method, ocp.sense, ocp.catch_keyboard_interrupt,
            ocp.derivatives.method, ocp.derivatives.order, ocp.ipopt_options.get_options(),
        ),  # fmt: skip
        "functions": {
            name: id(getattr(ocp.functions, name))
            for name in dir(ocp.functions)
            if not name.startswith("_")
        },
        "auxdata": {key: id(item) for key, item in vars(ocp.auxdata).items()},
        "bounds": (
            lower_upper(ocp.bounds.discrete),
            lower_upper(ocp.bounds.parameter),
            [[lower_upper(getattr(b, n)) for n in bound_names] for b in ocp.bounds.phase],
        ),
        "guess": (
            value(ocp.guess.parameter),
            [
                [value(getattr(g, n)) for n in ("time", "state", "control", "integral")]
                for g in ocp.guess.phase
            ],
        ),
        "scale": (
            value(ocp.scale.objective),
            value(ocp.scale.parameter),
            value(ocp.scale.discrete),
            [
                [
                    value(getattr(c, n))
                    for n in ("time", "state", "control", "integral", "path", "dynamics")
                ]
                for c in ocp.scale.phase
            ],
        ),
        "mesh": [(value(m.collocation_points), value(m.fraction)) for m in ocp.mesh.phase],
    }


# ---------------------------------------------------------------- what a user may do


def test_solve_leaves_the_problem_unchanged():
    """Nothing a solve writes into the problem can affect the next solve.

    Checked exactly, before any solve result is involved. Through 0.2.2 the `"auto"`
    method wrote into `auxdata`, replacing a user helper, and a second solve then called
    the wrong function.
    """
    ocp = callback_problem()
    ocp.auxdata.helper = lambda x: x
    before = problem_inputs(ocp)
    ocp.solve()
    assert problem_inputs(ocp) == before


def test_consecutive_solves_each_reach_the_known_optimum():
    """Each solve is checked against the reference answer, not against the other.

    Solves are not reproducible run to run on every machine, so a comparison between
    two runs cannot separate left-over state from ordinary variation.
    """
    ocp = brachistochrone_minimal.setup()
    ocp.ipopt_options.print_level = 0
    for _ in range(2):
        solution = ocp.solve()
        assert solution.nlp_info.ipopt_status == 0
        assert solution.objective == pytest.approx(BRACHISTOCHRONE_J, rel=BRACHISTOCHRONE_REL)


def test_solve_from_a_worker_thread():
    result = {}

    def target():
        result["status"] = callback_problem().solve().nlp_info.ipopt_status

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    assert result["status"] == 0


def test_maximize_negated_objective_gives_the_negated_optimum():
    minimize = callback_problem()
    maximize = callback_problem()
    maximize.sense = "maximize"

    def negated(arg):
        default_objective(arg)
        arg.objective = -arg.objective

    maximize.functions.objective = negated
    assert maximize.solve().objective == pytest.approx(-minimize.solve().objective, rel=1e-9)


@pytest.mark.parametrize(
    ("name", "value", "stored"),
    [
        ("max_iter", 200, 200),
        ("max_iter", np.int64(200), 200),
        ("tol", 1e-9, 1e-9),
        ("tol", 1, 1.0),
        ("mu_strategy", "monotone", "monotone"),
    ],
    ids=["int", "numpy int", "float", "int for a Number option", "str"],
)
def test_option_of_the_right_kind_is_accepted(name, value, stored):
    ocp = callback_problem()
    setattr(ocp.ipopt_options, name, value)
    assert getattr(ocp.ipopt_options, name) == stored
    assert type(getattr(ocp.ipopt_options, name)) is type(stored)
    assert ocp.solve().nlp_info.ipopt_status == 0


def test_option_set_to_none_is_unset():
    ocp = callback_problem()
    ocp.ipopt_options.max_iter = 10
    ocp.ipopt_options.max_iter = None
    assert "max_iter" not in ocp.ipopt_options.get_options()


def test_unconverged_solve_returns_a_solution_and_warns_at_the_users_line():
    ocp = unconverged()
    with pytest.warns(yapss.IpoptConvergenceWarning, match=r"Status -1\b") as record:
        solution = ocp.solve()
    assert solution.nlp_info.ipopt_status == -1
    assert record[0].filename == __file__


def test_every_unconverged_solve_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", yapss.IpoptConvergenceWarning)
        for _ in range(3):
            unconverged().solve()
    assert sum(issubclass(w.category, yapss.IpoptConvergenceWarning) for w in caught) == 3


@pytest.mark.parametrize("action", ["ignore", "error"])
def test_convergence_warning_obeys_ignore_and_error_filters(action):
    with warnings.catch_warnings():
        warnings.simplefilter(action, yapss.IpoptConvergenceWarning)
        if action == "ignore":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("ignore", yapss.IpoptConvergenceWarning)
                unconverged().solve()
            assert not caught
        else:
            with pytest.raises(yapss.IpoptConvergenceWarning):
                unconverged().solve()


def test_converged_solve_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptConvergenceWarning)
        assert callback_problem().solve().nlp_info.ipopt_status == 0


# ---------------------------------------------------------- what a user may get wrong


@pytest.mark.parametrize("value", ["100", 1.5, True], ids=["str", "float", "bool"])
def test_option_of_the_wrong_kind_raises_at_the_assignment(value):
    ocp = callback_problem()
    with raises(TypeError, "max_iter", at="max_iter ="):
        ocp.ipopt_options.max_iter = value


@pytest.mark.parametrize(
    ("name", "knob"),
    [
        ("obj_scaling_factor", "problem.objective.scale"),
        ("nlp_scaling_method", "scales set on"),
        ("hessian_approximation", "problem.derivatives.order"),
    ],
)
def test_option_managed_by_yapss_raises_at_the_assignment_naming_the_knob(name, knob):
    """The table of managed options is shared with the 0.4.0 front end, and on this branch its
    messages name 0.4.0's settings: 0.3.0's own wording lives on `release/v0.3.0`.
    """
    ocp = callback_problem()
    with raises(ValueError, name, knob, at="setattr"):
        setattr(ocp.ipopt_options, name, "yes" if name != "obj_scaling_factor" else 2.0)


# ----------------------------------------------------------------- not yet met


def test_misspelled_option_raises_at_the_assignment():
    ocp = callback_problem()
    with raises(AttributeError, "max_iter", at="max_iters ="):
        ocp.ipopt_options.max_iters = 10


@pytest.mark.filterwarnings("ignore::yapss.IpoptOptionSettingWarning")
def test_out_of_range_option_raises_at_solve_start():
    ocp = callback_problem()
    ocp.ipopt_options.max_iter = -1
    with raises(ValueError, "max_iter"):
        ocp.solve()


@pytest.mark.filterwarnings("ignore::yapss.IpoptOptionSettingWarning")
def test_invalid_string_choice_raises_at_solve_start():
    ocp = callback_problem()
    ocp.ipopt_options.mu_strategy = "adaptiv"
    with raises(ValueError, "mu_strategy"):
        ocp.solve()


def test_nan_number_option_raises_at_the_assignment():
    ocp = callback_problem()
    with raises(ValueError, "tol", at="tol ="):
        ocp.ipopt_options.tol = float("nan")


def test_option_container_methods_are_reserved():
    ocp = callback_problem()
    with raises((AttributeError, ValueError), "reset", at="reset ="):
        ocp.ipopt_options.reset = 5


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_status_without_an_iterate_raises():
    """Two equality constraints on one variable: Ipopt stops before iterating (status -10).

    Deterministic on every build, unlike an unavailable linear solver (status -12).
    """
    ocp = Problem(name="too few degrees of freedom", nx=[], ns=1, nd=2)
    ocp.functions.objective = lambda arg: setattr(arg, "objective", arg.parameter[0] ** 2)

    def discrete(arg):
        arg.discrete[:] = (arg.parameter[0], 2.0 * arg.parameter[0])

    ocp.functions.discrete = discrete
    ocp.bounds.discrete.lower = ocp.bounds.discrete.upper = [1.0, 3.0]
    ocp.ipopt_options.print_level = 0
    with raises(ValueError, "degrees of freedom"):
        ocp.solve()


def test_status_enum_and_top_level_status():
    solution = callback_problem().solve()
    assert solution.status == yapss.IpoptStatus(0)
    assert solution.status == 0


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_converged_flag():
    assert callback_problem().solve().converged is True
    assert unconverged().solve().converged is False


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_status_names_the_outcome_and_is_the_nlp_info_status():
    solution = unconverged().solve()
    assert solution.status is yapss.IpoptStatus.MAXIMUM_ITERATIONS_EXCEEDED
    assert solution.status.message == "Maximum Number of Iterations Exceeded."
    assert solution.status.converged is False
    assert solution.nlp_info.ipopt_status is solution.status


def test_yapss_warning_hierarchy():
    for category in (
        yapss.IpoptConvergenceWarning,
        yapss.IpoptOptionSettingWarning,
    ):
        assert issubclass(category, yapss.YapssWarning)
    assert issubclass(yapss.YapssWarning, UserWarning)
    assert issubclass(yapss.YapssDeprecationWarning, FutureWarning)


def test_yapss_error_hierarchy():
    assert issubclass(yapss.UnsupportedMathFunctionError, yapss.YapssError)
    assert issubclass(yapss.UnsupportedMathFunctionError, TypeError)
