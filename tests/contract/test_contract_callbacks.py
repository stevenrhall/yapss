"""Contract: the objective, continuous, and discrete callbacks, and user derivatives.

What a user may do
    - Assign continuous outputs by whole rows: as a tuple or list of rows, a 2-D array, row by
      row (negative indices count from the end), or by slices (any step); each row an
      expression over the phase's points, a list with one value per point, or a constant
      scalar. A constant scalar may fill several rows. In-place operators (`+=`, `-=`, ...)
      work on a row and on the whole output.
    - Assign `arg.objective` a scalar expression, and `arg.discrete` whole, by element, or
      by slice.
    - Loop over `arg.phase_list`, read `arg.auxdata`, and use `yapss.math` or NumPy's
      element-wise functions.
    - Get the same optimal control problem under every derivative method.
    - Under `"user"`, supply derivative entries as scalar expressions or constants.

What a user may get wrong
    - An exception raised inside a callback propagates unchanged, with the traceback at the
      raising line, under every derivative method.
    - A tuple of the wrong number of rows, or rows for an output the phase does not have:
      `ValueError` at the assignment, naming the output, the phase, and the count.
    - A write below row level (an element, part of a row, a column, a mask, `np.copyto`, a
      ufunc's `out=`): `TypeError` at the line, showing the whole-row forms. A row value that
      fits only by broadcasting (a length-1 array over several points, one expression over
      the points into several rows): `ValueError` at the assignment. A row index out of
      range: `IndexError` naming the count.
    - A misspelled output, or an assignment to an input: `AttributeError` at the line.
    - A Python `if` on a problem variable: `TypeError` under `"auto"` (pointing to
      `yapss.math.where`), `ValueError` in the continuous callback under the numeric
      methods.
    - A required callback left unset: `ValueError` from `validate()` naming it. A callback
      that is not callable with one argument: `TypeError` at the assignment.
    - Functions not finite at the initial guess: `ValueError` naming the output.
    - An output row never assigned (at the initial guess): `UnsetOutputWarning` at the
      callback's `def` line, once per row, under every method. A row assigned zero, or
      assigned and then updated in place, does not warn.
    - A continuous callback that is not pointwise (`t - t[0]`, `len`, `mean`, `cumsum`, ...):
      `ValueError` naming the callback and output and stating the rule, under every method;
      differences below the tolerance are not reported.
    - Under `"user"`, an invalid derivative key: `ValueError` (unknown name, wrong length,
      index out of range, parameter not keyed with phase 0) or `TypeError` (a part of the
      wrong type: a key that is not a tuple, a name that is not a string, an index that is
      not an integer), naming the key, the callback, and the phase. NumPy integer indices
      are accepted.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import yapss
import yapss.math as ym
from yapss import Problem
from yapss._private.setup_check import UnsetOutputWarning
from yapss.examples import brachistochrone, goddard_problem_3_phase

from ._contract import (
    G0,
    METHODS,
    callback_problem,
    default_continuous,
    not_yet,
    raises,
)


def solve_objective(ocp):
    solution = ocp.solve()
    assert solution.nlp_info.ipopt_status == 0
    return solution.objective


@pytest.fixture(scope="module")
def baseline():
    """The optimal objective of the default problem, which every method must agree on."""
    return solve_objective(callback_problem("auto"))


def continuous_writing(form):
    """Return a continuous callback that writes the default outputs in the given form."""

    def continuous(arg):
        for p in arg.phase_list:
            _, _, v = arg.phase[p].state
            (u,) = arg.phase[p].control
            rows = (v * ym.cos(u), v * ym.sin(u), G0 * ym.sin(u))
            dynamics = arg.phase[p].dynamics
            if form == "tuple":
                dynamics[:] = rows
            elif form == "list":
                dynamics[:] = list(rows)
            elif form == "row by row":
                dynamics[0], dynamics[1], dynamics[2] = rows
            elif form == "slices":
                dynamics[0:2] = rows[:2]
                dynamics[2:] = rows[2:]
            elif form == "numpy ufuncs":
                dynamics[:] = v * np.cos(u), v * np.sin(u), G0 * np.sin(u)
            elif form == "negative indices":
                dynamics[-3], dynamics[-2], dynamics[-1] = rows
            elif form == "stepped slice":
                dynamics[::2] = rows[0], rows[2]
                dynamics[1] = rows[1]
            elif form == "attribute":
                arg.phase[p].dynamics = rows
            elif form == "2-D array":
                dynamics[:] = np.vstack(rows)
            elif form == "list per point":
                for i, row in enumerate(rows):
                    dynamics[i] = [row[k] for k in range(len(row))]
            elif form == "in-place on rows":
                dynamics[0] = rows[0] / 2
                dynamics[0] += rows[0] / 2
                dynamics[1] = 2 * rows[1]
                dynamics[1] /= 2
                dynamics[2] = 0.0
                dynamics[2] -= -rows[2]
            elif form == "in-place on the output":
                arg.phase[p].dynamics = 0.0
                arg.phase[p].dynamics += rows
            arg.phase[p].path[0] = v
            arg.phase[p].integrand[0] = u**2

    return continuous


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("method", METHODS)
def test_every_method_solves_the_same_problem(method, baseline):
    assert solve_objective(callback_problem(method)) == pytest.approx(baseline, rel=1e-9)


@pytest.mark.parametrize(
    "form",
    [
        "tuple",
        "list",
        "row by row",
        "slices",
        "numpy ufuncs",
        "negative indices",
        "stepped slice",
        "attribute",
        "2-D array",
        "list per point",
        "in-place on rows",
        "in-place on the output",
    ],
)
@pytest.mark.parametrize("method", METHODS)
def test_continuous_outputs_accept_every_documented_form(method, form, baseline):
    ocp = callback_problem(method, continuous=continuous_writing(form))
    assert solve_objective(ocp) == pytest.approx(baseline, rel=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_a_constant_scalar_row_is_broadcast_over_the_points(method):
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].integrand[:] = (0.25,)

    assert callback_problem(method, continuous=continuous).solve().nlp_info.ipopt_status == 0


@pytest.mark.parametrize("method", METHODS)
def test_a_constant_scalar_may_fill_several_rows(method):
    def continuous(arg):
        _, _, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = 0.0
        arg.phase[0].dynamics[0:2] = v * ym.cos(u), v * ym.sin(u)
        arg.phase[0].dynamics[2] = G0 * ym.sin(u)
        arg.phase[0].path[:] = (v,)
        arg.phase[0].integrand[:] = (u**2,)

    assert solve_objective(callback_problem(method, continuous=continuous)) == pytest.approx(
        solve_objective(callback_problem(method)),
        rel=1e-9,
    )


@pytest.mark.parametrize("form", ["element", "slice"])
@pytest.mark.parametrize("method", METHODS)
def test_discrete_accepts_element_and_slice_assignment(method, form, baseline):
    def discrete(arg):
        if form == "element":
            arg.discrete[0] = arg.phase[0].final_state[1]
        else:
            arg.discrete[0:1] = arg.phase[0].final_state[1:2]

    ocp = callback_problem(method, discrete=discrete)
    assert solve_objective(ocp) == pytest.approx(baseline, rel=1e-9)


@pytest.mark.parametrize("method", METHODS)
def test_auxdata_is_the_problems_own_namespace(method):
    def continuous(arg):
        arg.auxdata.seen = True
        default_continuous(arg)

    ocp = callback_problem(method, continuous=continuous)
    ocp.solve()
    assert ocp.auxdata.seen is True


def test_user_derivatives_solve_the_same_problem_as_auto():
    user = brachistochrone.setup()
    user.ipopt_options.print_level = 0
    auto = brachistochrone.setup()
    auto.derivatives.method = "auto"
    auto.ipopt_options.print_level = 0
    assert solve_objective(user) == pytest.approx(solve_objective(auto), rel=1e-12)


def test_user_derivative_entry_may_be_a_constant():
    ocp = brachistochrone.setup()
    ocp.ipopt_options.print_level = 0
    jacobian = ocp.functions.continuous_jacobian

    def continuous_jacobian(arg):
        jacobian(arg)
        arg.phase[0].jacobian[("f", 1), ("x", 2)] = 0.0 + np.sin(arg.phase[0].control[0])

    ocp.functions.continuous_jacobian = continuous_jacobian
    assert ocp.solve().nlp_info.ipopt_status == 0


# ---------------------------------------------------------- what a user may get wrong


@pytest.mark.parametrize("method", METHODS)
def test_exception_in_a_callback_propagates_at_the_raising_line(method):
    def continuous(arg):
        raise ValueError("user error in continuous")

    with raises(ValueError, "user error in continuous", at="raise ValueError"):
        callback_problem(method, continuous=continuous).solve()


@pytest.mark.parametrize("method", METHODS)
def test_wrong_number_of_rows_raises_at_the_assignment(method):
    def continuous(arg):
        _, _, v = arg.phase[0].state
        arg.phase[0].dynamics[:] = (v, v)

    with raises(ValueError, "3 rows", at="dynamics[:] ="):
        callback_problem(method, continuous=continuous).solve()


BELOW_ROW_LEVEL = {
    "element": "arg.phase[0].dynamics[0, 1] = 0.0",
    "column": "arg.phase[0].dynamics[:, 1] = 0.0",
    "mask": "arg.phase[0].dynamics[[True, False, True]] = 0.0",
    "part of a row": "arg.phase[0].dynamics[0][:2] = 0.0",
    "element of a row": "arg.phase[0].dynamics[0][1] = 0.0",
    "in-place on part of a row": "arg.phase[0].dynamics[0][:2] += 1.0",
    "np.copyto": "np.copyto(arg.phase[0].dynamics[0], 0.0)",
    "ufunc out=": "np.multiply(arg.phase[0].dynamics[0], 2.0, out=arg.phase[0].dynamics[0])",
}


@pytest.mark.parametrize("statement", BELOW_ROW_LEVEL.values(), ids=BELOW_ROW_LEVEL.keys())
@pytest.mark.parametrize("method", METHODS)
def test_a_write_below_row_level_raises_at_the_line(method, statement):
    def continuous(arg):
        default_continuous(arg)
        exec(statement, {"arg": arg, "np": np})  # noqa: S102 -- one parametrized statement per case

    with raises(TypeError, "arg.phase[0].dynamics", "whole row", at="exec"):
        callback_problem(method, continuous=continuous).solve()


def test_slice_write_to_an_output_with_no_rows_raises():
    ocp = brachistochrone.setup()
    ocp.derivatives.method = "central-difference"
    continuous = ocp.functions.continuous

    def with_path(arg):
        continuous(arg)
        arg.phase[0].path[:] = arg.phase[0].state[2]

    ocp.functions.continuous = with_path
    with raises(ValueError, "nh", at="path[:] ="):
        ocp.solve()


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
def test_one_row_broadcast_into_several_rows_raises():
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].dynamics[:] = arg.phase[0].state[2]

    with raises(ValueError, "dynamics", at="dynamics[:] ="):
        callback_problem("central-difference", continuous=continuous).solve()


def test_length_one_array_row_raises():
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].integrand[0] = np.atleast_1d(arg.phase[0].control[0])[:1]

    with raises(ValueError, "integrand", at="integrand[0] ="):
        callback_problem("central-difference", continuous=continuous).solve()


def test_row_count_message_names_the_output_and_phase():
    def continuous(arg):
        _, _, v = arg.phase[0].state
        arg.phase[0].dynamics[:] = (v, v)

    with raises(ValueError, "phase 0", "dynamics"):
        callback_problem(continuous=continuous).solve()


@pytest.mark.parametrize("method", METHODS)
def test_a_row_index_out_of_range_raises_naming_the_count(method):
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].path[1] = arg.phase[0].state[2]

    with raises(IndexError, "arg.phase[0].path[1]", "nh = 1", at="path[1] ="):
        callback_problem(method, continuous=continuous).solve()


@pytest.mark.parametrize(
    ("callback", "statement", "typo"),
    [
        ("continuous", "arg.phase[0].dynamic = 0.0", "dynamic"),
        ("continuous", "arg.phase[0].time = 0.0", "time"),
        ("objective", "arg.objectiv = arg.phase[0].final_time", "objectiv"),
        ("objective", "arg.phase[0].objective = arg.phase[0].final_time", "objective"),
        ("discrete", "arg.discret = (0.0,)", "discret"),
    ],
    ids=[
        "ContinuousPhase output typo",
        "ContinuousPhase input",
        "ObjectiveArg typo",
        "DiscretePhase typo",
        "DiscreteArg typo",
    ],
)
def test_misspelled_output_or_input_assignment_raises_at_the_line(callback, statement, typo):
    def body(arg):
        if callback == "continuous":
            default_continuous(arg)
        exec(statement, {"arg": arg})  # noqa: S102 -- one parametrized statement per case

    with raises(AttributeError, f"attribute '{typo}'", at="exec"):
        callback_problem(**{callback: body}).solve()


def test_python_if_on_a_symbol_raises_type_error_under_auto():
    def continuous(arg):
        _, _, v = arg.phase[0].state
        if v > 1:
            pass
        default_continuous(arg)

    with raises(TypeError, "yapss.math.where", at="if v > 1"):
        callback_problem("auto", continuous=continuous).solve()


@pytest.mark.parametrize("method", ["central-difference", "central-difference-full"])
def test_python_if_in_the_continuous_callback_raises_value_error_under_numeric_methods(method):
    def continuous(arg):
        _, _, v = arg.phase[0].state
        if v > 1:
            pass
        default_continuous(arg)

    with raises(ValueError, "ambiguous", at="if v > 1"):
        callback_problem(method, continuous=continuous).solve()


@pytest.mark.parametrize("name", ["objective", "continuous", "discrete"])
def test_missing_required_callback_is_reported_by_validate(name):
    ocp = callback_problem()
    setattr(ocp.functions, name, None)
    with raises(ValueError, f"'functions.{name}' function is required"):
        ocp.validate()


def test_user_method_requires_its_derivative_callbacks():
    ocp = brachistochrone.setup()
    ocp.functions.continuous_jacobian = None
    with raises(ValueError, "'functions.continuous_jacobian' function is required"):
        ocp.validate()


@pytest.mark.parametrize("value", [3, lambda a, b: None], ids=["not callable", "two parameters"])
def test_callback_that_is_not_callable_with_one_argument_raises_at_the_assignment(value):
    ocp = callback_problem()
    with raises(TypeError, "callable object with one argument", at="functions.objective ="):
        ocp.functions.objective = value


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
@pytest.mark.parametrize("output", ["arg.phase[0].path[0]", "arg.discrete[0]", "arg.objective"])
@pytest.mark.parametrize("method", METHODS)
def test_an_output_never_assigned_warns_at_the_callbacks_def_line(method, output):
    def continuous(arg):
        _, _, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = v * ym.cos(u), v * ym.sin(u), G0 * ym.sin(u)
        arg.phase[0].integrand[:] = (u**2,)
        if output != "arg.phase[0].path[0]":
            arg.phase[0].path[:] = (v,)

    def discrete(arg):
        if output != "arg.discrete[0]":
            arg.discrete[:] = (arg.phase[0].final_state[1],)

    def objective(arg):
        if output != "arg.objective":
            arg.objective = arg.phase[0].final_time

    callbacks = {"continuous": continuous, "discrete": discrete, "objective": objective}
    ocp = callback_problem(method, **callbacks)
    ocp.ipopt_options.max_iter = 0
    with pytest.warns(UnsetOutputWarning) as record:
        ocp.solve()
    unset = [w for w in record if issubclass(w.category, UnsetOutputWarning)]
    assert len(unset) == 1, [str(w.message) for w in unset]
    (warning,) = unset
    assert f"never assigned {output}" in str(warning.message)
    owner = {"arg.phase[0].path[0]": continuous, "arg.discrete[0]": discrete}.get(output, objective)
    assert warning.filename == owner.__code__.co_filename
    assert warning.lineno == owner.__code__.co_firstlineno


@pytest.mark.parametrize("method", METHODS)
def test_a_row_assigned_zero_or_updated_in_place_does_not_warn(method):
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].path[0] = 0.0
        arg.phase[0].path[0] += arg.phase[0].state[2]

    ocp = callback_problem(method, continuous=continuous)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnsetOutputWarning)
        solve_objective(ocp)


NOT_POINTWISE = {
    "t - t[0]": lambda t, x: x + (t - t[0]),
    "len": lambda t, x: x * len(t),
    "mean": lambda t, x: x + np.mean(x),
    "cumsum": lambda t, x: np.cumsum(x),
}


@pytest.mark.parametrize("violation", NOT_POINTWISE.values(), ids=NOT_POINTWISE.keys())
@pytest.mark.parametrize("method", METHODS)
def test_a_continuous_callback_that_is_not_pointwise_raises_stating_the_rule(method, violation):
    def continuous(arg):
        default_continuous(arg)
        (u,) = arg.phase[0].control
        _, _, v = arg.phase[0].state  # varies along the guess, as the check needs
        arg.phase[0].integrand[0] = u**2 + violation(arg.phase[0].time, v)

    with raises(ValueError, "not pointwise", "phase 0 integrand[0]", "continuous", "depend only"):
        callback_problem(method, continuous=continuous).solve()


@pytest.mark.parametrize("method", METHODS)
def test_rounding_noise_below_the_tolerance_is_pointwise(method):
    def continuous(arg):
        default_continuous(arg)
        (u,) = arg.phase[0].control
        noise = 1 + 1e-12 * len(arg.phase[0].time)  # order- and length-dependent, tiny
        arg.phase[0].integrand[0] = u**2 * noise

    assert solve_objective(callback_problem(method, continuous=continuous)) > 0


def test_a_phase_with_two_points_is_still_checked_for_pointwise():
    """Two evaluation points leave one comparison point, so only a violation visible there."""

    def continuous(arg):
        default_continuous(arg)
        (u,) = arg.phase[0].control
        _, _, v = arg.phase[0].state
        arg.phase[0].integrand[0] = u**2 + v[-1] - v

    ocp = callback_problem("central-difference-full", continuous=continuous)
    ocp.mesh.phase[0].collocation_points = (2,)
    ocp.mesh.phase[0].fraction = (1.0,)
    with raises(ValueError, "not pointwise"):
        ocp.solve()


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize("method", METHODS)
def test_functions_not_finite_at_the_initial_guess_raise_naming_the_output(method):
    def continuous(arg):
        default_continuous(arg)
        _, _, v = arg.phase[0].state
        arg.phase[0].path[0] = ym.sqrt(v - 1.0)

    with raises(ValueError, "not finite at the initial guess", "phase 0 path[0]"):
        callback_problem(method, continuous=continuous).solve()


# ------------------------------------------------------------------- user derivative keys


def user_key_problem(callback, key):
    """A one-phase problem with a parameter, a discrete constraint, and every user
    derivative callback, whose `callback` also sets the derivative entry `key`."""
    ocp = Problem(name="keys", nx=[2], nu=[1], ns=1, nd=1)

    def objective(arg):
        arg.objective = arg.phase[0].final_time + arg.parameter[0] ** 2

    def objective_gradient(arg):
        arg.gradient[0, "tf", 0] = 1.0
        arg.gradient[0, "s", 0] = 2.0 * arg.parameter[0]

    def objective_hessian(arg):
        arg.hessian[(0, "s", 0), (0, "s", 0)] = 2.0

    def continuous(arg):
        x, _ = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = [x * u, u]

    def continuous_jacobian(arg):
        x, _ = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].jacobian[("f", 0), ("x", 0)] = u
        arg.phase[0].jacobian[("f", 0), ("u", 0)] = x
        arg.phase[0].jacobian[("f", 1), ("u", 0)] = 1.0

    def continuous_hessian(arg):
        arg.phase[0].hessian[("f", 0), ("x", 0), ("u", 0)] = 1.0

    def discrete(arg):
        arg.discrete[:] = [arg.phase[0].final_state[0]]

    def discrete_jacobian(arg):
        arg.jacobian[0, (0, "xf", 0)] = 1.0

    def discrete_hessian(_arg):
        pass

    callbacks = {
        "objective_gradient": (objective_gradient, lambda arg: arg.gradient),
        "objective_hessian": (objective_hessian, lambda arg: arg.hessian),
        "continuous_jacobian": (continuous_jacobian, lambda arg: arg.phase[0].jacobian),
        "continuous_hessian": (continuous_hessian, lambda arg: arg.phase[0].hessian),
        "discrete_jacobian": (discrete_jacobian, lambda arg: arg.jacobian),
        "discrete_hessian": (discrete_hessian, lambda arg: arg.hessian),
    }
    functions = ocp.functions
    functions.objective = objective
    functions.continuous = continuous
    functions.discrete = discrete
    for name, (function, _) in callbacks.items():
        setattr(functions, name, function)
    function, entries = callbacks[callback]

    def with_key(arg):
        function(arg)
        entries(arg)[key] = 1.0

    setattr(functions, callback, with_key)

    bounds = ocp.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower, bounds.final_time.upper = 1.0, 2.0
    bounds.state.lower[:], bounds.state.upper[:] = -5.0, 5.0
    bounds.control.lower[:], bounds.control.upper[:] = -5.0, 5.0
    ocp.bounds.parameter.lower[:], ocp.bounds.parameter.upper[:] = -1.0, 1.0
    ocp.bounds.discrete.lower[:], ocp.bounds.discrete.upper[:] = -5.0, 5.0
    ocp.guess.phase[0].time = [0.0, 1.0]
    ocp.guess.phase[0].state = [[1.0, 2.0], [0.0, 1.0]]
    ocp.guess.phase[0].control = [[1.0, 1.0]]
    ocp.guess.parameter = [0.5]
    ocp.derivatives.method = "user"
    ocp.derivatives.order = "second"
    return ocp


# (callback, key, exception, message fragments): one case per rule of E7b, plus the
# shape of every callback's key. Every message also names the key and the callback.
INVALID_KEYS = {
    # decision variable keys, through the objective gradient
    "misspelled name": ("objective_gradient", (0, "tF", 0), ValueError, "did you mean 'tf'?"),
    "control in a gradient": ("objective_gradient", (0, "u", 0), ValueError, "one of 'x0'"),
    "short key": ("objective_gradient", (0, "tf"), ValueError, "(phase, name, index)"),
    "key not a tuple": ("objective_gradient", "tf", TypeError, "(phase, name, index)"),
    "name not a string": ("objective_gradient", (0, 3, 0), TypeError, "not a string"),
    "phase out of range": ("objective_gradient", (1, "tf", 0), ValueError, "has 1 phase"),
    "state out of range": (
        "objective_gradient",
        (0, "xf", 2),
        ValueError,
        "the state index 2 is out of range: phase 0 has 2 states, so the index must be in range(2)",
    ),
    "negative index": ("objective_gradient", (0, "xf", -1), ValueError, "state index -1"),
    "float index": ("objective_gradient", (0, "xf", 0.0), TypeError, "not an integer"),
    "bool index": ("objective_gradient", (0, "xf", True), TypeError, "not an integer"),
    "time index": ("objective_gradient", (0, "t0", 1), ValueError, "must be 0"),
    "no integrals": ("objective_gradient", (0, "q", 0), ValueError, "has no integrals"),
    "parameter phase": ("objective_gradient", (1, "s", 0), ValueError, "(0, 's', 0)"),
    "parameter index": ("objective_gradient", (0, "s", 1), ValueError, "parameter index 1"),
    # continuous function and variable keys, through the continuous Jacobian
    "dynamics index": (
        "continuous_jacobian",
        (("f", 2), ("x", 0)),
        ValueError,
        "phase 0 dynamics has 2 elements",
    ),
    "no path": ("continuous_jacobian", (("h", 0), ("x", 0)), ValueError, "has no elements"),
    "integral as a variable": ("continuous_jacobian", (("f", 0), ("q", 0)), ValueError, "'q'"),
    "variable index": ("continuous_jacobian", (("f", 0), ("u", 1)), ValueError, "control index"),
    "flat key": (
        "continuous_jacobian",
        ("f", 0, "x", 0),
        ValueError,
        "(function key, variable key)",
    ),
    "variable key not a tuple": (
        "continuous_jacobian",
        (("f", 0), "x"),
        TypeError,
        "continuous variable key",
    ),
    # the shape of each other callback's key
    "objective Hessian": (
        "objective_hessian",
        ((0, "tf", 0),),
        ValueError,
        "(variable key, variable key)",
    ),
    "continuous Hessian": (
        "continuous_hessian",
        (("f", 0), ("x", 0), ("y", 0)),
        ValueError,
        "'y'",
    ),
    "discrete Jacobian": (
        "discrete_jacobian",
        (1, (0, "tf", 0)),
        ValueError,
        "discrete function index 1",
    ),
    "discrete Hessian": (
        "discrete_hessian",
        (0, (0, "tf", 0)),
        ValueError,
        "(discrete function index, variable key, variable key)",
    ),
}


@pytest.mark.parametrize(
    ("callback", "key", "exc", "fragment"),
    INVALID_KEYS.values(),
    ids=INVALID_KEYS.keys(),
)
def test_invalid_user_derivative_key_raises_naming_it(callback, key, exc, fragment):
    ocp = user_key_problem(callback, key)
    phase = " in phase 0" if callback.startswith("continuous") else ""
    with raises(exc, f"key {key!r} set by functions.{callback}{phase}:", fragment):
        ocp.solve()


def test_numpy_integer_indices_are_accepted():
    reference = solve_objective(brachistochrone.setup())
    ocp = brachistochrone.setup()
    gradient, jacobian = ocp.functions.objective_gradient, ocp.functions.continuous_jacobian

    def objective_gradient(arg):
        gradient(arg)
        arg.gradient[0, "tf", np.int64(0)] = arg.gradient.pop((0, "tf", 0))

    def continuous_jacobian(arg):
        jacobian(arg)
        entries = arg.phase[0].jacobian
        entries[("f", np.int64(0)), ("x", np.int32(2))] = entries.pop((("f", 0), ("x", 2)))

    ocp.functions.objective_gradient = objective_gradient
    ocp.functions.continuous_jacobian = continuous_jacobian
    assert solve_objective(ocp) == pytest.approx(reference, rel=1e-8)


# ----------------------------------------------------------------- not yet met: outputs


@pytest.mark.filterwarnings("ignore::yapss._private.setup_check.UnsetOutputWarning")
@not_yet("E1", "a callback that returns a value raises TypeError showing the assignment idiom")
@pytest.mark.parametrize("method", METHODS)
def test_returning_the_objective_raises(method):
    def objective(arg):
        return arg.phase[0].final_time

    with raises(TypeError, "arg.objective"):
        callback_problem(method, objective=objective).solve()


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
@pytest.mark.filterwarnings("ignore::yapss._private.setup_check.UnsetOutputWarning")
@not_yet("E1", "a continuous or discrete callback that returns a value raises TypeError")
@pytest.mark.parametrize("callback", ["continuous", "discrete"])
def test_returning_from_continuous_or_discrete_raises(callback):
    def continuous(arg):
        default_continuous(arg)
        return arg.phase[0].dynamics

    def discrete(arg):
        return (arg.phase[0].final_state[1],)

    with raises(TypeError):
        callback_problem(
            **{callback: {"continuous": continuous, "discrete": discrete}[callback]}
        ).solve()


@not_yet("W5", "a non-scalar objective raises naming arg.objective")
def test_non_scalar_objective_raises():
    def objective(arg):
        arg.objective = arg.phase[0].final_state

    with raises((TypeError, ValueError), "objective"):
        callback_problem("central-difference", objective=objective).solve()


# ------------------------------------------------------------------ not yet met: inputs


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
@not_yet("E2 part 1", "callback inputs are read-only; an in-place write raises")
@pytest.mark.parametrize("target", ["state", "parameter"])
def test_writing_an_input_in_place_raises(target):
    ocp = brachistochrone.setup()
    ocp.derivatives.method = "central-difference"
    continuous = ocp.functions.continuous

    def writer(arg):
        continuous(arg)
        if target == "state":
            arg.phase[0].state[2][:] = 0.0
        else:
            arg.parameter[:] = 0.0

    ocp.functions.continuous = writer
    with raises(ValueError, at="] = 0.0"):
        ocp.solve()


# NumPy < 2.3 warns before raising here; newer NumPy raises directly
@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")
@not_yet("W5", "a non-yapss.math function on a symbol raises naming yapss.math")
def test_math_module_function_on_a_symbol_raises_helpfully():
    def continuous(arg):
        default_continuous(arg)
        math.sin(arg.phase[0].state[2])

    with raises(TypeError, "yapss.math", at="math.sin"):
        callback_problem("auto", continuous=continuous).solve()


@not_yet("W5", "an exception raised in a callback carries a note naming the callback")
def test_exception_in_a_callback_has_a_context_note():
    def continuous(arg):
        raise ValueError("user error")

    with pytest.raises(ValueError) as info:
        callback_problem("central-difference", continuous=continuous).solve()
    assert any("continuous" in note for note in getattr(info.value, "__notes__", []))


@not_yet("B7", "a derivative callback the chosen method does not use warns, naming it")
def test_unused_derivative_callback_warns():
    ocp = callback_problem("auto")
    ocp.functions.continuous_jacobian = lambda arg: None
    with pytest.warns(Warning, match="continuous_jacobian"):
        ocp.solve()


@not_yet("W3", "the arity check accepts a callable whose extra parameters have defaults")
def test_callback_with_defaulted_extra_parameter_is_accepted():
    ocp = callback_problem()

    def objective(arg, scale=1.0):
        arg.objective = scale * arg.phase[0].final_time

    ocp.functions.objective = objective


# -------------------------------------------------------- not yet met: user derivatives


@not_yet("E7b", "a derivative key first set after the first call raises")
def test_key_set_only_on_later_calls_raises():
    ocp = brachistochrone.setup()
    ocp.ipopt_options.print_level = 0
    jacobian = ocp.functions.continuous_jacobian
    calls = {"n": 0}

    def continuous_jacobian(arg):
        jacobian(arg)
        calls["n"] += 1
        if calls["n"] > 1:
            arg.phase[0].jacobian[("f", 0), ("x", 0)] = 0.0

    ocp.functions.continuous_jacobian = continuous_jacobian
    with raises(ValueError, "('f', 0)"):
        ocp.solve()


@not_yet("W5", "a length-1 array entry in a user Hessian raises ValueError naming the key")
def test_length_one_array_hessian_entry_raises_naming_it():
    ocp = brachistochrone.setup()
    hessian = ocp.functions.continuous_hessian

    def continuous_hessian(arg):
        hessian(arg)
        arg.phase[0].hessian[("f", 2), ("u", 0), ("u", 0)] = np.array([-1.0])

    ocp.functions.continuous_hessian = continuous_hessian
    with raises(ValueError, "('f', 2)"):
        ocp.solve()


@pytest.mark.filterwarnings("ignore::yapss.MirroredHessianPairWarning")
@not_yet("0.3.0 scheduled", "setting both orders of one Hessian pair raises ValueError")
def test_mirrored_hessian_pair_raises():
    ocp = brachistochrone.setup()
    hessian = ocp.functions.continuous_hessian

    def continuous_hessian(arg):
        hessian(arg)
        arg.phase[0].hessian[("f", 0), ("u", 0), ("x", 2)] = 0.0

    ocp.functions.continuous_hessian = continuous_hessian
    with raises(ValueError):
        ocp.solve()


# ------------------------------------------------------------------ not yet met: targets


# ------------------------------------------------- callback arguments keep outputs apart
#
# Each callback's argument exposes the outputs that callback produces. Writing another
# callback's output raises `AttributeError` at the line. The objective and discrete
# families meet this already. The continuous family does not: one object carrying every
# output goes to `continuous`, `continuous_jacobian`, and `continuous_hessian` (E7c). A
# derivative entry written from `continuous` is silently ignored; worse, a dynamics row
# written from `continuous_jacobian` overwrites the constraint values the shared evaluator
# has cached on that object, and the solve silently goes wrong (measured 2026-09-14: status
# -1, objective 0.312455 against 0.312480). E7c's target (option A) removes
# the foreign outputs entirely; its accepted fallback (option C) keeps them visible but
# refuses writes. The write clause is required under either; absence is the target only.


def _user_problem(example):
    ocp = example.setup()
    ocp.derivatives.method = "user"
    ocp.ipopt_options.print_level = 0
    return ocp


def _run_statement_in(ocp, callback, statement):
    """Wrap `callback` so it executes `statement` before doing its own work, then solve."""
    original = getattr(ocp.functions, callback)

    def wrapped(arg):
        exec(statement, {"arg": arg})  # noqa: S102 -- one parametrized statement per case
        original(arg)

    setattr(ocp.functions, callback, wrapped)
    ocp.solve()


def _visible_outputs(ocp, callback, names, where):
    """Return which of `names` are attributes of `arg` (or `arg.phase[0]`) in `callback`."""
    original = getattr(ocp.functions, callback)
    seen = {}

    def wrapped(arg):
        owner = arg.phase[0] if where == "phase" else arg
        seen.update({name: hasattr(owner, name) for name in names})
        original(arg)

    setattr(ocp.functions, callback, wrapped)
    ocp.solve()
    return seen


OBJECTIVE_AND_DISCRETE_OUTPUTS = {
    "objective": ("objective", ["gradient", "hessian", "discrete", "jacobian"]),
    "objective_gradient": ("gradient", ["objective", "hessian"]),
    "objective_hessian": ("hessian", ["objective", "gradient"]),
    "discrete": ("discrete", ["objective", "jacobian", "hessian"]),
    "discrete_jacobian": ("jacobian", ["discrete", "hessian", "objective"]),
    "discrete_hessian": ("hessian", ["discrete", "jacobian", "objective"]),
}
CONTINUOUS_OUTPUTS = {
    "continuous": ["dynamics", "path", "integrand"],
    "continuous_jacobian": ["jacobian"],
    "continuous_hessian": ["hessian"],
}
CONTINUOUS_ALL = ["dynamics", "path", "integrand", "jacobian", "hessian"]
CONTINUOUS_FOREIGN_WRITES = {
    "continuous": {
        "jacobian": 'arg.phase[0].jacobian[("f", 0), ("x", 2)] = 0.0',
        "hessian": 'arg.phase[0].hessian[("f", 0), ("x", 2), ("u", 0)] = 0.0',
    },
    "continuous_jacobian": {
        "dynamics": "arg.phase[0].dynamics[0] = 0.0",
        "hessian": 'arg.phase[0].hessian[("f", 0), ("x", 2), ("u", 0)] = 0.0',
    },
    "continuous_hessian": {
        "dynamics": "arg.phase[0].dynamics[0] = 0.0",
        "jacobian": 'arg.phase[0].jacobian[("f", 0), ("x", 2)] = 0.0',
    },
}


@pytest.mark.parametrize("callback", OBJECTIVE_AND_DISCRETE_OUTPUTS)
def test_objective_and_discrete_arguments_expose_only_their_own_output(callback):
    own, foreign = OBJECTIVE_AND_DISCRETE_OUTPUTS[callback]
    seen = _visible_outputs(
        _user_problem(goddard_problem_3_phase), callback, [own, *foreign], "arg"
    )
    assert seen == {own: True, **{name: False for name in foreign}}


@pytest.mark.parametrize(
    ("callback", "foreign"),
    [
        (callback, name)
        for callback, (_, names) in OBJECTIVE_AND_DISCRETE_OUTPUTS.items()
        for name in names
    ],
)
def test_writing_another_callbacks_output_raises_in_objective_and_discrete_callbacks(
    callback, foreign
):
    value = "0.0" if foreign == "objective" else "(0.0,)" if foreign == "discrete" else "{}"
    with raises(AttributeError, foreign, at="exec"):
        _run_statement_in(
            _user_problem(goddard_problem_3_phase), callback, f"arg.{foreign} = {value}"
        )


@pytest.mark.parametrize("callback", CONTINUOUS_OUTPUTS)
def test_continuous_family_arguments_expose_their_own_outputs(callback):
    own = CONTINUOUS_OUTPUTS[callback]
    seen = _visible_outputs(_user_problem(brachistochrone), callback, own, "phase")
    assert all(seen.values()), seen


@not_yet(
    "E7c", "writing another continuous-family callback's output raises AttributeError at the line"
)
@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
@pytest.mark.parametrize(
    ("callback", "foreign"),
    [(callback, name) for callback, writes in CONTINUOUS_FOREIGN_WRITES.items() for name in writes],
)
def test_writing_another_callbacks_output_raises_in_continuous_callbacks(callback, foreign):
    statement = CONTINUOUS_FOREIGN_WRITES[callback][foreign]
    with raises(AttributeError, foreign, at="exec"):
        _run_statement_in(_user_problem(brachistochrone), callback, statement)


@not_yet("E7c target (option A)", "a continuous-family argument has no other callback's outputs")
@pytest.mark.parametrize("callback", CONTINUOUS_OUTPUTS)
def test_continuous_family_arguments_have_no_foreign_outputs(callback):
    foreign = [name for name in CONTINUOUS_ALL if name not in CONTINUOUS_OUTPUTS[callback]]
    seen = _visible_outputs(_user_problem(brachistochrone), callback, foreign, "phase")
    assert not any(seen.values()), seen


@not_yet("F7 target", "every callback argument reports why it is called in arg.evaluation")
def test_arg_evaluation_names_the_kind_of_call():
    kinds = set()

    def continuous(arg):
        kinds.add(arg.evaluation)
        default_continuous(arg)

    callback_problem("central-difference", continuous=continuous).solve()
    assert "value" in kinds
    assert "probe" in kinds
