"""Contract: the objective, continuous, and discrete callbacks, and user derivatives.

What a user may do
    - Assign continuous outputs as a tuple or list of rows, row by row, or by slices; each
      row an expression over the phase's points or a constant scalar.
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
      `ValueError` at the assignment.
    - A misspelled output, or an assignment to an input: `AttributeError` at the line.
    - A Python `if` on a problem variable: `TypeError` under `"auto"` (pointing to
      `yapss.math.where`), `ValueError` in the continuous callback under the numeric
      methods.
    - A required callback left unset: `ValueError` from `validate()` naming it. A callback
      that is not callable with one argument: `TypeError` at the assignment.
    - Functions not finite at the initial guess: `ValueError` naming the output.
    - Under `"user"`, a derivative key with an invalid name: `ValueError` naming the term.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import yapss.math as ym
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
            arg.phase[p].path[0] = v
            arg.phase[p].integrand[0] = u**2

    return continuous


# ---------------------------------------------------------------- what a user may do


@pytest.mark.parametrize("method", METHODS)
def test_every_method_solves_the_same_problem(method, baseline):
    assert solve_objective(callback_problem(method)) == pytest.approx(baseline, rel=1e-9)


@pytest.mark.parametrize("form", ["tuple", "list", "row by row", "slices", "numpy ufuncs"])
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


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
@pytest.mark.parametrize("method", METHODS)
def test_functions_not_finite_at_the_initial_guess_raise_naming_the_output(method):
    def continuous(arg):
        default_continuous(arg)
        _, _, v = arg.phase[0].state
        arg.phase[0].path[0] = ym.sqrt(v - 1.0)

    with raises(ValueError, "not finite at the initial guess", "phase 0 path[0]"):
        callback_problem(method, continuous=continuous).solve()


def test_user_derivative_key_with_an_invalid_name_is_reported_naming_the_term():
    ocp = brachistochrone.setup()
    jacobian = ocp.functions.continuous_jacobian

    def continuous_jacobian(arg):
        jacobian(arg)
        arg.phase[0].jacobian[("f", 0), ("q", 0)] = 1.0

    ocp.functions.continuous_jacobian = continuous_jacobian
    with raises(ValueError, "(('f', 0), ('q', 0))", "phase 0"):
        ocp.solve()


# ----------------------------------------------------------------- not yet met: outputs


@not_yet("E1", "a callback that returns a value raises TypeError showing the assignment idiom")
@pytest.mark.parametrize("method", METHODS)
def test_returning_the_objective_raises(method):
    def objective(arg):
        return arg.phase[0].final_time

    with raises(TypeError, "arg.objective"):
        callback_problem(method, objective=objective).solve()


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
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


@pytest.mark.filterwarnings("ignore::yapss.IpoptConvergenceWarning")
@not_yet("E1", "an output never written warns (UnsetOutputWarning), naming callback and output")
@pytest.mark.parametrize("output", ["path", "discrete"])
def test_output_never_written_warns(output):
    def continuous(arg):
        _, _, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = v * ym.cos(u), v * ym.sin(u), G0 * ym.sin(u)
        arg.phase[0].integrand[:] = (u**2,)
        if output != "path":
            arg.phase[0].path[:] = (v,)

    def discrete(arg):
        if output != "discrete":
            arg.discrete[:] = (arg.phase[0].final_state[1],)

    with pytest.warns(Warning, match=output):
        callback_problem(continuous=continuous, discrete=discrete).solve()


@not_yet("E1", "a slice write to an output with no rows raises, naming the count (e.g. nh)")
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
@not_yet("E1", "one row written into several rows by broadcasting raises")
def test_one_row_broadcast_into_several_rows_raises():
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].dynamics[:] = arg.phase[0].state[2]

    with raises(ValueError, "dynamics", at="dynamics[:] ="):
        callback_problem("central-difference", continuous=continuous).solve()


@not_yet("E1", "a row that is a length-1 array (not a scalar or the points' shape) raises")
def test_length_one_array_row_raises():
    def continuous(arg):
        default_continuous(arg)
        arg.phase[0].integrand[0] = np.atleast_1d(arg.phase[0].control[0])[:1]

    with raises(ValueError, "integrand", at="integrand[0] ="):
        callback_problem("central-difference", continuous=continuous).solve()


@not_yet("W5", "shape errors name the output and phase")
def test_row_count_message_names_the_output_and_phase():
    def continuous(arg):
        _, _, v = arg.phase[0].state
        arg.phase[0].dynamics[:] = (v, v)

    with raises(ValueError, "phase 0", "dynamics"):
        callback_problem(continuous=continuous).solve()


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


@not_yet("E15", "a callback that is not pointwise in time raises, stating the rule")
def test_non_pointwise_continuous_callback_raises():
    def continuous(arg):
        default_continuous(arg)
        t = arg.phase[0].time
        arg.phase[0].integrand[0] = arg.phase[0].control[0] ** 2 + (t - t[0])

    with pytest.raises(Exception, match="pointwise"):
        callback_problem("central-difference", continuous=continuous).solve()


@not_yet("F10b", "np.where under central differences raises, pointing to yapss.math.where")
def test_numpy_where_under_central_differences_raises():
    def continuous(arg):
        default_continuous(arg)
        (u,) = arg.phase[0].control
        arg.phase[0].path[0] = np.where(u > 0, arg.phase[0].state[2], 0.0)

    with raises(TypeError, "yapss.math.where", at="np.where"):
        callback_problem("central-difference", continuous=continuous).solve()


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


def _with_extra_jacobian_entry(key, value=1.0):
    ocp = brachistochrone.setup()
    jacobian = ocp.functions.continuous_jacobian

    def continuous_jacobian(arg):
        jacobian(arg)
        arg.phase[0].jacobian[key] = value

    ocp.functions.continuous_jacobian = continuous_jacobian
    return ocp


@not_yet("E7b", "a derivative key out of range, or of the wrong shape, raises ValueError naming it")
@pytest.mark.parametrize(
    "key",
    [(("f", 5), ("x", 0)), (("f", 0), ("x", 7)), ("f", 0, "x", 2)],
    ids=["function index", "variable index", "flat key"],
)
def test_invalid_user_jacobian_key_raises_naming_it(key):
    with raises(ValueError, "phase 0"):
        _with_extra_jacobian_entry(key).solve()


@not_yet("E7b", "a misspelled objective gradient key raises ValueError naming it")
def test_misspelled_gradient_key_raises_naming_it():
    ocp = brachistochrone.setup()
    gradient = ocp.functions.objective_gradient

    def objective_gradient(arg):
        gradient(arg)
        arg.gradient[0, "tF", 0] = 1.0

    ocp.functions.objective_gradient = objective_gradient
    with raises(ValueError, "'tF'"):
        ocp.solve()


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
