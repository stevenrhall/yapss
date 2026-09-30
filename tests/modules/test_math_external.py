"""`yapss.math.external`: a function ``"auto"`` cannot trace, used in a callback.

The oracle throughout is the same model written so that ``"auto"`` can trace it. A problem
whose callbacks call wrapped functions must assemble the same NLP as the traced one: the same
structures exactly, and the same values to the accuracy of the wrapped functions' own
derivatives, which is rounding when they are supplied and the differencing error when not.

The synthetic problem reaches every place a wrapped function can be used: the continuous
callback, with one wrapped function feeding another and with a function of two arguments; the
objective; and a discrete constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

import yapss
import yapss.examples.brachistochrone as brachistochrone
from yapss._api.compile import to_transcription_spec
from yapss._api.spec import snapshot
from yapss._backend.auto import make_auto_functions
from yapss._backend.difference_steps import EPS, difference_steps
from yapss._backend.guess import make_initial_guess_nlp
from yapss._backend.mesh import Mesh
from yapss._backend.nlp import NLP
from yapss.math import cos, external, sin
from yapss.math._external import tracing
from yapss.math.wrapper import SXW, SXArray, sx_array

# ------------------------------------------------------------------------------------------
# the synthetic problem
# ------------------------------------------------------------------------------------------


class State(yapss.State):
    x = yapss.scalar()
    v = yapss.scalar()


class Control(yapss.Control):
    u = yapss.scalar()


class Path(yapss.Path):
    g = yapss.scalar()


class Integral(yapss.Integral):
    q = yapss.scalar()


class Parameter(yapss.Parameter):
    p = yapss.scalar()


class Discrete(yapss.Discrete):
    d = yapss.scalar()


class Phase(yapss.Phase):
    state: State
    control: Control
    path: Path
    integral: Integral


class Phases(yapss.Phases):
    phase: Phase


class Problem(yapss.Problem):
    phases: Phases
    parameter: Parameter
    discrete: Discrete


def make_problem(sine, cosine, blend):
    """Return the problem, its model built from the three functions given."""
    problem = Problem("external")
    ph = problem.phases.phase

    @ph.register.continuous
    def continuous(arg, out):
        x, v, u, p = arg.state.x, arg.state.v, arg.control.u, arg.parameter.p
        speed = 2 + sine(x)
        mach = v / speed  # a wrapped function's result, which the next two are given
        out.dynamics.x = v * cosine(u)
        out.dynamics.v = blend(mach, x) * u + p
        out.path.g = speed * u
        out.integrand.q = blend(mach, u)

    @problem.register.objective
    def objective(arg):
        return arg[ph].integral.q + sine(arg[ph].final_state.x)

    @problem.register.discrete
    def discrete(arg, out):
        out.discrete.d = blend(arg.parameter.p, arg[ph].final_state.v)

    ph.time.initial = (0.0, 0.0)
    ph.time.final = (1.0, 2.0)
    ph.time.guess = (0.0, 1.5)
    ph.state.x.guess = (0.3, 0.9)
    ph.state.v.guess = (0.5, 1.1)
    ph.control.u.guess = (0.4, 0.7)
    problem.parameter.p.guess = 0.6
    ph.mesh = yapss.Mesh.uniform(segments=2, points=4)
    problem.derivatives.method = "auto"
    problem.derivatives.order = "second"
    return problem


def traced_blend(a, b):
    return a**2 * cos(b) + a * b


def blend(a, b):
    return a**2 * np.cos(b) + a * b


def blend_jacobian(a, b):
    return [2 * a * np.cos(b) + b, -(a**2) * np.sin(b) + a]


def blend_hessian(a, b):
    mixed = -2 * a * np.sin(b) + 1
    return [[2 * np.cos(b), mixed], [mixed, -(a**2) * np.cos(b)]]


def negative_sin(a):
    return -np.sin(a)


def negative_cos(a):
    return -np.cos(a)


def supplied():
    return (
        external(np.sin, jacobian=np.cos, hessian=negative_sin, vectorized=True),
        external(np.cos, jacobian=negative_sin, hessian=negative_cos, vectorized=True),
        external(blend, jacobian=blend_jacobian, hessian=blend_hessian, vectorized=True),
    )


def differenced():
    return (
        external(np.sin, vectorized=True),
        external(np.cos, vectorized=True),
        external(blend, vectorized=True),
    )


def one_point_at_a_time():
    return (
        external(np.sin, jacobian=np.cos, hessian=negative_sin),
        external(np.cos),
        external(blend, jacobian=blend_jacobian, hessian=blend_hessian),
    )


def gradient_only():
    return (
        external(np.sin, jacobian=np.cos, vectorized=True),
        external(np.cos, jacobian=negative_sin, vectorized=True),
        external(blend, jacobian=blend_jacobian, vectorized=True),
    )


def assemble(problem):
    """Return every quantity the solver is given, a little way off the initial guess."""
    spec = to_transcription_spec(snapshot(problem))
    mesh = Mesh(spec.phases)
    mesh.set_matrices(spec.spectral_method)
    z0 = make_initial_guess_nlp(spec, mesh)
    nlp = NLP(spec, make_auto_functions(spec), mesh)
    z = z0 + 0.01 * np.sin(1.0 + np.arange(len(z0)))
    lam = 0.5 + 0.3 * np.cos(1.0 + np.arange(len(nlp.constraints(z))))
    return {
        "objective": np.asarray(nlp.objective(z)),
        "gradient": np.asarray(nlp.gradient(z)),
        "constraints": np.asarray(nlp.constraints(z)),
        "jacobian": np.asarray(nlp.jacobian(z)),
        "hessian": np.asarray(nlp.hessian(z, lam, np.float64(1.3))),
        "jacobian_structure": np.asarray(nlp.jacobianstructure()),
        "hessian_structure": np.asarray(nlp.hessianstructure()),
    }


@pytest.fixture(scope="module")
def traced():
    return assemble(make_problem(sin, cos, traced_blend))


# first derivatives and second derivatives of the assembled NLP, relative to the largest entry
CASES = {
    "supplied": (supplied, 1e-13, 1e-13),
    "one point at a time": (one_point_at_a_time, 1e-8, 1e-5),
    "gradient only": (gradient_only, 1e-13, 1e-8),
    "differenced": (differenced, 1e-8, 1e-5),
}


@pytest.mark.parametrize("case", CASES)
def test_the_assembled_nlp_is_the_traced_one(case, traced):
    functions, first, second = CASES[case]
    wrapped = assemble(make_problem(*functions()))

    for name in ("jacobian_structure", "hessian_structure"):
        np.testing.assert_array_equal(wrapped[name], traced[name], err_msg=name)
    for name, tolerance in (
        ("objective", 1e-13),
        ("constraints", 1e-13),
        ("gradient", first),
        ("jacobian", first),
        ("hessian", second),
    ):
        scale = np.abs(traced[name]).max()
        np.testing.assert_allclose(
            wrapped[name], traced[name], rtol=0, atol=tolerance * scale, err_msg=name
        )


def test_every_use_was_reached(monkeypatch):
    """The problem really does hold wrapped calls in all three callbacks."""
    import yapss._backend.auto as auto

    counts = []
    original = auto.stand_in_inputs

    def recording(variables, uses):
        counts.append(len(uses))
        return original(variables, uses)

    monkeypatch.setattr(auto, "stand_in_inputs", recording)
    make_auto_functions(to_transcription_spec(snapshot(make_problem(*supplied()))))
    # objective, discrete, then the one phase: sine; blend; sine, cosine and blend twice
    assert counts == [1, 1, 4]


# ------------------------------------------------------------------------------------------
# a solve
# ------------------------------------------------------------------------------------------


def solve_brachistochrone(monkeypatch, method, **replacements):
    for name, function in replacements.items():
        monkeypatch.setattr(brachistochrone, name, function)
    problem = brachistochrone.setup()
    problem.derivatives.method = method
    problem.ipopt_options.print_level = 0
    return problem.solve()


@pytest.mark.parametrize("vectorized", [True, False])
def test_a_solve_reaches_the_traced_answer(monkeypatch, vectorized):
    traced_solution = solve_brachistochrone(monkeypatch, "auto")
    wrapped = solve_brachistochrone(
        monkeypatch,
        "auto",
        sin=external(np.sin, vectorized=vectorized),
        cos=external(np.cos, vectorized=vectorized),
    )
    assert wrapped.converged
    assert wrapped.objective == pytest.approx(traced_solution.objective, rel=1e-9)


def test_under_central_differences_the_wrapper_is_the_function(monkeypatch):
    plain = solve_brachistochrone(monkeypatch, "central-difference")
    wrapped = solve_brachistochrone(
        monkeypatch,
        "central-difference",
        sin=external(np.sin, vectorized=True),
        cos=external(np.cos, vectorized=True),
    )
    assert wrapped.objective == pytest.approx(plain.objective, rel=1e-12)


# ------------------------------------------------------------------------------------------
# on numbers
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("vectorized", [True, False])
def test_on_numbers_it_is_the_function_broadcast(vectorized):
    wrapped = external(blend, vectorized=vectorized)
    a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_allclose(wrapped(a, 0.7), blend(a, 0.7), rtol=1e-15)
    assert wrapped(a, 0.7).shape == (2, 3)
    assert float(wrapped(0.2, 0.7)) == pytest.approx(blend(0.2, 0.7), rel=1e-15)
    assert np.ndim(wrapped(0.2, 0.7)) == 0


def test_a_function_of_single_floats_is_given_single_floats():
    import math

    wrapped = external(lambda h: 1.225 * math.exp(-h / 8500.0))
    np.testing.assert_allclose(
        wrapped(np.array([0.0, 1000.0])), 1.225 * np.exp(-np.array([0.0, 1000.0]) / 8500.0)
    )


def test_it_is_a_decorator_with_or_without_options():
    @external
    def plain(a):
        """The docstring."""
        return 2 * a

    @external(scale=3.0, vectorized=True, name="doubled")
    def optioned(a):
        return 2 * a

    assert plain(1.5) == 3.0
    assert plain.__doc__ == "The docstring."
    assert optioned(1.5) == 3.0
    assert repr(plain) == "<external plain>"
    assert repr(optioned) == "<external doubled>"


# ------------------------------------------------------------------------------------------
# the difference steps
# ------------------------------------------------------------------------------------------


def test_the_steps_follow_the_central_difference_rule():
    from yapss._backend import central_difference

    assert difference_steps() == (central_difference.DELTA1, central_difference.DELTA2)
    assert difference_steps(EPS) == ((3 * EPS) ** (1 / 3), (3 * EPS) ** (1 / 4))


def steps_used(wrapped, a):
    """Return the distinct distances from `a` at which `wrapped` evaluates its function."""
    seen = []
    inner = wrapped._function

    def recording(x):
        seen.append(np.array(x, dtype=float))
        return inner(x)

    wrapped._function = recording
    wrapped.derivatives(np.array([[a]]), 2)
    return sorted({round(abs(float(x[0]) - a), 12) for x in seen} - {0.0})


def test_the_steps_are_scaled_and_differ_for_first_and_second_differences():
    first, second = difference_steps()
    wrapped = external(np.sin, scale=100.0, vectorized=True)
    assert steps_used(wrapped, 1.0) == pytest.approx([100 * first, 100 * second])


def test_a_single_precision_function_is_differenced_as_one():
    single = external(lambda a: np.sin(a).astype(np.float32), vectorized=True)
    first, second = difference_steps(float(np.finfo(np.float32).eps) / 2)
    assert steps_used(single, 1.0) == pytest.approx([first, second])


def test_a_stated_precision_wins():
    stated = external(np.sin, vectorized=True, eps=1e-8)
    first, second = difference_steps(1e-8)
    assert steps_used(stated, 1.0) == pytest.approx([first, second])


def test_a_supplied_gradient_is_differenced_for_the_hessian():
    wrapped = external(blend, jacobian=blend_jacobian, vectorized=True)
    a = np.array([[0.3, 0.8], [0.5, 0.2]])
    _, gradient, hessian = wrapped.derivatives(a, 2)
    np.testing.assert_allclose(gradient, np.array(blend_jacobian(*a)), rtol=1e-15)
    exact = np.array(blend_hessian(*a))
    np.testing.assert_allclose(hessian, [exact[0, 0], exact[1, 1], exact[0, 1]], rtol=1e-8)


# ------------------------------------------------------------------------------------------
# refusals
# ------------------------------------------------------------------------------------------


def test_a_vectorized_function_that_couples_points_is_refused():
    coupled = external(lambda a: a - a.mean(), vectorized=True, name="coupled")
    with pytest.raises(ValueError, match=r"coupled\(\) is wrapped with vectorized=True.*element 0"):
        coupled(np.array([1.0, 2.0, 4.0]))


def test_a_vectorized_function_that_reduces_is_refused():
    total = external(np.sum, vectorized=True, name="total")
    with pytest.raises(ValueError, match=r"total\(\) was given arrays of 3 values"):
        total(np.array([1.0, 2.0, 4.0]))


def test_a_function_that_returns_several_values_is_refused():
    pair = external(lambda a: (a, a), name="pair")
    with pytest.raises(ValueError, match=r"pair\(\) returned a value of shape \(2,\)"):
        pair(1.0)


def test_a_symbol_outside_a_callback_is_refused():
    wrapped = external(np.sin, name="lookup")
    with pytest.raises(RuntimeError, match=r"lookup\(\) was called with a symbolic value outside"):
        wrapped(SXW(1.0))


def test_no_arguments_are_refused():
    with pytest.raises(TypeError, match="takes at least one argument"):
        external(np.sin)()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"eps": 0.0}, ValueError, "eps must be positive"),
        ({"eps": -1e-8}, ValueError, "eps must be positive"),
        ({"scale": 0.0}, ValueError, "scale must be positive"),
        ({"scale": [1.0, -2.0]}, ValueError, "scale must be positive"),
        ({"jacobian": 3.0}, TypeError, "jacobian must be callable"),
        ({"hessian": "no"}, TypeError, "hessian must be callable"),
    ],
)
def test_bad_options_are_refused_at_the_wrapping(options, error, message):
    with pytest.raises(error, match=message):
        external(np.sin, **options)


def test_what_is_wrapped_must_be_callable():
    with pytest.raises(TypeError, match="function must be callable"):
        external(3.0)


# ------------------------------------------------------------------------------------------
# in a trace
# ------------------------------------------------------------------------------------------


def test_a_symbolic_array_gets_one_stand_in_for_each_element():
    import casadi as ca

    wrapped = external(np.sin, vectorized=True)
    symbols = sx_array([SXW(ca.SX.sym(f"x{i}")) for i in range(3)])
    with tracing() as uses:
        result = wrapped(symbols)
        scalar = wrapped(symbols[0])
    assert isinstance(result, SXArray)
    assert result.shape == (3,)
    assert isinstance(scalar, SXW)
    assert len(uses) == 4


def test_traces_nest_without_mixing_their_uses():
    wrapped = external(np.sin)
    with tracing() as outer:
        wrapped(SXW(1.0))
        with tracing() as inner:
            wrapped(SXW(2.0))
            wrapped(SXW(3.0))
        wrapped(SXW(4.0))
    assert (len(outer), len(inner)) == (2, 2)
