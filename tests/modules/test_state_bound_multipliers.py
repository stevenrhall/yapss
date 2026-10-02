"""The multipliers of a state's bounds and of an integral's bounds.

A state's general bound has a density, ``state_multiplier``, on the collocation points. Each
end of a phase has a multiplier per state, ``initial_state_multiplier`` and
``final_state_multiplier``: the multiplier of the bound the NLP holds the state by there,
which is the tighter of the general bound and the end's own. It is an atom, one number at a
point, whichever of the two bounds is the active one.

Where an end is a collocation point (both under LGL, the initial one under LGR) the density
has that same number too, divided by the quadrature weight, unless the end's own bound is
strictly the tighter on the active side, in which case the density is zero there. The sign of
the multiplier picks the side, and no threshold is applied. So at a collocated end the
density's value and the atom are one number, never to be added; at an end that is not
collocated the density has no value and the atom stands alone.

What the multipliers mean is checked the way every multiplier here is: against the objective's
sensitivity to the bound, from a central difference of two solves. The problem is the
brachistochrone with a floor, ``y <= b`` (``y`` positive down), which the bead reaches and
rides to the end: a state bound active over an interval that includes the final point.
"""

import numpy as np
import pytest

import yapss
from yapss._private.mesh import Mesh
from yapss._private.solution import _general_share
from yapss._private.structure import get_nlp_dv_structure
from yapss.math import cos, sin

METHODS = ("lgl", "lgr", "lg")
DB = 1e-5
X, Y, V = 0, 1, 2


def floor(method, b=0.45, xf=1.0, effort=None, final_y=None):
    """The brachistochrone with a floor at `b`, ending at `xf`, its effort bounded above.

    `final_y`, if given, is an upper bound on ``y`` at the final point, beside the floor.
    """
    problem = yapss.Problem(name="floor", nx=[3], nu=[1], nq=[1])

    def objective(arg):
        arg.objective = arg.phase[0].final_time

    def continuous(arg):
        _, _, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics = [v * cos(u), v * sin(u), 32.174 * sin(u)]
        arg.phase[0].integrand = [u**2]

    problem.functions.objective = objective
    problem.functions.continuous = continuous
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.initial_state.lower = bounds.initial_state.upper = [0.0, 0.0, 0.0]
    bounds.final_state.lower[X] = bounds.final_state.upper[X] = xf
    bounds.state.upper[Y] = b
    if final_y is not None:
        bounds.final_state.upper[Y] = final_y
    bounds.control.lower, bounds.control.upper = [-2.0], [2.0]
    bounds.integral.upper = [1e3 if effort is None else effort]
    guess = problem.guess.phase[0]
    guess.time = [0.0, 1.0]
    guess.state = [[0.0, 1.0], [0.0, 0.3], [0.0, 5.0]]
    guess.control = [[0.0, 0.0]]
    problem.mesh.phase[0].collocation_points = 10 * (10,)
    problem.mesh.phase[0].fraction = 10 * (0.1,)
    problem.spectral_method = method
    problem.ipopt_options.print_level = 0
    problem.ipopt_options.tol = 1e-12
    return problem


def sensitivity(method, **bound):
    """Return dJ/d(bound) by a central difference, for the one keyword given."""
    ((name, value),) = bound.items()
    up = floor(method, **{name: value + DB}).solve().objective
    down = floor(method, **{name: value - DB}).solve().objective
    return (up - down) / (2 * DB)


def integrate(solution, density):
    """Integrate a density on the collocation points with the method's own quadrature."""
    mesh = Mesh(solution.problem.mesh.phase)
    mesh.set_matrices(solution.problem.spectral_method)
    phase = solution.phase[0]
    return float((phase.final_time - phase.initial_time) / 2 * (mesh.w[0] @ density))


def test_the_new_multipliers_have_their_shapes():
    for method in METHODS:
        phase = floor(method).solve().phase[0]
        assert phase.state_multiplier.shape == (3, len(phase.time_c))
        assert phase.initial_state_multiplier.shape == (3,)
        assert phase.final_state_multiplier.shape == (3,)
        assert phase.integral_bound_multiplier.shape == (1,)


def weights(solution):
    """Return the weight of each collocation point in an integral over the phase."""
    mesh = Mesh(solution.problem.mesh.phase)
    mesh.set_matrices(solution.problem.spectral_method)
    phase = solution.phase[0]
    return (phase.final_time - phase.initial_time) / 2 * mesh.w[0]


def raw_end_multipliers(problem, solution):
    """Return the NLP's own multipliers of the bounds on the initial and the final state."""
    index = get_nlp_dv_structure(problem, np.intp)
    index.z[:] = np.arange(len(index.z))
    nlp = solution.nlp_info
    multiplier = np.asarray(nlp.mult_x_U) - np.asarray(nlp.mult_x_L)
    return multiplier[np.asarray(index.phase[0].x0)], multiplier[np.asarray(index.phase[0].xf)]


@pytest.mark.parametrize("method", METHODS)
def test_the_density_and_the_atoms_give_the_sensitivity_to_the_bound(method):
    """The density's integral, and the final atom where the final point is not collocated.

    Under LGL the final point is collocated, so the density already holds its multiplier and
    the atom is the same number again. Under LGR and LG it is not, and the atom is the part
    of the sensitivity the density cannot hold.
    """
    solution = floor(method).solve()
    phase = solution.phase[0]
    total = integrate(solution, phase.state_multiplier[Y])
    if method != "lgl":  # the final point is a collocation point under LGL alone
        total += phase.final_state_multiplier[Y]
    assert -total == pytest.approx(sensitivity(method, b=0.45), rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_the_general_bound_s_multipliers_sum_to_the_sensitivity(method):
    """Every stored point the bound acts on, collocated or not, read from the solver's record."""
    problem = floor(method)
    solution = problem.solve()
    index = get_nlp_dv_structure(problem, np.intp)
    index.z[:] = np.arange(len(index.z))
    rows = np.asarray(index.phase[0].x[Y])
    nlp = solution.nlp_info
    at_points = (nlp.mult_x_U - nlp.mult_x_L)[rows]
    # all but the initial point, where y(0) = 0 is held by the initial bound
    initial = int(index.phase[0].x0[Y])
    total = at_points.sum() - (nlp.mult_x_U - nlp.mult_x_L)[initial]
    assert -total == pytest.approx(sensitivity(method, b=0.45), rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_an_end_multiplier_is_the_nlp_s_own_whichever_bound_is_active(method):
    """Every state, both ends: fixed, bounded by the general bound alone, or not bounded."""
    problem = floor(method)
    solution = problem.solve()
    phase = solution.phase[0]
    initial, final = raw_end_multipliers(problem, solution)
    np.testing.assert_array_equal(phase.initial_state_multiplier, initial)
    np.testing.assert_array_equal(phase.final_state_multiplier, final)
    assert phase.final_state_multiplier[Y] > 0  # the floor, which y has no final bound beside
    assert phase.initial_state_multiplier[Y] != 0.0  # y(0) = 0, held by the initial bound


@pytest.mark.parametrize("method", METHODS)
def test_a_fixed_end_value_s_multiplier_is_its_sensitivity(method):
    """x(tf) = xf has no general bound, so its end multiplier is the whole of what is there."""
    phase = floor(method).solve().phase[0]
    expected = sensitivity(method, xf=1.0)
    assert -phase.final_state_multiplier[X] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("final_y", [None, 0.45], ids=["no end bound", "equal bounds"])
def test_at_a_collocated_end_the_density_holds_the_atom_too(final_y):
    """The general bound is the tighter at the final point, or equal to the end's: LGL."""
    solution = floor("lgl", final_y=final_y).solve()
    phase = solution.phase[0]
    atom = phase.final_state_multiplier[Y]
    assert atom > 0
    assert phase.state_multiplier[Y, -1] * weights(solution)[-1] == pytest.approx(atom, rel=1e-12)


@pytest.mark.parametrize("method", ["lgl", "lgr"])
def test_the_density_is_zero_at_a_collocated_end_whose_own_bound_is_tighter(method):
    """y(0) = 0 is held by the initial bound, not the floor, so the density does not see it.

    Under LGL and LGR the initial point is a collocation point, the first of `time_c`.
    """
    phase = floor(method).solve().phase[0]
    assert phase.time_c[0] == phase.time[0]
    assert phase.state_multiplier[Y, 0] == 0.0
    assert phase.initial_state_multiplier[Y] != 0.0


def test_a_tighter_final_bound_takes_the_final_point_from_the_density():
    """A final bound just inside the floor, under LGL: the atom is its, the density zero."""
    phase = floor("lgl", final_y=0.44).solve().phase[0]
    assert phase.state[Y, -1] == pytest.approx(0.44, abs=1e-7)
    assert phase.final_state_multiplier[Y] > 0
    assert phase.state_multiplier[Y, -1] == 0.0


inf = np.inf


@pytest.mark.parametrize(
    ("raw", "general", "end", "share"),
    [
        # the upper side is active where the multiplier is positive, else the lower
        pytest.param(2.0, (0.0, 10.0), (-inf, inf), 2.0, id="no end bound, upper"),
        pytest.param(-2.0, (0.0, 10.0), (-inf, inf), -2.0, id="no end bound, lower"),
        pytest.param(2.0, (0.0, 10.0), (0.0, 12.0), 2.0, id="general tighter"),
        pytest.param(2.0, (0.0, 10.0), (0.0, 10.0), 2.0, id="equal, upper"),
        pytest.param(-2.0, (0.0, 10.0), (0.0, 10.0), -2.0, id="equal, lower"),
        pytest.param(2.0, (0.0, 10.0), (0.0, 9.0), 0.0, id="end tighter, upper"),
        pytest.param(-2.0, (0.0, 10.0), (0.1, 10.0), 0.0, id="end tighter, lower"),
        pytest.param(2.0, (0.0, 10.0), (0.1, 10.0), 2.0, id="end tighter on the idle side"),
        # an end fixed on the general bound: the sign says whether the general bound is active
        pytest.param(-2.0, (0.0, 10.0), (0.0, 0.0), -2.0, id="fixed on the bound, lower"),
        pytest.param(2.0, (0.0, 10.0), (0.0, 0.0), 0.0, id="fixed on the bound, upper"),
        pytest.param(2.0, (0.0, 10.0), (5.0, 5.0), 0.0, id="fixed inside, upper"),
        pytest.param(-2.0, (0.0, 10.0), (5.0, 5.0), 0.0, id="fixed inside, lower"),
        pytest.param(2.0, (-inf, inf), (5.0, 5.0), 0.0, id="no general bound"),
        pytest.param(0.0, (0.0, 10.0), (0.0, 10.0), 0.0, id="zero multiplier"),
        pytest.param(1e-30, (0.0, 10.0), (0.0, 10.0), 1e-30, id="no threshold"),
    ],
)
def test_the_general_bound_s_share_at_a_collocated_end(raw, general, end, share):
    def pair(bound):
        return np.array([bound[0]]), np.array([bound[1]])

    (result,) = _general_share(np.array([raw]), pair(general), pair(end))
    assert result == share


@pytest.mark.parametrize("method", METHODS)
def test_an_integral_bound_s_multiplier_is_its_sensitivity(method):
    """The effort bounded below its free value, so the bound is active."""
    phase = floor(method, effort=0.2).solve().phase[0]
    assert phase.integral[0] == pytest.approx(0.2, rel=1e-7)  # within Ipopt's bound relaxation
    expected = sensitivity(method, effort=0.2)
    assert -phase.integral_bound_multiplier[0] == pytest.approx(expected, rel=1e-5)
