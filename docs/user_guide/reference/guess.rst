Initial Guess
=============

To solve an optimal control problem using YAPSS, the user must provide an initial guess for the
decision variables. These include:

- Initial and final times,
- State and control histories for each phase, and
- Any parameters and integrals in the optimization.

The initial guess for a problem is set using the ``guess`` attribute of the ``Problem`` instance,
which is structured with attributes corresponding to the decision variables:

* ``guess.parameter``
* ``guess.phase[p].time``
* ``guess.phase[p].state``
* ``guess.phase[p].control``
* ``guess.phase[p].integral``

where ``p`` is the phase index.

Initial Guess for Parameters
----------------------------

To initialize the guess for the parameter array, assign a one-dimensional array-like object to the
``guess.parameter`` attribute, with length equal to the number of parameters in the optimization.
For example, in the Rosenbrock problem, we might have:

.. code-block:: python

    import yapss


    class Parameter(yapss.Parameter):
        x = yapss.scalar()
        y = yapss.scalar()


    class Rosenbrock(yapss.Problem):
        parameter: Parameter


    problem = Rosenbrock("Rosenbrock")
    problem.parameter.x.guess = -2.0
    problem.parameter.y.guess = 2.0

``guess.parameter`` accepts real numbers --- Python ``int`` and ``float``, NumPy integers and
floats, and any sequence or array of them --- with length ``ns``. A string, a bool, a complex
value, or ``None``, whole or inside the sequence, raises ``TypeError`` naming the attribute; a
wrong length or a non-finite value raises ``ValueError``.

The initial guess array is stored as a NumPy array in the ``guess.parameter`` attribute, so
individual elements can be modified using indexing or slicing. The example above could also be
written as:

.. code-block:: python

    class Parameter(yapss.Parameter):
        point = yapss.vector(2)


    class Rosenbrock(yapss.Problem):
        parameter: Parameter


    problem = Rosenbrock("Rosenbrock")
    problem.parameter.point.guess[:] = [-2.0, 2.0]
    problem.parameter.point.guess[0] = -2.0
    problem.parameter.point.guess[1] = 2.0

The default initial guess for the parameters is an array of zeros.

Initial Guess for Integrals
---------------------------

The initial guess for integral values is similar to that for parameters. Assign a one-dimensional
array-like object to ``guess.phase[p].integral``, where ``p`` is the phase index. The length of this
array-like object should match the number of integrals in the phase. For instance, in the
isoperimetric problem, we might have:

.. code-block:: python

    from yapss.examples.isoperimetric import setup

    problem = setup()
    ph = problem.phases.phase
    ph.integral.area.guess = 0.0
    ph.integral.x_moment.guess = 0.0
    ph.integral.y_moment.guess = 0.0

``guess.phase[p].integral`` accepts the same forms, with length ``nq[p]``, and refuses the same
values.

As with parameters, the default initial guess for each phase's integrals is an array of zeros. Thus,
in this example, we could omit the ``integral`` assignment and obtain the same result.

Initial Guess for Time, State, and Control
------------------------------------------

The initial guesses for the time, state, and control histories are more detailed than those for
parameters and integrals. The user-provided guess will be interpolated to the mesh points of the
phase. The time vector may have as few as two elements or more, depending on the desired level of
detail. The first and last elements of the time vector must be the initial and final times of the
phase, and the time vector must be strictly increasing.

If the time array for phase ``p`` contains ``k`` elements, the initial guesses for the state and
control histories should be two-dimensional array-like objects, with shapes ``(nx[p], k)`` and ``(nu[p],
k)``, respectively.

The ``guess.phase[p].state`` and ``guess.phase[p].control`` attributes are arrays of zeros of
the right shape until an array is assigned, so a guess can be built up by indexing or slicing
(``guess.phase[p].state[0, :] = ...``) as well as by assigning a whole array. The time array
fixes the shape of the zeros, so it must be set first; reading either attribute before that
raises ``ValueError``, unless an array has already been assigned to it.

If no initial guess is assigned to the time array, an exception will be raised when the ``solve()``
method is called. An exception is also raised if the shapes of the arrays assigned to
``guess.phase[p].state`` or ``guess.phase[p].control`` are not ``(nx[p], k)`` and ``(nu[p], k)``,
respectively, where ``k`` is the length of the time array for phase ``p``.

If no array is assigned to ``guess.phase[p].state`` or ``guess.phase[p].control``, the default
initial guess is an array of zeros. A guess that is still all zeros follows the time array: if
the time array is reassigned with a different length, the zeros are regenerated at the new
length, while an array with values in it is kept and reported by ``validate()`` if its length no
longer matches.

Below is an example from the Dynamic Soaring problem:

.. code-block:: python

    import numpy as np

    import yapss
    from yapss.examples.dynamic_soaring import setup
    from yapss.math import cos, pi, radians, sin

    problem = setup()
    ph = problem.phases.phase

    tf = 24.0
    t = np.linspace(0.0, tf, num=50)
    turn = 2 * pi * t / tf
    x = 600 * (cos(turn) - 1)
    ph.time.guess = (0.0, tf)
    ph.state.x.guess = yapss.interp(t, x)
    ph.state.y.guess = yapss.interp(t, -200 * sin(turn))
    ph.state.h.guess = yapss.interp(t, -0.7 * x)
    ph.state.v.guess = (150.0, 150.0)
    ph.state.gamma.guess = (0.0, 0.0)
    ph.state.psi.guess = yapss.interp(t, radians(t / tf * 360))
    ph.control.cl.guess = (0.5, 0.5)
    ph.control.phi.guess = (radians(45), radians(45))
    problem.parameter.beta.guess = 0.08

The ``reset()`` Method
----------------------

To start a guess over, call the ``reset()`` method of the guess or of one of its phases. The
guess is then as it was when the problem was created: the time, state, and control guesses
are unset, and the integral and parameter guesses are zeros. Continuing the example above:

.. code-block:: python

    # a guess starts over by assigning the default again
    for field in (ph.state.x, ph.state.y, ph.state.h):
        field.guess = (0.0, 0.0)
    problem.parameter.beta.guess = 0.0

Initial Guess from Previous Solution
------------------------------------

The initial guess can also be set from a previous solution of the same problem—for example,
after modifying its bounds—or from a related problem with different path or discrete constraints.
In either case, the previous solution must have the same number of phases, the same numbers of
states, controls, and integrals in each phase, and the same number of parameters. This approach
is particularly useful in two scenarios:

-  Mesh Refinement – Start with a coarse mesh to find an initial solution, then use that
   solution as a guess for a finer mesh.
-  Problem Variations – Solve a related problem with minor modifications by using the solution
   from the original problem as an initial guess.

For example, in the `JupyterLab notebook <../notebooks/minimum_time_to_climb.ipynb>`_ that
solves the minimum time to climb problem, the resulting solution is reused as the initial
guess for solving the minimum fuel to climb problem. This approach provides a guess closer
to the ultimate solution and reduces computation time.

.. code-block:: python

    from yapss.examples.minimum_time_to_climb import setup

    problem = setup()
    ph = problem.phases.phase

    solution = problem.solve()  # solve the minimum time to climb problem

    # modify the problem to solve the minimum fuel to climb problem:
    problem.objective.sense = "maximize"


    @problem.register.objective  # replaces the objective callback
    def objective_2(arg):
        return arg[ph].final_state.mass  # final vehicle mass


    problem.guess_from_solution(solution)  # use prior solution as a guess
    solution_2 = problem.solve()  # solve the minimum fuel to climb problem

Alternatively, the initial guess can be set explicitly using the ``from_solution`` method:

.. code-block:: python

    problem.guess_from_solution(solution, solution_phase="phase", guess_phase=ph)

Both methods achieve the same result. The first syntax (``problem.guess(solution)``) is
concise, while the second (``from_solution``) may enhance readability.

Impact of the Initial Guess
---------------------------

For small problems, the initial guess may have little effect on the solution, and the default guess
may be sufficient. However, for larger problems, the initial guess can significantly influence the
solution. A poor initial guess may cause the algorithm to converge to a local minimum or fail to
find a feasible solution. A common approach is to start with a simple guess, using only two time
points per phase. If this is unsuccessful, a more refined initial guess may be needed to bring it
closer to a feasible solution.
