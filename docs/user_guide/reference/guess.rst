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

.. doctest:: guess-rosenbrock

    >>> from yapss._legacy import Problem
    >>>
    >>> problem = Problem(name="Rosenbrock", nx=[], ns=2)
    >>> problem.guess.parameter = [-2.0, 2.0]

``guess.parameter`` accepts real numbers --- Python ``int`` and ``float``, NumPy integers and
floats, and any sequence or array of them --- with length ``ns``. A string, a bool, a complex
value, or ``None``, whole or inside the sequence, raises ``TypeError`` naming the attribute; a
wrong length or a non-finite value raises ``ValueError``.

The initial guess array is stored as a NumPy array in the ``guess.parameter`` attribute, so
individual elements can be modified using indexing or slicing. The example above could also be
written as:

.. doctest:: guess-rosenbrock

    >>> problem.guess.parameter[0] = -2.0
    >>> problem.guess.parameter[1] = 2.0

The default initial guess for the parameters is an array of zeros.

Initial Guess for Integrals
---------------------------

The initial guess for integral values is similar to that for parameters. Assign a one-dimensional
array-like object to ``guess.phase[p].integral``, where ``p`` is the phase index. The length of this
array-like object should match the number of integrals in the phase. For instance, in the
isoperimetric problem, we might have:

.. doctest:: guess-isoperimetric

    >>> from yapss._legacy import Problem
    >>>
    >>> problem = Problem(name="Isoperimetric Problem", nx=[2], nu=[2], nq=[3], nh=[1], nd=4)
    >>> problem.guess.phase[0].integral = [0.0, 0.0, 0.0]

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

.. doctest:: guess-dynamic-soaring

    >>> import numpy as np
    >>> from yapss._legacy import Problem
    >>>
    >>> problem = Problem(name="Dynamic Soaring", nx=[6], nu=[2], nh=[1], ns=1, nd=3)
    >>>
    >>> pi = np.pi
    >>> tf = 24
    >>> one = np.ones(50, dtype=float)
    >>> t = np.linspace(0, tf, num=50, dtype=float)
    >>> y = -200 * np.sin(2 * pi * t / tf)
    >>> x = 600 * (np.cos(2 * pi * t / tf) - 1)
    >>> h = -0.7 * x
    >>> v = 150 * one
    >>> gamma = 0 * one
    >>> psi = np.radians(t / tf * 360)
    >>> cl = 0.5 * one
    >>> phi = np.radians(45) * one
    >>>
    >>> problem.guess.phase[0].time = t
    >>> problem.guess.phase[0].state = x, y, h, v, gamma, psi
    >>> problem.guess.phase[0].control = cl, phi
    >>> problem.guess.parameter = 0.08,

The ``reset()`` Method
----------------------

To start a guess over, call the ``reset()`` method of the guess or of one of its phases. The
guess is then as it was when the problem was created: the time, state, and control guesses
are unset, and the integral and parameter guesses are zeros. Continuing the example above:

.. doctest:: guess-dynamic-soaring

    >>> problem.guess.phase[0].reset()  # Resets the guess for phase 0
    >>> print(problem.guess.phase[0].time)
    None
    >>> problem.guess.reset()  # Resets the whole guess
    >>> problem.guess.parameter
    array([0.])

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

    from yapss._legacy import Problem

    problem = Problem(name="Bryson Minimum Time to Climb", nx=[4], nu=[1])

    # more code here to define the problem ...

    solution = problem.solve()  # solve the minimum time to climb problem

    # modify the problem to solve the minimum fuel to climb problem:
    problem.sense = "maximize"

    def objective_2(arg):
        arg.objective = arg.phase[0].final_state[3]  # final vehicle mass

    problem.functions.objective = objective_2   # change only the objective function
    problem.guess(solution)   # use prior solution as a guess
    solution_2 = problem.solve()  # solve the minimum fuel to climb problem

Alternatively, the initial guess can be set explicitly using the ``from_solution`` method:

.. code-block:: python

    problem.guess.from_solution(solution)

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

Class Reference
---------------

.. autoclass:: yapss._legacy.guess.Guess
   :members:
   :no-special-members:
   :no-undoc-members:
