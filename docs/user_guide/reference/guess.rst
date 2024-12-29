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
``guess.parameters`` attribute, with length equal to the number of parameters in the optimization.
For example, in the Rosenbrock problem, we might have:

.. code-block:: python

    from yapss import Problem

    problem = Problem(name="Rosenbrock", nx=[], ns=2)
    problem.guess.parameter = [-2.0, 2.0]

An exception will be raised if the object assigned to ``guess.parameters`` cannot be converted to a
numpy array with ``dtype=float`` and shape ``(ns,)``.

The initial guess array is stored as a numpy array in the ``guess.parameters`` attribute, so
individual elements can be modified using indexing or slicing. The example above could also be
written as:

.. code-block:: python

    problem.guess.parameter[0] = -2.0
    problem.guess.parameter[1] = 2.0

The default initial guess for the parameters is an array of zeros.

Initial Guess for Integrals
---------------------------

The initial guess for integral values is similar to that for parameters. Assign a one-dimensional
array-like object to ``guess.phase[p].integral``, where ``p`` is the phase index. The length of this
array-like object should match the number of integrals in the phase. For instance, in the
isoperimetric problem, we might have:

.. code-block:: python

    from yapss import Problem

    problem = Problem(name="Isoperimetric Problem", nx=[2], nu=[2], nq=[3], nh=[1], nd=4)
    problem.guess.phase[0].integral = [0.0, 0.0, 0.0]

An exception is raised if the object assigned to ``guess.phase[p].integral`` cannot be converted to
a numpy array with ``dtype=float`` and shape ``(nq[p],)``.

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
control histories should be two-dimensional array-like objects, with shapes ``(k, nx[p])`` and ``(k,
nu[p])``, respectively.

The ``guess.phase[p].state`` and ``guess.phase[p].control`` attributes default to ``None`` until an
array is assigned. Therefore, indexing or slicing cannot be used for assignment until an array is
explicitly provided.

If no initial guess is assigned to the time array, an exception will be raised when the ``solve()``
method is called. An exception is also raised if the shapes of the arrays assigned to
``guess.phase[p].state`` or ``guess.phase[p].control`` are not ``(k, nx[p])`` and ``(k, nu[p])``,
respectively, where ``k`` is the length of the time array for phase ``p``.

If no array is assigned to ``guess.phase[p].state`` or ``guess.phase[p].control``, the default
initial guess is an array of zeros.

Below is an example from the Dynamic Soaring problem:

.. code-block:: python

    import numpy as np
    from yapss import Problem

    problem = Problem(name="Dynamic Soaring", nx=[6], nu=[2], nh=[1], ns=1, nd=3)

    pi = np.pi
    tf = 24
    one = np.ones(50, dtype=float)
    t = np.linspace(0, tf, num=50, dtype=float)
    y = -200 * np.sin(2 * pi * t / tf)
    x = 600 * (np.cos(2 * pi * t / tf) - 1)
    h = -0.7 * x
    v = 150 * one
    gamma = 0 * one
    psi = np.radians(t / tf * 360)
    cl = 0.5 * one
    phi = np.radians(45) * one

    problem.guess.phase[0].time = t
    problem.guess.phase[0].state = x, y, h, v, gamma, psi
    problem.guess.phase[0].control = cl, phi
    problem.guess.parameter = 0.08,

Initial Guess from Previous Solution
------------------------------------

The initial guess can also be set from a previous solution of the same (or similar) problem.
This approach is particularly useful in two scenarios:

-  Mesh Refinement – Start with a coarse mesh to find an initial solution, then use that
   solution as a guess for a finer mesh.
-  Problem Variations – Solve a related problem with minor modifications by using the solution
   from the original problem as an initial guess.

For example, in the `JupyterLab notebook <../notebooks/minimum_time_to_climb.ipynb>`_ that
solves the minimum time to climb problem, the resulting solution is reused as the initial
guess for solving the minimum fuel to climb problem. This approach provides a guess closer
to the ultimate solution and reduces computation time.

.. code-block:: python

    from yapss import Problem

    problem = Problem(name="Bryson Minimum Time to Climb", nx=[4], nu=[1])

    # more code here to define the problem ...

    solution = problem.solve()  # solve the minimum time to climb problem

    # modify the problem to solve the minimum fuel to climb problem:
    def objective_2(arg):
        arg.objective = -arg.phase[0].final_state[3]  # maximize final vehicle mass

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

.. autoclass:: yapss._private.guess.Guess
   :members:
   :no-special-members:
   :no-undoc-members:
