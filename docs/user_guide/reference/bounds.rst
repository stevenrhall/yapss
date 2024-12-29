Setting Bounds
==============

After a problem has been instantiated, bounds on decision variables and constraints can be
set using the ``bounds`` attribute of the problem object. The hierarchical structure of
the ``Bounds`` class reflects the structure of the problem decision variables and
constraints. In YAPSS, all bounds are initialized with default values *+inf* for upper
bounds and *-inf* for lower bounds, except for phase durations, for which the lower bound
defaults to zero.

Consider the HS071 problem example below, where bounds are defined for parameters and discrete
constraints.

.. doctest:: group1

    >>> from yapss import Problem
    >>>
    >>> problem = Problem(name="HS071", nx=[], ns=4, nd=2)
    >>> bounds = problem.bounds
    >>> bounds.parameter.lower = [1.0, 1.0, 1.0, 1.0]
    >>> bounds.parameter.upper = [5.0, 5.0, 5.0, 5.0]
    >>> bounds.discrete.lower = 25.0, 40.0
    >>> bounds.discrete.upper[1] = 40.0

In this example, there are four decision variables (the four parameters), and two discrete
constraints. Each parameter is bounded between 1 and 5, and the first discrete constraint is bounded
between 25 and 40, while the second discrete constraint is bounded below by 40, without an upper
limit.

Additional bounds can be set for each phase in the problem, covering variables like initial and
final time, duration, states, controls, integral values, and path constraints.

Except for the bounds on the initial time, final time, and duration of each phase, all the bounds
are arrays, and their values are implemented as NumPy arrays. Bound values to be set can be
addressed using either slicing or direct assignment. For instance, the following three lines are functionally
equivalent:

.. doctest:: group1

    >>> bounds.parameter.lower[:] = [1.0, 1.0, 1.0, 1.0]
    >>> bounds.parameter.lower = [1.0, 1.0, 1.0, 1.0]
    >>> bounds.parameter.lower = 1.0, 1.0, 1.0, 1.0


List of Bounds
--------------

The available bounds include:

**Time Bounds**:

- ``bounds.phase[k].initial_time.upper``, ``bounds.phase[k].initial_time.lower``
- ``bounds.phase[k].final_time.upper``, ``bounds.phase[k].final_time.lower``
- ``bounds.phase[k].duration.upper``, ``bounds.phase[k].duration.lower``

To ensure feasibility, set these bounds with care. For example, each phase must satisfy:

    ``initial_time.lower`` ≤ ``final_time.upper``

and

    ``final_time.lower`` - ``initial_time.upper`` ≤ ``duration.upper``

**State Bounds**:

- ``bounds.phase[k].initial_state.upper``, ``bounds.phase[k].initial_state.lower``
- ``bounds.phase[k].final_state.upper``, ``bounds.phase[k].final_state.lower``
- ``bounds.phase[k].state.upper``, ``bounds.phase[k].state.lower``

These bounds should be consistent across phases. For example:

    ``final_state.lower`` ≤ ``state.upper``

**Control, Path, and Integral Bounds**:

- ``bounds.phase[k].control.upper``, ``bounds.phase[k].control.lower``
- ``bounds.phase[k].path.upper``, ``bounds.phase[k].path.lower``
- ``bounds.phase[k].integral.upper``, ``bounds.phase[k].integral.lower``

**Parameter and Discrete Constraint Bounds**:

- ``bounds.parameter.upper``, ``bounds.parameter.lower``
- ``bounds.discrete.upper``, ``bounds.discrete.lower``

Example
-------

Consider the dynamic soaring problem. It has six states, two controls, one path
constraint, three discrete constraints, and a single parameter for wind shear rate. We
initialize the bounds for this problem as follows:

.. doctest:: group2

    >>> from yapss import Problem
    >>> import numpy as np
    >>> problem = Problem(name="dynamic soaring", nx=[6], nu=[2], nh=[1], ns=1, nd=3)

The initial time is set to zero, with an expected duration between 10 and 30 seconds:

.. doctest:: group2

    >>> # The initial time is fixed to be 0. The final time will be between 10 and 30.
    >>> bounds = problem.bounds.phase[0]
    >>> bounds.initial_time.lower = bounds.initial_time.upper = 0
    >>> bounds.final_time.lower = 10
    >>> bounds.final_time.upper = 30
    >>>
    >>> # The initial time is fixed to be 0. The final time will be between 10 and 30.
    >>> bounds.initial_time.lower = 0
    >>> bounds.initial_time.upper = 0
    >>> bounds.final_time.lower = 10
    >>> bounds.final_time.upper = 30

The initial and final states are set to zero, and there are bounds on the control
variables (lift coefficient and bank angle) from the problem statement:

.. doctest:: group2

    >>> # The initial and final positions are at the origin
    >>> bounds.initial_state.lower[:3] = bounds.initial_state.upper[:3] = 0, 0, 0
    >>> bounds.final_state.lower[:3] = bounds.final_state.upper[:3] = 0, 0, 0
    >>>
    >>> # CL_max <= 1.5. Set loose box bound on bank angle.
    >>> bounds.control.lower = 0, np.radians(-75)
    >>> bounds.control.upper = 1.5, np.radians(75)
    >>>
    >>> # Limits on the normal load
    >>> bounds.path.lower = (-2,)
    >>> bounds.path.upper = (5,)

There's also a discrete constraint that imposes a periodicity condition on the velocity,
flight path angle, and heading angle:

.. doctest:: group2

    >>> # Discrete constraints
    >>> problem.bounds.discrete.lower = problem.bounds.discrete.upper = 0, 0, np.radians(360)

In addition, it's good practice to set loose box bounds on the decision variables, which
can sometimes improve the performance of the Ipopt solver:

.. doctest:: group2

    >>> # Set loose box bounds on the state.
    >>> # None of these should be active in the solution.
    >>> bounds.state.lower = -1500, -1000, 0, 10, np.radians(-75), np.radians(-225)
    >>> bounds.state.upper = +1500, +1000, 1000, 350, np.radians(75), np.radians(225)

Bounds on initial and final states, position, and control variables are then set based on the
problem requirements, as shown in this detailed example.

Special Considerations for State Bounds
---------------------------------------

.. note::

    This section applies only to problems where state bounds act as path constraints, and
    where accurate Lagrange multipliers are required.

Broadly speaking, state constraints of the form

.. doctest:: group2

    >>> problem.bounds.phase[0].state.lower[1] = -1000.0
    >>> problem.bounds.phase[0].state.upper[1] = +1000.0

might be used in one of two ways:

1.	As inactive bounds to aid solver convergence without constraining the final solution.
2.	As path constraints, where bounds are expected to be active in the final solution.

In the case where state bounds are intended to be path constraints, the user should instead
use path constraints in the user-defined continuous function, as below:

.. doctest:: group2

    >>> def continuous(arg):
    >>>     # Apply state[1] as a path constraint
    >>>     arg.phase[0].path[0] = arg.phase[0].state[1]
    >>>
    >>> problem.continuous = continuous
    >>> problem.bounds.phase[0].path.lower[0] = -1000.0
    >>> problem.bounds.phase[0].path.upper[0] = +1000.0

While both approaches yield the correct primal solution (decision variables), state bounds
applied directly as path constraints may lead to incorrect Lagrange multipliers —
particularly for initial and final states. In order to obtain accurate numerical results,
path constraints should be applied and Lagrange multipliers calculated only at collocation
points, not all interpolation points, to be consistent with pseudospectral integration
scheme. In addition, without additional logic, it's difficult to determine whether the
Lagrange multipliers returned by the NLP solver should be associated with the endpoint
state constraints or the state path constraint.

For these reasons, state bounds expected to be active should be implemented as true path
constraints, as shown in the example above. Note that these considerations do *not* apply
to constraints on control variables.

The ``reset()`` Method
----------------------

To reset bounds, call the ``reset()`` method at any level in the ``Bounds`` hierarchy. For
instance:

.. doctest:: group2

   >>> problem.bounds.reset()  # Resets all bounds

or for a single phase:

.. doctest:: group2

   >>> problem.bounds.phase[0].reset()  # Resets bounds for phase 0

Input Validation
----------------

Setting bounds correctly can be error-prone, so YAPSS helps reduce errors by validating
bounds configurations. Errors are caught when assigning values, with helpful feedback. For
example, trying to assign a scalar instead of a sequence for path constraints raises an
error:

.. doctest:: group2

    >>> bounds.path.lower = -2
    Traceback (most recent call last):
        ...
    ValueError: ArrayBound must be a sequence of floats of length 1.

Similarly, trying to set conflicting control bounds results in an error:

.. doctest:: group2

    >>> bounds.control.upper = 1.5, np.radians(0)
    >>> bounds.control.lower = 0, np.radians(75)
    >>> bounds.validate()
    Traceback (most recent call last):
        ...
    ValueError: Failed to set bound.phase[0].control.lower.
    Must have bound.phase[0].control.upper[i] >= bound.phase[0].control.lower[i] for all i.
    Condition failed for i in [1].

``Bounds`` Class Reference
--------------------------

.. autoclass:: yapss._private.bounds.Bounds
   :members:
   :no-special-members:
   :no-undoc-members:
