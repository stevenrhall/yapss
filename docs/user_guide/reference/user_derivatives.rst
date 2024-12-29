User-Defined Derivatives
========================

.. note::

   User-defined derivatives are rarely necessary. While they can be as accurate as
   automatic differentiation and may offer modest speed improvements, calculating them
   correctly requires significant effort. Most users can safely skip this section.

For most problems, the best choice for calculating derivatives is either automatic
differentiation or central difference numerical differentiation. However, there may be
cases where the user wants or needs to supply their own derivatives. For example, one
could have a problem for which almost all the derivatives are known analytically, but a
few must be calculated numerically. This section explains how to provide user-defined
derivatives.

The following example illustrates how to set up a problem with user-defined derivatives.
We use the classic brachistochrone problem, where the objective is to minimize the time
for a particle to slide down a curve between two points. This example shows how to specify
derivatives for objectives and constraints manually. (See the Jupyter Lab notebook for
additional problem details.) As usual, we start by defining the problem:

.. testcode:: group1

   from yapss import Problem
   from yapss import numpy as np
   from yapss.numpy import cos, pi, sin

   ocp = Problem(name="Brachistochrone", nx=[3], nu=[1], nq=[0])
   ocp.derivatives.method = "user"
   ocp.derivatives.order = "second"

   g0 = 32.174

   def objective(arg):
       arg.objective = arg.phase[0].final_time

   def continuous(arg):
       x, y, v = arg.phase[0].state
       (u,) = arg.phase[0].control
       arg.phase[0].dynamics[:] = v * cos(u), v * sin(u), g0 * sin(u)

Because we've specified that we're using user-defined, second-order derivatives, we need to
define the gradient and Hessian functions for the objective, and the Jacobian and Hessian
functions for the continuous functions.

The Continuous Jacobian and Hessian
-----------------------------------

We’ll start with the continuous function, which generically computes the dynamics function,
the integrand array, and the path constraint functions. For our example, the continuous
Jacobian function is defined as follows:

.. testcode:: group1

   def continuous_jacobian(arg):
       x, y, v = arg.phase[0].state
       (u,) = arg.phase[0].control

       jacobian = arg.phase[0].jacobian
       jacobian[("f", 0), ("x", 2)] = cos(u)
       jacobian[("f", 0), ("u", 0)] = -v * sin(u)
       jacobian[("f", 1), ("x", 2)] = sin(u)
       jacobian[("f", 1), ("u", 0)] = v * cos(u)
       jacobian[("f", 2), ("u", 0)] = g0 * cos(u)

   ocp.functions.continuous_jacobian = continuous_jacobian

For each phase ``p``, the `jacobian` dictionary (``arg.phase[p].jacobian``) contains entries
representing partial derivatives. The keys follow the pattern:

- ``("f", i)``: Element ``i`` of the dynamics function.
- ``("g", i)``: Element ``i`` of the integrand.
- ``("h", i)``: Element ``i`` of the path constraint.

Each entry is a partial derivative with respect to a decision variable. The decision variables are:

- ``("x", i)``: State vector element ``i``.
- ``("u", i)``: Control vector element ``i``.
- ``("s", i)``: Parameter vector element ``i``.
- ``("t", 0)``: Time variable.

The Hessian of the continuous functions is defined similarly:

.. testcode:: group1

   def continuous_hessian(arg):
       x, y, v = arg.phase[0].state
       (u,) = arg.phase[0].control

       hessian = arg.phase[0].hessian
       hessian[("f", 0), ("x", 2), ("u", 0)] = -sin(u)
       hessian[("f", 0), ("u", 0), ("u", 0)] = -v * cos(u)
       hessian[("f", 1), ("x", 2), ("u", 0)] = cos(u)
       hessian[("f", 1), ("u", 0), ("u", 0)] = -v * sin(u)
       hessian[("f", 2), ("u", 0), ("u", 0)] = -g0 * sin(u)

   ocp.functions.continuous_hessian = continuous_hessian

The main difference with the Hessian is that each entry represents a second partial derivative,
so two keys corresponding to decision variables are required for each partial derivative. Note that
for mixed partial derivatives, the order of the keys is unimportant. For example, if you set
``hessian[("f", 0), ("x", 2), ("u", 0)]``, ``hessian[("f", 0), ("u", 0), ("x", 2)]`` should not be set.

The Objective Gradient and Hessian
----------------------------------

We'll start with the derivative functions for the objective. The gradient function
is defined as follows:

.. testcode:: group1

   def objective_gradient(arg):
       arg.gradient[0, "tf", 0] = 1

   ocp.functions.objective_gradient = objective_gradient

The gradient values are stored in the dictionary `arg.gradient`. In this example, the key is
``(0, "tf", 0)``, which means the gradient is with respect to the final time of phase 0.
The gradient keys are of the form

- ``(p, "x0", i)``: Element ``i`` of the initial state vector in phase ``p``.
- ``(p, "xf", i)``: Element ``i`` of the final state vector in phase ``p``.
- ``(p, "t0", 0)``: The initial time of phase ``p``.
- ``(p, "tf", 0)``: The final time of phase ``p``.
- ``(0, "s", i)``: Element ``i`` of the parameter vector.

Because the gradient is constant with respect to the decision variables, the Hessian function
is zero, so the Hessian function is simply

.. testcode:: group1

   def objective_hessian(arg):
       return

   ocp.functions.objective_hessian = objective_hessian

Had the Hessian been nonzero, each key would have been a tuple of two decision variable keys, so for
example a key of the dictionary ``arg.hessian`` in a multiphase problem might be something like
``((3, "xf", 1), (0, "s", 2))``. More generally, the keys are of the form

   ``(p, var1, i1), (p, var2, i2)``

where ``p`` is the index of the phase (or 0 for an element of the parameter vector), ``var1`` and
``var2`` are one of the variables ``x0``, ``xf``, ``t0``, ``tf``, or ``s``, and ``i1`` and ``i2``
are the index of the variable (or 0 for ``t0`` for ``tf``).

The Discrete Jacobian and Hessian
---------------------------------

For problems with discrete functions, the discrete Jacobian and discrete Hessian are defined
similarly to the objective functions. The keys for the discrete Jacobian and Hessian dictionaries
have an additional index for the discrete function:

   ``(k, (p, var, i))``

where ``k`` is the discrete function index, and ``p``, ``var``, and ``i`` are as in
the objective derivative functions. Finally, the ``arg.hessian`` keys have the form

   ``(k, (p1, var1, i1), (p2, var2, i2))``
