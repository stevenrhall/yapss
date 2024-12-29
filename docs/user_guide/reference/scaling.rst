Scaling
=======

Theory
------

YAPSS, like other pseudospectral optimal control solvers, converts the optimal control
problem into a nonlinear programming problem (NLP) by discretizing the the continuous
decision variables (the state and control variables) and continuous constraint functions
(the state dynamics and path constraints). In addition, there are inherently additional
discrete decision variables (the static parameters) and constraint functions (discrete
constraints at the boundaries of phases, and bounds on integrals, for example). Together,
all these discrete decision variables and constraint functions, along with the objective
function, are passed to the NLP solver (Ipopt) to find the optimal solution.

An inherent difficulty in solving NLPs is that the problem as described in natural units
may be very badly scaled. For example, the `orbit raising problem
<../notebooks/orbit_raising.ipynb>`_ has different variables with very different
magnitudes. The distance of the spacecraft from the Sun as it transits from the Earth to
Mars (one of the decision variables) varies from :math:`1.5 \times 10^{11}\text{ m}`
meters :math:`1.5 \times 10^{11}\text{ m}`. On the other hand, the angular position of the
spacecraft measured in radians varies from 0 to less that :math:`\pi`. Large variation in
magnitudes of the variables and constraints make the problem ill-conditioned, and can
result in very slow convergence of the NLP solver, or even failure to converge.

There are two ways to improve the conditioning of the NLP problem:

1.  The problem can be scaled by hand, by using units that are more appropriate for the
    problem. That is in fact what is done in the orbit raising problem, where the distance
    is measured in astronomical units, and the angle in radians.

2.  The problem can be scaled within the NLP solver. Ipopt has has the option to provide
    scaling factors for the variables and constraints, which it then uses to scale the
    problem internally. YAPSS uses this feature to provide scaling through its API.

If scaling a problem by hand, one natural approach to finding the natural scale for a
decision variable (say, the state :math:`x`) is to use the range (or perhaps half the
range) that the variable is expected to have, and define that to be that variable scale,
:math:`S_x`. Then the nondimensional variable is

.. math::

    \bar{x} = \frac{x}{S_x}

The same approach can be used for the control inputs :math:`u` and parameters :math:`p`.
(Of course it would be done elementwise for vector variables.) The time scale for the
independent time variable :math:`t` would typically be the expected length of the phase.
Then the state dynamics

.. math::

    \dot{x} = f(x, u, p, t)

can be nondimensionalized as

.. math::

    \frac{d\bar{x}}{d\bar{t}} &= \bar{f}(\bar{x}, \bar{u}, \bar{p}, \bar{t}) \\
    &= \frac{S_t}{S_x} f(S_x \bar{x}, S_u \bar{u}, S_p \bar{p}, S_t \bar{t})

where :math:`S_t` is the time scale.

Scaling the constraints is a bit different. For example, a path constraint of the
form

.. math::

    g(x, u, p, t) = 0

is *expected* to be zero, and so it's not obvious how the function should be scaled.
The answer is that we should consider how the constraint varies with its arguments.
Perturbations in the constraints are, for small perturbation about the final solution,

.. math::

    \delta g = \frac{\partial g}{\partial x} \delta x
        + \frac{\partial g}{\partial u} \delta u
        + \frac{\partial g}{\partial p} \delta p
        + \frac{\partial g}{\partial t} \delta t

If :math:`g` were a function of only one variable, say :math:`x`, then the scale of the
constraint function would be

.. math::

    S_g = \frac{\partial g}{\partial x} S_x

If the constraint is a function of several variables, then we might take the constraint
function scale to be

.. math::

    S_g = \max\left(\frac{\partial g}{\partial x} S_x, \frac{\partial g}{\partial u} S_u,
    \frac{\partial g}{\partial p} S_p, \frac{\partial g}{\partial t} S_t\right)

or some other combination of the scales of the variables, such as the sum or the root mean
square.

Finally, it should be noted that one doesn't actually have to determine the partial
derivatives of the constraints to scale the problem — one just needs a good approximation
of the sensitivity of the constraint to the variables.

YAPSS Scaling
-------------

YAPSS provides scaling through the ``scale`` attribute of ``Problem`` instances. The
attributes that can be set are:

-    ``scale.objective`` (`float`): Object function scale.
-    ``scale.parameter`` (`Sequence[float]`): Parameter scale.
-    ``scale.discrete`` (`Sequence[float]`): Discrete constraint function scale.

-    ``scale.phase[k].state`` (`Sequence[float]`): State variable scale for phase ``k``.
-    ``scale.phase[k].control`` (`Sequence[float]`): Control variable scale for phase ``k``.
-    ``scale.phase[k].time`` (`float`): Time variable scale for phase ``k``.
-    ``scale.phase[k].path`` (`Sequence[float]`): Path constraint function scale for phase ``k``.
-    ``scale.phase[k].dynamics`` (`Sequence[float]`): Dynamics constraint scale for phase ``k``.
-    ``scale.phase[k].integral`` (`Sequence[float]`): Integral scale for phase ``k``.

Two of these attributes require further explanation. First, the value of an integral over
a phase is actually a decision variable in the YAPSS implementation, and the condition
that this decision variable is equal to the integral of the integrand function over the
phase is a constraint. The ``scale.phase[k].integral`` attribute is the scale for both the
decision variable and the constraint.

Second, based on the discussion in the `Theory`_ section, one would expect the dynamics
constraint scale to be the state scale divided by the time scale. However, because of the
way the dynamics is implemented, the dynamics scale should usually be set to the state
scale.

All the scales described are initialized to 1.0 (or an array of ones), so for problems
that are nicely scaled, no scaling is necessary.


Note also that each of the array scales can be set elementwise, or using a slice.

Example
-------

Consider for example the `dynamic soaring problem <../notebooks/dynamic_soaring.ipynb>`_,
which is the problem to find a trajectory allows a bird or glider to fly continuously
using dynamic soaring, with the minimum possible wind speed gradient. The optimization for
this problems performs quite poorly without proper scaling — it fails to converge at all.

The state variables are the three spatial positions of the glider, its velocity, its
flight path angle, and its heading angle. The control variables are the lift coefficient
and the bank angle. The one parameter is the wind speed gradient. There is one path
constraint, that the load factor be in the range :math:`[-2,5]`. There are three discrete
constraints, that the three components of the initial velocity be equal to the components
of the final velocity.

The scales for each variable were set to be roughly the expected range of the variable,
and the scales for the discrete constraints were set as discussed in the Theory section:

.. code-block:: python

    # scales for the problem
    scale = problem.scale
    scale.objective = 0.1
    scale.parameter = [0.1]
    scale.discrete = [200.0, 200.0, 200.0]

    # scales for the first (and only) phase
    phase = scale.phase[0]
    phase.dynamics = phase.state = 1000.0, 1000.0, 1000.0, 200.0, 1.0, 6.0
    phase.control = 1.0, 1.0
    phase.time = 30.0
    phase.path = [7.0]

With these scales, the problem converges in a quite reasonable number of iterations (about
32).
