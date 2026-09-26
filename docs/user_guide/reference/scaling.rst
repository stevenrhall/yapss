Scaling
=======

Theory
------

YAPSS, like other pseudospectral optimal control solvers, converts the optimal control
problem into a nonlinear programming problem (NLP) by discretizing the continuous
decision variables (the state and control variables) and continuous constraint functions
(the state dynamics and path constraints). In addition, there are inherently additional
discrete decision variables (the static parameters) and constraint functions (discrete
constraints at the boundaries of phases, and bounds on integrals, for example). Together,
all these discrete decision variables and constraint functions, along with the objective
function, are passed to the NLP solver (Ipopt) to find the optimal solution.

An inherent difficulty in solving NLPs is that the problem as described in natural units
may be badly scaled. For example, the `orbit raising problem
<../notebooks/orbit_raising.ipynb>`_ has different variables with very different
magnitudes. The distance of the spacecraft from the Sun as it transits from the Earth to
Mars (one of the decision variables) varies from approximately :math:`1.5 \times 10^{11}\text{ m}`
to :math:`2.3 \times 10^{11}\text{ m}`. On the other hand, the angular position of the
spacecraft measured in radians varies from 0 to less than :math:`\pi`. A large variation in
magnitudes of the variables and constraints makes the problem ill-conditioned, and can
result in very slow convergence of the NLP solver, or even failure to converge.

There are two ways to improve the conditioning of the NLP problem:

1.  The problem can be scaled by hand, by using units that are more appropriate for the
    problem. That is in fact what is done in the orbit raising problem, which is written in
    canonical units: the initial radius of the orbit is the unit of distance, and the
    gravitational parameter is 1.

2.  The problem can be scaled within the NLP solver. Ipopt has the option to provide
    scaling factors for the variables and constraints, which it then uses to scale the
    problem internally. YAPSS uses this feature to provide scaling through its API.

If scaling a problem by hand, one natural approach to finding the natural scale for a
decision variable (say, the state :math:`x`) is to use the range (or perhaps half the
range) that the variable is expected to have, and define that to be the variable's scale,
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

Every quantity the solver sees has a scale, set where the quantity is declared. A scale is a
characteristic magnitude: one finite, positive number. An assignment that is not raises
``TypeError`` or ``ValueError`` at the line that makes it, and leaves the previous value in
place. Every scale is 1.0 until it is set, so a problem that is already well scaled needs none.
For a phase ``ph`` of a problem ``problem``, the scales are:

-   ``ph.state.x.scale``: the state ``x``.
-   ``ph.state.x.defect_scale``: the collocation defects of the dynamics of ``x``.
-   ``ph.control.u.scale``: the control ``u``.
-   ``ph.time.scale``: the phase's initial and final times, and the constraint on its
    duration. A phase that names its independent variable otherwise uses that name:
    ``ph.r.scale`` for a variable ``r``.
-   ``ph.path.g.scale``: the path constraint ``g``.
-   ``ph.integral.q.scale``: the integral ``q``.
-   ``problem.parameter.p.scale``: the parameter ``p``.
-   ``problem.discrete.d.scale``: the discrete constraint ``d``.
-   ``problem.objective.scale``: the objective. It conditions the objective's magnitude only;
    to maximize rather than minimize, set ``problem.objective.sense`` rather than a negative
    scale.

Two of these need further explanation. First, the value of an integral over a phase is a
decision variable in the YAPSS implementation, and the condition that this variable equals the
integral of the integrand over the phase is a constraint. An integral's ``scale`` is the scale
of both.

Second, the state has two scales where every other quantity has one. Based on the discussion
in the `Theory`_ section, one would expect the dynamics to be scaled by the state scale divided
by the time scale. But YAPSS writes the dynamics as collocation defects,

.. math::

    \frac{t_f - t_0}{2} f(x, u, p, t) - D x

where :math:`D` is the differentiation matrix on the normalized interval and is dimensionless,
so a defect is in the units of the state. So ``defect_scale`` should usually be set to the same
value as ``scale``.

A field declared with ``yapss.vector(n)`` has one scale per row, set by row:
``ph.state.r.scale[:] = 1000.0`` gives every row the same scale,
``ph.state.r.scale[:] = [1000.0, 1000.0, 10.0]`` one each, and ``ph.state.r.scale[0] = 1000.0``
one row. Each value is checked as a whole assignment is, and a refused write leaves every row
unchanged.

Example
-------

Consider for example the `dynamic soaring problem <../notebooks/dynamic_soaring.ipynb>`_,
which is the problem to find a trajectory that allows a bird or glider to fly continuously
using dynamic soaring, with the minimum possible wind speed gradient.

The state variables are the three spatial positions of the glider, its velocity, its flight
path angle, and its heading angle. The control variables are the lift coefficient and the bank
angle. The one parameter is the wind speed gradient. There is one path constraint: the load
factor is in the range :math:`[-2,5]`. There are three discrete constraints that impose
periodicity of velocity, flight-path angle, and heading angle.

The scale of each variable was set to roughly its expected range, and the scales of the
discrete constraints as discussed in the `Theory`_ section; the controls are left at 1.0:

.. literalinclude:: ../../../src/yapss/examples/dynamic_soaring.py
   :language: python
   :start-after: # Scaling, which this problem needs
   :end-before: # A dense mesh
   :dedent: 4

With these scales the problem converges in about 30 iterations. With every scale left at 1.0 it
still converges, to the same optimum, but takes about twice as many.
