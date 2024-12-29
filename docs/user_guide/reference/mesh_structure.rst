Mesh Structure
==============

Theory
------

In pseudospectral optimal control methods, the state variable history over an interval is
approximated by a polynomial, defined by the value of the state at specific points in
time, the *interpolation* points. The state variable is then defined the the *Lagrange
interpolating polynomial*, which is the polynomial of minimal degree that passes through
the interpolation points. In addition, some of the interpolation points are also
*collocation* points, where the dynamics (and also) path constraints are enforced. In
addition, any integrands in the problem are evaluated at the collocation points, and
integrated using Gauss-Radau quadrature.

For example, for the Legendre-Gauss-Radau (LGR) pseudospectral method with :math:`N`
collocation points over the interval :math:`[-1,1]`, the collocation points are defined as
the roots :math:`t_i` of the equation

.. math::

    P_N(t_i) +P_{N-1}(t_i) = 0,\quad i=0,1,\dots,N-1

where :math:`P_N(x)` is the :math:`N\text{th}` Legendre polynomial. All the collocation
points lie in the interval :math:`t_i\in[-1,1)`. There is an additional interpolation
point at :math:`t_N=1`.

Now consider the simplest example of solving an optimal control problem using the LGR
method, with a scalar state :math:`x(t)` and scalar control :math:`u(t)`, where the
problem is to optimize

.. math::

    J = \int_{-1}^{1} g(x(t),u(t)) \, dt

subject to the dynamics

.. math::

    \dot{x}(t) = f(x(t),u(t)),\quad x(-1) = x_0

The decision variables are the elements of the  :math:`N+1`-dimensional state vector
:math:`\boldsymbol{x}` and the :math:`N`-dimensional control vector
:math:`\boldsymbol{u}`, where

.. math::

    \boldsymbol{x}_i &= x(t_i),\quad N=0,1,\dots,N \\
    \boldsymbol{u}_i &= u(t_i),\quad N=0,1,\dots,N-1

Because the interpolating polymonial for :math:`\boldsymbol{x}` is unique, the time
derivative at each of the collocation points is also unique, and can be related to the
dynamics function :math:`f` evaluated at the collocation points. If the dynamics function
values is represented by the vector :math:`\boldsymbol{f}`, then the state dynamics
constraint becomes

.. math::

    D \boldsymbol{x} = \boldsymbol{f(\boldsymbol{x},\boldsymbol{u})}

where :math:`D` is the differentation matrix for the interpolating scheme. Further, the
integral is approximated by Gauss-Radau quadrature, so that the objective function becomes

.. math::

    J = \sum_{i=0}^{N-1} w_i g(\boldsymbol{x}_i,\boldsymbol{u}_i)

where :math:`w_i` are the quadrature weights.

The Legendre-Gauss (LG), Legendre-Gauss-Radau (LGR), and Legendre-Gauss-Lobatto (LGL)
interpolation points give (in some sense) the most accurate approximations for the
integral and the dynamics constraint for a given number of collocation points. The methods
can be remarkably accurate for a modest number of points, especially for problems that are
sufficiently smooth. For many problem, the errors become exponentially small as the number of
collocation points increases.

However, for some problems the error converges to zero slowly with increasing number of
collocation points. This is often the case for problems with discontinuities due to path
constraints, but can occur for other reasons as well. For such problems, a better approach
is to use a segmented mesh, where each phase is divided into segments, and each segment
has LG, LGR, or LGL collocation points, each perhaps with a smaller number of collocation
points. YAPSS provides for such user-defined mesh structures.

Defining the Mesh Structure
---------------------------

The mesh structure is controlled by the ``mesh`` attribute of ``Problem`` instances. The
attributes that can be set are:

-  ``mesh.phase[k].collocation_points`` (Sequence[int]): Number of collocation points for
   each segment of phase ``k``. Each integer must be greater than or equal to 3. The
   length of the sequence is the number segments in the phase.

-  ``mesh.phase[k].fraction`` (Sequence[float]): The fraction of the phase duration of
   each segment. Each element must be greater than 0.0 and less than 1.0, and the sum of
   the elements must be close 1.0. The length of the ``fraction`` attribute  must be the
   same as the length of the ``collocation_points`` attribute.

The default mesh structure is 10 segments of equal duration duration, each with 10 collocation
points. That is, the default is

.. code-block:: python

    problem.mesh.phase[k].collocation_points = 10 * [10]
    problem.mesh.phase[k].fraction = 10 * [0.1]

Consider first the `Delta III ascent problem <../notebooks/delta_iii_ascent.html>`_ to minimize the
fuel require to reach a specific orbit. The vehicle has four stages, and hence the problems has
four phases. The default mesh results in a very large number of decision variables, and hence the
problem is slow to solve. We can speed up the solution by reducing the number of collocation points:

.. code-block:: python

    # set the mesh structure for the Delta III ascent problem
    m, n = 5, 5  # 5 segments, each with 5 collocation points
    for p_ in range(4):
        problem.mesh.phase[p_].collocation_points = m * (n,)
        problem.mesh.phase[p_].fraction = m * (1.0 / m,)

Or consider the `dynamic soaring problem <../notebooks/dynamic_soaring.html>`_, where the objective
is to find the trajectory that allows a bird or glider to fly continuously using dynamics soaring,
with the minimum possible wind speed gradient. The solution is not smooth (it has discontinuous derivatives), because there is a
path constraint (on the maximum lift coefficient) that is active only for part of the soaring cycle.
To give a good solution even in the vicinity of the discontinuity, we can use a segmented mesh with
many segments:

.. code-block:: python

    # set the mesh structure for the dynamic soaring problem
    m, n = 50, 6  # 50 segments, each with 6 collocation points
    problem.mesh.phase[0].collocation_points = m * (n,)
    problem.mesh.phase[0].fraction = m * (1.0 / m,)
    problem.spectral_method = "lgl"

It would be better to refine the mesh only in the vicinity of the discontinuities, but YAPSS does not
yet support automatic mesh refinement.
