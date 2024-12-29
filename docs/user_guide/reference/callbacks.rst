Callback Functions
==================

Callback functions form the core of defining and solving optimal control problems.
They allow the user to specify the objective, system dynamics, performance integrals,
path constraints, and discrete constraints of the problem.

This section outlines the callback functions required to define an optimal control
problem in YAPSS. Most problems will require at most three callback functions:

- The objective callback is **always required**.
- The continuous callback is **optional, but required for dynamic problems**.
- The discrete callback is **optional, but required when discrete constraints exist**.

If a required callback is not defined for a problem, the solver will raise a ``ValueError``
when the problem ``solve()`` method is called.

Objective Callback Function
---------------------------

The general expression for an objective function for a problem with :math:`P` phases is

.. math::

   \begin{aligned}
      J=\phi\Big[
         & x^{(0)}(t_{0}^{(0)}),\ldots,x^{(P-1)}(t_{0}^{(P-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(P-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(P-1)}(t_{f}^{(P-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(P-1)},q^{(0)},\ldots,q^{(P-1)},s\Big]
   \end{aligned}

where :math:`\phi` is a scalar function, :math:`x^{(p)}(t)` is the state vector
for phase :math:`p`, :math:`t_{0}^{(p)}` is the initial time for phase :math:`p`,
:math:`t_{f}^{(p)}` is the final time for phase :math:`p`, :math:`q^{(p)}` is
the vector of integral values for phase :math:`p`, and :math:`s` is the vector
of parameter values for the problem. (Note that we use 0-based indexing throughout.)

For example, consider the :ref:`Goddard rocket problem </notebooks/goddard_problem_3_phase.ipynb>`
with :math:`P=3` phases. Each phase has three states: altitude :math:`h`, velocity
:math:`v`, and vehicle mass :math:`m`. Each phase has a single control, the thrust
:math:`T`. The second phase has a singular arc and a path constraint to enforce the
optimality condition along the singular arc. Additionally, there are eight discrete
constraints enforcing the continuity of the time and state variables at the phase boundaries.
The objective of the problem is to maximize the final altitude of the rocket, which is
the final altitude of the third phase. Given this description, the problem initialization
and objective callback function are:

.. code-block:: python

    from yapss import Problem

    problem = Problem(
        name="Goddard Rocket Problem with Singular Arc",
        nx=[3, 3, 3],
        nu=[1, 1, 1],
        nh=[0, 1, 0],
        nd=8,
    )

    def objective(arg):
        """Goddard Rocket Problem objective function."""
        # Maximize the final altitude of the third phase
        arg.objective = arg.phase[2].final_state[0]  # h_f

    problem.functions.objective = objective

The discrete variables that can be extracted from the ``arg`` object are:

- ``arg.phase[p].initial_state``: The initial state vector of phase :math:`p`.
- ``arg.phase[p].final_state``: The final state vector of phase :math:`p`.
- ``arg.phase[p].initial_time``: The initial time of phase :math:`p`.
- ``arg.phase[p].final_time``: The final time of phase :math:`p`.
- ``arg.phase[p].integral``: The integral vector of phase :math:`p`.
- ``arg.parameter``: The parameter vector of the problem.

All these attributes are immutable.

The value of the objective function is assigned to the ``arg.objective`` attribute.

In addition, the ``arg`` object has the attribute ``arg.auxdata``, which is a
:class:`SimpleNamespace` object that can be used to store any auxiliary data for the
problem.

Continuous Callback Function
----------------------------

The dynamics of the problem for each phase are given by:

.. math::

    \dot{x}^{(p)} = f^{(p)} ( x^{(p)}, u^{(p)},
    t, s ),\quad(p=0,\ldots,P-1)

subject to the path constraints:

.. math::

    h_{\min }^{(p)} \leq h^{(p)} ( x^{(p)}, u^{(p)},
    t, s ) \leq h_{\max }^{(p)}, \quad(p=0, \ldots, P-1)

In addition, each phase may have integrals associated with it:

.. math::

    q^{(p)}=\int_{t_{0}^{(p)}}^{t_{f}^{(p)}}g(x^{(p)},
    u^{(p)},t,s)\,dt,\quad(p=0,\ldots,P-1)

where :math:`g` is a vector-valued function. The bounds on the integrals are given by:

.. math::

    q_{\min}^{(p)}\leq q^{(p)}\leq q_{\max}^{(p)},\quad(p=0,\ldots,P-1)

The continuous callback function is used to evaluate the dynamics, path constraints, and integrand for each phase. The continuous callback function is called once for each phase in the problem. For example, for the three-phase Goddard rocket problem, the continuous callback function is:

.. code-block:: python

   def continuous(arg):
       """Goddard Rocket Problem dynamics and path functions."""
       auxdata = arg.auxdata
       sigma = auxdata.sigma
       h0 = auxdata.h0
       c = auxdata.c
       g0 = auxdata.g
       exp = np.exp

       for p in arg.phase_list:
           h, v, mass = arg.phase[p].state
           T, = arg.phase[p].control
           D = sigma * v**2.0 * exp(-h / h0)
           h_dot = v
           v_dot = (T - D) / mass - g0
           m_dot = -T / c
           arg.phase[p].dynamics[:] = (h_dot, v_dot, m_dot)

           if p == 1:
               arg.phase[p].path[:] = (mass * g0 - (1 + v / c) * D,)

The continuous callback function is called with a single argument, ``arg``, which is an
instance of the :class:`ContinuousArgument` class. The values that can be extracted
from the ``arg`` object are:

- ``arg.phase_list``: the phase indices listed as a *tuple*
- ``arg.phase[p].state``: the state vector for phase ``p``
- ``arg.phase[p].control``: the control vector for phase ``p``
- ``arg.phase[p].time``: the time variable for phase ``p``
- ``arg.parameter``: the parameter vector for the problem

In addition, the values of the dynamics, path constraints, and integrand are assigned to
the attributes:

- ``arg.phase[p].dynamics``
- ``arg.phase[p].path``
- ``arg.phase[p].integrand``

When setting the values of one of these attributes, each value must be a sequence of
length equal to the number of states, controls, or integrand variables, respectively. Each
element of the sequence must be a scalar, or an array-like object with the same shape as
the ``time`` attribute of the phase.`` The values of the dynamics, path constraints, and
integrand can also be set as slices of the corresponding attributes.

.. note::
    Always iterate over `arg.phase_list` instead of, say, `range(3)`. It’s essential to
    use this idiom, especially when the derivatives are determined using the
    "central-difference" differentiation method. The finite difference routines calculate
    the derivatives of the continuous functions one phase at a time, and failure to use
    this idiom may require significant extra computation.

Discrete Callback Function
--------------------------

The general expression for the discrete constraints of a problem with :math:`P` phases is:

.. math::

   \begin{aligned}
      d_{\text{min}}\le d\Big[
         & x^{(0)}(t_{0}^{(0)}),\ldots,x^{(P-1)}(t_{0}^{(P-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(P-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(P-1)}(t_{f}^{(P-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(P-1)},q^{(0)},\ldots,q^{(P-1)},s\Big]
            \le d_{\text{max}}
   \end{aligned}

For example, in the three-phase Goddard rocket problem, discrete constraints enforce the continuity of time and state variables at phase boundaries. The discrete function for this problem is:

.. math::

    \mathbf{0}\le d[\,\cdot\,]=\left[\begin{array}{c}
    t_{f}^{(0)}-t_{0}^{(1)}\\
    x^{(0)}(t_{f}^{(0)})-x^{(1)}(t_{0}^{(1)})\\
    t_{f}^{(1)}-t_{0}^{(2)}\\
    x^{(1)}(t_{f}^{(1)})-x^{(2)}(t_{0}^{(2)})
    \end{array}\right]\le\mathbf{0}

The corresponding discrete callback function for this problem is:

.. code-block:: python

    def discrete(arg):
        """Goddard Rocket Problem discrete constraint function."""
        phase = arg.phase

        # Discrete constraints enforce continuity between phases
        arg.discrete = [
            phase[0].final_time - phase[1].initial_time,  # Time continuity
            *(phase[0].final_state - phase[1].initial_state),  # State continuity
            phase[1].final_time - phase[2].initial_time,
            *(phase[1].final_state - phase[2].initial_state),
        ]

The discrete variables that can be extracted from the ``arg`` object are the same as those available in the objective callback function. The value of the discrete function must be assigned to the ``arg.discrete`` attribute, and it should be a one-dimensional array-like object with length equal to the number of discrete variables, as specified by the ``nd`` argument in the :class:`Problem` constructor.

Values in the ``arg.discrete`` attribute can also be set as slices. For instance, the example above can be rewritten as:

.. code-block:: python

    def discrete(arg):
        """Goddard Rocket Problem discrete constraint function."""
        phase = arg.phase
        arg.discrete[0] = phase[0].final_time - phase[1].initial_time
        arg.discrete[1:4] = phase[0].final_state - phase[1].initial_state
        arg.discrete[4] = phase[1].final_time - phase[2].initial_time
        arg.discrete[5:8] = phase[1].final_state - phase[2].initial_state

Mathematical Functions
----------------------

When defining callback functions for the optimal control problem, you will often need
mathematical functions like ``sin``, ``arctan2``, and ``log``. The data type for these
functions depends on the chosen differentiation method:

-  For **user-defined** and **central-difference** differentiation methods, data is
   passed as real NumPy arrays with elements of type ``np.float``.

-  For **automatic differentiation**, data is passed as NumPy object arrays, where each
   element is an encapsulated ``casadi.SX`` instance.

In most cases, Numpy `ufuncs` can be used to evaluate mathematical functions, without regard to the
data type. However, this fails for a few functions, either because there's no CasADi equivalent, or
because the function requires two arguments.

To handle this, the math functions used in the callback functions should be imported from ``yapps.math``
instead of directly from ``numpy``. Essentially, ``yapps.math``  is a drop-in replacement for
``numpy`` that works for all numpy objects, and correctly no matter the differentiation method.

Here’s a usage example for the ``arctan2`` function within a continuous callback function:

.. code-block:: python

    >>> from yapss.numpy import arctan2
    >>>
    >>> def continuous(arg):
    ...    x1, x2, x3 = arg.phase[0].state
    ...    x1_dot = arctan2(x2, x3)
    ...    # more code here

Available Functions
...................

Essentially all Numpy `ufuncs` that are likely to be used in callback functions are available in
``yapps.math``. Here are some of the most commonly used functions:

**Trigonometric functions**
    - ``cos``, ``sin``, ``tan``

**Inverse trigonometric functions**
    - ``arccos``, ``arcsin``, ``arctan``, ``arctan2``

**Hyperbolic functions**
    - ``cosh``, ``sinh``, ``tanh``

**Inverse hyperbolic functions**
    - ``arccosh``, ``arcsinh``, ``arctanh``

**Angular conversion**
    - ``degrees``, ``radians``, ``deg2rad``, ``rad2deg``

**Exponentials and logarithms**
    - ``exp``, ``exp2``, ``log``, ``log2``, ``log10``

**Comparison functions**
    - ``maximum``, ``minimum``

**Miscellaneous functions**
    - ``abs``, ``cbrt``, ``hypot``, ``power``, ``sign``, ``reciprocal``, ``square``, ``sqrt``
