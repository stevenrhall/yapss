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

The general expression for an objective function for a problem with :math:`n_{p}` phases is

.. math::

   \begin{aligned}
      J=\phi\Big[
         & x^{(0)}(t_{0}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{0}^{(n_{p}-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(n_{p}-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{f}^{(n_{p}-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(n_{p}-1)},q^{(0)},\ldots,q^{(n_{p}-1)},s\Big]
   \end{aligned}

where :math:`\phi` is a scalar function, :math:`x^{(p)}(t)` is the state vector
for phase :math:`p`, :math:`t_{0}^{(p)}` is the initial time for phase :math:`p`,
:math:`t_{f}^{(p)}` is the final time for phase :math:`p`, :math:`q^{(p)}` is
the vector of integral values for phase :math:`p`, and :math:`s` is the vector
of parameter values for the problem. (Note that we use 0-based indexing throughout.)

For example, consider the :ref:`Goddard rocket problem </notebooks/goddard_problem_3_phase.ipynb>`
with :math:`n_{p}=3` phases. Each phase has three states: altitude :math:`h`, velocity
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
    problem.sense = "maximize"

    def objective(arg):
        """Goddard Rocket Problem objective function."""
        # h_f, the final altitude of the third phase, is the quantity being
        # maximized; `problem.sense = "maximize"` above is what makes it so.
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
:class:`~types.SimpleNamespace` object that can be used to store any auxiliary data for the
problem.

Minimizing vs. Maximizing
..........................

.. versionadded:: 0.2.0

    The ``problem.sense`` attribute.

By default, YAPSS minimizes ``arg.objective``. To maximize it instead, set

.. code-block:: python

    problem.sense = "maximize"

and write ``arg.objective`` as the quantity you actually want maximized, unnegated, as
in the Goddard rocket example above. ``problem.sense`` accepts ``"minimize"`` (the
default) or ``"maximize"``; any other value raises ``ValueError``.

In previous versions of YAPSS, users were advised instead to either negate the
objective in the objective callback function, or to set ``problem.scale.objective``
to ``-1`` (or generally, to a negative number). Neither is recommended now, for
different reasons. Negating the objective yourself gives the right primal solution,
but the dual solution -- the Lagrange multipliers and phase costates returned as part
of the :doc:`solution` object -- comes back with the wrong sign, and
``solution.objective`` itself reports the negative of the objective you actually
care about, since it is simply whatever the callback returned. Setting
``problem.scale.objective`` to a negative number avoids both of those problems --
Ipopt reports the correct primal solution, correctly-signed multipliers and costates
(:math:`\mu = dJ/dc`), and the correct objective value -- but it conflates the sign
of the objective with a number that is otherwise purely about conditioning the
problem for Ipopt (see :doc:`scaling`), which is confusing to read and impossible to
validate.

``problem.sense`` resolves both problems. It flips the sign the way Ipopt itself
expects, through its own scaling mechanism, so the primal solution, multipliers, and
costates all come back correct regardless of which sense is chosen, while
``problem.scale.objective`` keeps a single, unambiguous role. Since ``problem.sense``
is now the only way to flip the objective's sign, ``problem.scale.objective`` must be
strictly positive; setting it to zero or a negative number raises ``ValueError``.

Continuous Callback Function
----------------------------

The dynamics of the problem for each phase are given by:

.. math::

    \dot{x}^{(p)} = f^{(p)} ( x^{(p)}, u^{(p)},
    t, s ),\quad(p=0,\ldots,n_{p}-1)

subject to the path constraints:

.. math::

    h_{\min }^{(p)} \leq h^{(p)} ( x^{(p)}, u^{(p)},
    t, s ) \leq h_{\max }^{(p)}, \quad(p=0, \ldots, n_{p}-1)

In addition, each phase may have integrals associated with it:

.. math::

    q^{(p)}=\int_{t_{0}^{(p)}}^{t_{f}^{(p)}}g(x^{(p)},
    u^{(p)},t,s)\,dt,\quad(p=0,\ldots,n_{p}-1)

where :math:`g` is a vector-valued function. The bounds on the integrals are given by:

.. math::

    q_{\min}^{(p)}\leq q^{(p)}\leq q_{\max}^{(p)},\quad(p=0,\ldots,n_{p}-1)

The continuous callback function is used to evaluate the dynamics, path constraints, and integrand for each phase. Each invocation receives the phase indices in ``arg.phase_list``; normally this includes all phases, but derivative calculations may select a subset. For example, for the three-phase Goddard rocket problem, the continuous callback function is:

.. code-block:: python

   from yapss.math import exp

   def continuous(arg):
       """Goddard Rocket Problem dynamics and path functions."""
       auxdata = arg.auxdata
       sigma = auxdata.sigma
       h0 = auxdata.h0
       c = auxdata.c
       g0 = auxdata.g

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
instance of the :class:`ContinuousArg <yapss._private.input_args.ContinuousArg>` class.
The values that can be extracted from the ``arg`` object are:

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

When setting the values of one of these attributes, each value must be a sequence whose
length equals the number of states, path constraints, or integrals, respectively. Each
element of the sequence must be a scalar, or an array-like object with the same shape as
the ``time`` attribute of the phase. The values of the dynamics, path constraints, and
integrand can also be set as slices of the corresponding attributes.

.. note::
    Always iterate over `arg.phase_list` instead of, say, `range(3)`. It’s essential to
    use this idiom, especially when the derivatives are determined using the
    "central-difference" or "central-difference-full" differentiation methods. The finite
    difference routines calculate the derivatives of the continuous functions one phase at
    a time, and failure to use this idiom may require significant extra computation.

Discrete Callback Function
--------------------------

The general expression for the discrete constraints of a problem with :math:`n_{p}` phases is:

.. math::

   \begin{aligned}
      d_{\text{min}}\le d\Big[
         & x^{(0)}(t_{0}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{0}^{(n_{p}-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(n_{p}-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{f}^{(n_{p}-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(n_{p}-1)},q^{(0)},\ldots,q^{(n_{p}-1)},s\Big]
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

The discrete variables that can be extracted from the ``arg`` object are the same as those available in the objective callback function. The value of the discrete function must be assigned to the ``arg.discrete`` attribute, and it should be a one-dimensional array-like object with length equal to the number of discrete variables, as specified by the ``nd`` argument in the :class:`~yapss.Problem` constructor.

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

-  For **user-defined**, **central-difference**, and **central-difference-full**
   differentiation methods, data is passed as real NumPy arrays with elements of type
   ``np.float64``.

-  For **automatic differentiation**, data is passed as NumPy object arrays, where each
   element is an encapsulated ``casadi.SX`` instance.

In most cases, NumPy `ufuncs` can be used to evaluate mathematical functions, without regard to the
data type. However, this fails for a few functions, either because there's no CasADi equivalent, or
because the function requires two arguments.

To handle this, the math functions used in the callback functions should be imported from ``yapss.math``
instead of directly from ``numpy``. Essentially, ``yapss.math`` is a drop-in replacement for
``numpy`` that works for all NumPy objects, regardless of the differentiation method.

If a callback only needs one or two math functions, import them by name, as in the
``arctan2`` example below. If a callback uses many math functions, or you would simply like
existing NumPy-style code to keep working reliably under both differentiation methods without
renaming every call site, import the whole module instead:

.. code-block:: python

    import yapss.math as np

and use it exactly as you would ``numpy``, e.g. ``np.exp(...)``, ``np.sin(...)``. This mirrors
the ``import numpy as np`` convention on purpose, so ordinary-looking NumPy code keeps working
under automatic differentiation, not just under central differences.

.. note::
    ``yapss.math`` imported ``as np`` is unrelated to the ``Problem.np`` attribute (the
    number of phases in a problem). Same three letters, different object -- easy to
    conflate at a glance if you're skimming.

Here’s a usage example for the ``arctan2`` function within a continuous callback function:

>>> from yapss.math import arctan2
>>>
>>> def continuous(arg):
...     x1, x2, x3 = arg.phase[0].state
...     x1_dot = arctan2(x2, x3)
...     # more code here

The Rule ``yapss.math`` Enforces
................................

Every function ``yapss.math`` provides must give the **same result under every differentiation
method**. A function that quietly computed something different under automatic differentiation
than under central differences would mean the solver was handed a different optimal control
problem depending on a setting that is supposed to affect only how derivatives are obtained --
and nothing would raise.

So each NumPy `ufunc` falls into exactly one of three categories:

**Supported**
    Listed below. Checked against NumPy on both paths by the package's test suite.

**Unsupported**
    Raises ``yapss.math.UnsupportedMathFunctionError`` on a symbolic value, and warns on a real
    one until 0.3.0. See `Unsupported Functions`_.

**Not applicable**
    Array-level functions (``sum``, ``clip``, ``matmul``), integer-domain functions (``gcd``,
    ``left_shift``), and multiple-output functions (``frexp``, ``modf``). These are not
    elementwise scalar operations, so the question does not arise.

Note that "supported" is not the same as "smooth". ``abs``, ``sign``, ``floor``, ``maximum``,
and the comparisons are all supported and all non-differentiable somewhere. They are legitimate
modelling tools, but see the caution in `Comparisons and Gated Expressions`_.

Available Functions
...................

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
    - ``exp``, ``exp2``, ``expm1``, ``log``, ``log1p``, ``log2``, ``log10``, ``logaddexp``,
      ``logaddexp2``

**Powers and roots**
    - ``cbrt``, ``float_power``, ``hypot``, ``pow``, ``power``, ``reciprocal``, ``sqrt``,
      ``square``

**Arithmetic**
    - ``add``, ``subtract``, ``multiply``, ``divide``, ``true_divide``

**Modular arithmetic**
    - ``mod``, ``remainder``, ``fmod``, ``floor_divide``

    ``mod`` and ``remainder`` take the sign of the divisor; ``fmod`` truncates and takes the sign
    of the dividend. This follows NumPy, and the two conventions differ whenever the operands
    have opposite signs.

**Sign and magnitude**
    - ``abs``, ``absolute``, ``fabs``, ``copysign``, ``negative``, ``positive``, ``sign``,
      ``conj``, ``conjugate``

    ``copysign`` differs from NumPy at ``y == -0.0``: NumPy reads the floating-point sign bit,
    which a symbolic expression cannot represent, so negative zero is treated as positive.

**Rounding**
    - ``ceil``, ``floor``, ``trunc``

**Extrema**
    - ``maximum``, ``minimum``, ``fmax``, ``fmin``

**Comparison functions**
    - ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``, ``greater_equal``

**Logical functions**
    - ``logical_and``, ``logical_or``, ``logical_not``, ``logical_xor``

**Step function**
    - ``heaviside``

Comparisons and Gated Expressions
.................................

Comparisons are supported, and so are the corresponding **operators**. This matters because a
gate is normally written with operators rather than function calls:

>>> import numpy as np
>>> from yapss.math import abs as yabs
>>> y = np.array([-2.0, -0.5, 0.5, 2.0])
>>> (yabs(y) <= 1) * 3.0
array([0., 3., 3., 0.])

The idiom above -- multiplying an expression by a comparison -- lets a path constraint apply over
a finite interval instead of the whole phase. A constraint that should hold only while a state
lies in some range can be written:

.. code-block:: python

    def continuous(arg):
        x, y, v = arg.phase[0].state
        (u,) = arg.phase[0].control
        arg.phase[0].dynamics[:] = ...
        # constrain v only where |y| <= 1; elsewhere the row evaluates to 0
        arg.phase[0].path[:] = [(yabs(y) <= 1) * v]

Because Python cannot overload ``and``, ``or``, and ``not``, combine masks with ``&``, ``|``, and
``~`` instead:

.. code-block:: python

    inside = (y >= -1) & (y <= 1)
    outside = ~inside

.. caution::
    A gate is not differentiable at its edges, and the two differentiation methods do not
    disagree about the gate's *value* but do disagree about its *derivative* there. CasADi
    differentiates a comparison to zero almost everywhere, while central differencing straddles
    the discontinuity and sees a large finite difference.

    In practice this means Ipopt may behave differently under the two methods on a gated
    problem, and may converge to different points. A gate often works well near a solution and
    can cause trouble far from one. If a gate is giving the solver difficulty, a smooth
    approximation -- a ``tanh`` gate with a sharpness parameter, tightened over successive
    solves -- trades the kink for a well-behaved gradient.

Unsupported Functions
.....................

A few NumPy `ufuncs` inspect the floating-point representation of a number rather than its value,
and have no symbolic equivalent. YAPSS declines to support these:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Reason
   * - ``nextafter``
     - Steps between adjacent floating-point values.
   * - ``rint``
     - Rounds half to even, which CasADi cannot reproduce. ``floor(x + 0.5)`` rounds half away
       from zero and would silently disagree with NumPy.
   * - ``signbit``
     - Reads the floating-point sign bit, including the sign of negative zero.
   * - ``spacing``
     - Returns the distance to the adjacent floating-point value.

Given a symbolic value -- that is, under the ``"auto"`` derivative method -- they raise
``yapss.math.UnsupportedMathFunctionError``, a subclass of the built-in ``TypeError``, which is
what NumPy itself raised for them before YAPSS 0.2.2:

>>> import casadi as ca
>>> from yapss.math import spacing
>>> from yapss.math.wrapper import SXW
>>> spacing(SXW(ca.SX.sym("x")))
Traceback (most recent call last):
    ...
yapss.math.functions.UnsupportedMathFunctionError: 'spacing' is not supported in YAPSS callback functions...

Given a real value, they emit ``yapss.math.UnsupportedMathFunctionWarning`` and evaluate as NumPy
does:

>>> import warnings
>>> with warnings.catch_warnings(record=True) as caught:
...     warnings.simplefilter("always")
...     spacing(1.0)
...     print(caught[0].category.__name__)
np.float64(2.220446049250313e-16)
UnsupportedMathFunctionWarning

.. deprecated:: 0.2.2
    Calling one of these on a real value will raise ``UnsupportedMathFunctionError`` in 0.3.0,
    as a symbolic value already does.

Rejecting on both paths is the goal. If one of these works under central differences and fails
under automatic differentiation, a formulation can come to depend on the differentiation method,
which is exactly what ``yapss.math`` exists to prevent. The real path is only still open because
it worked through 0.2.1, and a patch release does not take away working code.

If you have a reason to use one anyway, call it through ``numpy`` directly -- ``yapss.math``
declines to offer it, but does not prevent it.

``ContinuousArg`` Class Reference
----------------------------------

.. autoclass:: yapss._private.input_args.ContinuousArg
   :members:
   :no-special-members:
   :no-undoc-members:
