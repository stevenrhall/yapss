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

These are inputs, and they are read-only: a write into one raises ``ValueError`` at the
line, and assigning to the name itself raises ``AttributeError``, rather than being
silently swallowed by a copy or, for ``arg.parameter``, reaching the solver's own array.

.. versionchanged:: 0.3.0

    Callback inputs are read-only. Before, writing into ``arg.phase[p].initial_state`` and
    the other endpoint values changed a copy and had no effect, while writing into
    ``arg.parameter`` changed the solver's decision vector.

The value of the objective function is assigned to the ``arg.objective`` attribute. It must be a
scalar: assigning an array, even one with a single element, raises ``TypeError`` at that line.

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
but the dual solution --- the Lagrange multipliers and phase costates returned as part
of the :doc:`solution` object --- comes back with the wrong sign, and
``solution.objective`` itself reports the negative of the objective you actually
care about, since it is simply whatever the callback returned. Setting
``problem.scale.objective`` to a negative number avoids both of those problems --
Ipopt reports the correct primal solution, correctly-signed multipliers and costates
(:math:`\mu = dJ/dc`), and the correct objective value --- but it conflates the sign
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

The ``exp`` above is imported from ``yapss.math``. Import the math functions a callback uses
from there, and the callback works under every derivative method; see :doc:`math`.

The continuous callback function is called with a single argument, ``arg``, which is an
instance of the :class:`ContinuousArg <yapss._private.input_args.ContinuousArg>` class.
The values that can be extracted from the ``arg`` object are:

- ``arg.phase_list``: the phase indices listed as a *tuple*
- ``arg.phase[p].state``: the state vector for phase ``p``
- ``arg.phase[p].control``: the control vector for phase ``p``
- ``arg.phase[p].time``: the time variable for phase ``p``
- ``arg.parameter``: the parameter vector for the problem

These are read-only as well, both the arrays and the ``state`` and ``control`` containers
that hold them; each call reads the values of the point it was called at.

In addition, the values of the dynamics, path constraints, and integrand are assigned to
the attributes:

- ``arg.phase[p].dynamics``
- ``arg.phase[p].path``
- ``arg.phase[p].integrand``

Each of these outputs is a stack of rows, one for each state, path constraint, or integral,
with a value at every point in the ``time`` attribute of the phase. The outputs start every
call at zero --- nothing a previous call assigned is carried over --- and are assigned
**one whole row at a time**:

.. code-block:: python

    dynamics = arg.phase[p].dynamics
    dynamics[:] = (h_dot, v_dot, m_dot)   # every row: one value per row
    dynamics[0:2] = (h_dot, v_dot)        # some rows, by a slice (any step)
    dynamics[2] = m_dot                   # one row; dynamics[-1] is the last row
    dynamics[2] += drag_term              # in-place operators assign the whole row
    arg.phase[p].path[:] = 0.0            # a constant may fill several rows

A row value is a scalar constant, an expression over the points (such as ``v * cos(u)``),
or a list with one value per point. A row computed point by point is written as a list,
not by assigning its elements:

.. code-block:: python

    arg.phase[p].dynamics[0] = [f(x[k]) for k in range(len(x))]

Anything else raises at the line: assigning an element or part of a row
(``dynamics[0][k] = ...``), a column (``dynamics[:, k] = ...``), a value whose shape fits
only by broadcasting (one expression over the points into several rows, or a length-1 array
over several points), a row that does not exist, or writing into an output with
``np.copyto`` or a function's ``out=`` argument. The array itself is read-only.

.. versionchanged:: 0.3.0

    Outputs are assigned by whole rows only; element, partial-row, and column writes, and
    shapes that fit only by broadcasting, now raise.

A callback assigns its results and returns nothing. Returning a value instead raises
``TypeError`` naming the callback and the assignment to make, since YAPSS never looks at
what a callback returns and the results would otherwise be silently zero.

.. versionchanged:: 0.3.0

    Outputs are cleared before every call, and a callback that returns a value raises.

Each output at a point may depend only on the inputs **at that point** --- the time, states,
and controls there --- and on the parameters. A callback that reaches across points, with
``t[0]``, ``len``, ``mean``, ``sum``, ``cumsum``, ``diff``, or indexing by position, is not a
valid continuous function: the ``"auto"`` method evaluates the callback at a single point,
while the numeric methods pass every point at once, so the same callback then describes
different problems under different methods, and the sparsity probe of
``"central-difference"`` assumes each point is independent as well.

YAPSS checks this before the solve starts, by evaluating the continuous callback a second
time on every point of each phase but the last, in reverse order, and comparing. A callback
whose outputs change raises ``ValueError`` naming the output and the point. The check is
useful rather than complete: it can only see what the initial guess reveals --- nothing about
an input that is constant along the guess --- and it accepts differences below a relative
tolerance of 1e-7, since reordering can change floating-point rounding.

The same setup check raises ``ValueError`` when a callback never assigns an output row at the
initial guess, naming the callback, the line of its ``def``, and every row it left unassigned.
An unassigned row is zero, which almost always means a missing line; if zero is intended,
assign ``0.0`` explicitly. The check also raises ``ValueError`` if an output is NaN or infinite
there, naming the output and the number of points.

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

Each discrete constraint is one row of ``arg.discrete``, with a single value, and the same
whole-row rule applies: assign every value at once, one value, or a slice with one value per
constraint; in-place operators such as ``arg.discrete[0] += c`` work too. For instance, the
example above can be rewritten as:

.. code-block:: python

    def discrete(arg):
        """Goddard Rocket Problem discrete constraint function."""
        phase = arg.phase
        arg.discrete[0] = phase[0].final_time - phase[1].initial_time
        arg.discrete[1:4] = phase[0].final_state - phase[1].initial_state
        arg.discrete[4] = phase[1].final_time - phase[2].initial_time
        arg.discrete[5:8] = phase[1].final_state - phase[2].initial_state

Errors Raised in a Callback
---------------------------

An exception raised inside one of your callbacks reaches you unchanged, with its own message
and traceback, and with a note naming the callback YAPSS was calling and the line of its
``def``, since the traceback alone does not say which of your functions was running:

.. code-block:: pycon

   ValueError: math domain error
   Raised in functions.continuous = continuous (my_problem.py, line 12).

Under the ``"auto"`` derivative method each callback is first called with symbolic inputs. A
function that needs a float, such as ``math.sin``, fails there with a ``TypeError``, and a second
note says to use the functions of ``yapss.math`` instead, which :doc:`math` describes.

Continuous Argument Class Reference
-----------------------------------

Each of the three continuous callbacks receives its own argument class. All three carry the
same inputs; each carries only the output its own callback assigns, so a Jacobian callback
has no ``dynamics`` to write and the continuous callback has no ``jacobian``. Writing another
callback's output raises ``AttributeError`` at the line.

.. versionchanged:: 0.3.0

    The three continuous callbacks used to share one argument carrying every output. A
    derivative entry assigned from the continuous callback was silently ignored, and a
    dynamics row assigned from the Jacobian callback overwrote the constraint values the
    solver had already computed at that point, giving a wrong answer with no error.

.. autoclass:: yapss._private.input_args.ContinuousArg
   :members:
   :no-special-members:
   :no-undoc-members:

.. autoclass:: yapss._private.input_args.ContinuousJacobianArg
   :members:
   :no-special-members:
   :no-undoc-members:

.. autoclass:: yapss._private.input_args.ContinuousHessianArg
   :members:
   :no-special-members:
   :no-undoc-members:
