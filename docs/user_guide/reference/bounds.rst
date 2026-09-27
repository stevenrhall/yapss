Setting Bounds
==============

A bound is set on the quantity it bounds, as a ``(lower, upper)`` pair. Each side is a number, or
``None`` for no bound on that side:

.. doctest:: bounds

    >>> from yapss.examples.brachistochrone_minimal import setup
    >>> problem = setup()
    >>> ph = problem.phases.phase
    >>> ph.state.v.bounds = (0.0, 10.0)      # an interval
    >>> ph.state.y.bounds = (0.0, None)      # a floor, with no ceiling
    >>> ph.state.x.initial = (0.0, 0.0)      # fixed: an interval whose ends agree
    >>> ph.state.x.bounds
    (0.0, 10.0)

A fixed value is written as a pair whose ends agree; a bare number is not a bound. A field has
rows, so a number standing for a bound would be indistinguishable from a row count, and the pair
is what removes the doubt. An infinity on its own side means the same as ``None``.

Everything is unbounded until it is bounded, except the constraints: a declared path or discrete
constraint must be given a bound, since an unbounded constraint constrains nothing.

Where bounds are set
--------------------

For a phase ``ph`` of a problem ``problem``:

-   ``ph.time.initial`` and ``ph.time.final``: the phase's initial and final times. A phase that
    names its independent variable otherwise uses that name: ``ph.r.initial`` for a variable
    ``r``.
-   ``ph.state.x.bounds``: the state ``x`` throughout the phase; ``ph.state.x.initial`` and
    ``ph.state.x.final``: its value at the phase's two ends.
-   ``ph.control.u.bounds``: the control ``u``.
-   ``ph.path.g.bounds``: the path constraint ``g``, which must be bounded.
-   ``ph.integral.q.bounds``: the integral ``q``.
-   ``problem.parameter.p.bounds``: the parameter ``p``.
-   ``problem.discrete.d.bounds``: the discrete constraint ``d``, which must be bounded.

A phase's duration is at least zero; it has no bound of its own.

A field declared with ``yapss.vector(n)`` has one bound per row, set by row:
``ph.state.r.bounds[:] = (-1e7, 1e7)`` gives every row the same bound,
``ph.state.r.bounds[:] = [(0, 1), (2, 3), (4, 5)]`` one each, and
``ph.state.r.bounds[0] = (0, 10)`` one row.

Example
-------

The `dynamic soaring problem <../notebooks/dynamic_soaring.ipynb>`_ has six states, two controls,
one path constraint, three discrete constraints, and a single parameter for the wind shear rate.
The circuit starts and ends at the origin, and the load factor is limited; the discrete
constraints make the velocity, flight path angle and heading periodic, the heading after one full
turn. The altitude's lower bound is the ground, and the solution comes down to it; the other state
bounds are loose and inactive in the solution, there because loose box bounds on the variables
can help Ipopt converge.

.. literalinclude:: ../../../src/yapss/examples/dynamic_soaring.py
   :language: python
   :start-after: # ------------------------------------------------------------------- setup
   :end-before: # A circuit that is roughly
   :dedent: 4

What is checked, and when
-------------------------

A bound that is wrong on its own raises where it is written, naming the field: a value that is
not a pair, a side that is not a number or ``None``, a NaN, ``+inf`` as a lower bound or ``-inf``
as an upper bound, a lower bound above the upper, or the wrong number of bounds for a block
field's rows. A refused write leaves the bound as it was.

.. doctest:: bounds
    :options: +NORMALIZE_WHITESPACE

    >>> ph.state.x.bounds = 5.0
    Traceback (most recent call last):
        ...
    TypeError: phase 'phase' state bounds 'x': a bound is a pair, and 5.0 is one number. To fix
    the value, write (5.0, 5.0); for an interval, write its two ends.

Bounds that are each valid but contradict one another depend on more than one assignment, so
they are reported by ``problem.validate()``, which ``problem.solve()`` runs before Ipopt starts:
a state's initial or final bound that does not overlap its bound, a final time bound that lies
wholly before the initial one, and a path or discrete constraint left unbounded. Because nothing
is refused until then, bounds can be set in any order.

.. doctest:: bounds
    :options: +NORMALIZE_WHITESPACE

    >>> ph.state.v.initial = (20.0, 20.0)
    >>> problem.validate()
    Traceback (most recent call last):
        ...
    ValueError: the problem is not ready to solve:
      phase 'phase' state 'v': its initial bound (20.0, 20.0) does not overlap its bound
      (0.0, 10.0), so no initial value satisfies both

Special Considerations for State Bounds
---------------------------------------

.. note::

    This section applies only to problems where state bounds act as path constraints, and
    where accurate Lagrange multipliers are required.

Broadly speaking, a bound on a state, such as

.. code-block:: python

    ph.state.v.bounds = (-1000.0, 1000.0)

might be used in one of two ways:

1.  As an inactive bound, to aid solver convergence without constraining the final solution.
2.  As a path constraint, where the bound is expected to be active in the final solution.

Where a state bound is meant to be a path constraint, declare it as one instead, and fill it in
the continuous callback:

.. code-block:: python

    class Path(yapss.Path):
        speed = yapss.scalar()

    ...

    @ph.register.continuous
    def continuous(arg, out):
        ...
        out.path.speed = arg.state.v

    ph.path.speed.bounds = (-1000.0, 1000.0)

While both approaches yield the correct primal solution (decision variables), state bounds
applied directly as path constraints may lead to incorrect Lagrange multipliers ---
particularly for initial and final states. In order to obtain accurate numerical results,
path constraints should be applied and Lagrange multipliers calculated only at collocation
points, not all interpolation points, to be consistent with the pseudospectral integration
scheme. In addition, without additional logic, it's difficult to determine whether the
Lagrange multipliers returned by the NLP solver should be associated with the endpoint
state constraints or the state path constraint.

For these reasons, state bounds expected to be active should be implemented as true path
constraints, as shown in the example above. Note that these considerations do *not* apply
to constraints on control variables.
