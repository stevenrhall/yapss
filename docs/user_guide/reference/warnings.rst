Warnings and Errors
===================

YAPSS raises when what you supplied is wrong, and warns when it is valid but deserves your
attention --- an unconverged solve, an Ipopt option your build does not provide, a very
large mesh segment. `When YAPSS raises and when it warns`_ below draws the line.
Every warning points at a line in *your* code --- where you set the value, or where you
called ``solve()`` --- which is what makes the categories below worth knowing: a filter
written as ``module="yapss"`` matches none of them, because the module recorded is yours.

That is a promise you can check. If a YAPSS warning points at a line inside YAPSS rather
than at your own code, or an error's message leaves you unable to tell which of your lines
or settings caused it, that is a bug in YAPSS --- in the check itself or in its message ---
and a `bug report <https://github.com/stevenrhall/yapss/issues>`_ would help.

The hierarchy
-------------

.. code-block:: text

    UserWarning
     └── yapss.YapssWarning
          ├── yapss.IpoptConvergenceWarning      Ipopt did not report a converged solution
          ├── yapss.IpoptOptionSettingWarning    Ipopt refused an option
          ├── yapss.LargeSegmentWarning          a mesh segment has very many points
          └── yapss.YapssDeprecationWarning      (also a FutureWarning)

    Exception
     └── yapss.YapssError
          └── yapss.UnsupportedMathFunctionError  (also a TypeError)

``YapssDeprecationWarning`` subclasses :class:`FutureWarning` rather than
:class:`DeprecationWarning`, which Python shows only in ``__main__``: an optimal control
problem is normally solved from a script or a notebook, where a deprecation notice would be
hidden.

Errors raised for ordinary bad input --- a bound that is not a number, a mesh with too few
collocation points, a misspelled attribute --- are the plain built-in exceptions
(:class:`ValueError`, :class:`TypeError`, :class:`AttributeError`, :class:`IndexError`), since
that is what Python itself would raise.

When YAPSS raises and when it warns
-----------------------------------

YAPSS **raises** when what you supplied breaks its contract: when it cannot produce a correct
answer from it, or when it is almost certainly a mistake even though a solve could proceed. A
callback output left unassigned is an example: the row has no value, and a forgotten line
is far more likely than an intended zero, so the solve does not start and the error names
the row. Where zero is what you mean, say so explicitly --- assign ``0.0`` to the row.

YAPSS **warns** only when what you supplied is valid but something deserves your attention:
the outcome of the solve (Ipopt did not converge), the environment (an Ipopt option your build
does not provide, an environment variable that no longer has an effect), a choice with a cost
(a very large mesh segment), or a notice that a behavior will change.

Unconverged solves
------------------

When Ipopt stops at an iterate without converging, YAPSS returns the ``Solution`` and emits an
:class:`~yapss.IpoptConvergenceWarning`, so the outcome is at least not silent. For example,
forcing Ipopt to stop immediately by setting ``max_iter = 0`` reliably produces an unconverged
solve, and the warning appears as soon as ``solve()`` returns:

.. testsetup:: unconverged

   import warnings
   from yapss.examples.brachistochrone import setup

   # doctest reads standard output; show each warning there, as the example displays it
   saved_filters, saved_showwarning = warnings.filters[:], warnings.showwarning
   warnings.simplefilter("always")
   warnings.showwarning = lambda message, category, *args, **kwargs: print(
       f"{category.__name__}: {message}"
   )
   problem = setup()
   problem.ipopt_options.print_level = 0

.. testcleanup:: unconverged

   warnings.filters[:], warnings.showwarning = saved_filters, saved_showwarning

.. doctest:: unconverged
   :options: +NORMALIZE_WHITESPACE

   >>> problem.ipopt_options.max_iter = 0
   >>> solution = problem.solve()
   IpoptConvergenceWarning: Ipopt did not converge. Status -1: "Maximum Number of
   Iterations Exceeded." The returned solution does not satisfy Ipopt's convergence
   criteria and should not be treated as an optimal trajectory. Check solution.status
   and the Ipopt output before using these results.

Three Ipopt statuses are treated as success and do not warn: ``0`` (optimal solution
found), ``1`` (solved to acceptable level), and ``6`` (feasible point found for a square
problem). Status ``1`` is included deliberately. It is a normal outcome when tolerances are
pushed hard --- the answer is routinely correct to far more digits than requested --- and
warning on it would train users to disregard the warning, which would destroy its value for
the cases that matter.

What the warning does not tell you
..................................

**A cancelled solve carries no guarantee at all.** While ``problem.catch_keyboard_interrupt``
is ``True``, the default, interrupting with Ctrl-C stops Ipopt at whatever iterate it had
reached, reported as status ``5``. The warning tells you it did not converge, and that is the
only thing it tells you. With ``False``, Ctrl-C raises ``KeyboardInterrupt`` and there is no
solution (see :class:`~yapss.Problem`).

**Branch on the status, not on the warning.** Warnings are for people reading output.
Code that needs to know should test ``solution.converged``, or ``solution.status`` to tell
the failure modes apart. Both are described in :doc:`solution`, and neither is affected by
warning filters.

Solves that stop without a solution
-----------------------------------

A solve returns a ``Solution`` only when Ipopt has an iterate to report. For the statuses where
it has none, ``solve()`` raises instead of returning a solution made of placeholder values:
``ValueError`` for too few degrees of freedom (``-10``), inconsistent bounds (``-11``), an
invalid option (``-12``), or a NaN or Inf returned by a callback or its derivative during the
solve (``-13``); ``MemoryError`` when Ipopt runs out of memory (``-102``); and ``RuntimeError``
for a failure inside Ipopt itself, or for a status this version of YAPSS does not recognize.
With status ``-13``, Ipopt reports the point it had reached but sets every constraint value and
multiplier to zero, so a solution built from it would report zero costates and multipliers that
Ipopt never computed. :doc:`ipopt_backend` explains why these statuses leave nothing to report.

Filtering
---------

To turn every YAPSS warning into an error, which is the strictest way to run a script::

    import warnings
    import yapss

    warnings.simplefilter("error", yapss.YapssWarning)

Or one category at a time::

    warnings.filterwarnings("error", category=yapss.IpoptConvergenceWarning)
    warnings.filterwarnings("ignore", category=yapss.LargeSegmentWarning)

In a test suite, pytest accepts the fully qualified name, on the command line or in its
``filterwarnings`` setting, so an unconverged solve fails the test::

    pytest -W error::yapss.IpoptConvergenceWarning

Python's own ``-W`` option cannot name a YAPSS category, since it is read before any
package can be imported; ``python -W error`` turns every warning into an error, YAPSS's
included.

Every unconverged solve warns, including several run from the same line: Python's ``"default"``
and ``"once"`` actions would otherwise report only the first, and a solve that quietly returns
a non-optimal trajectory is exactly what the warning exists to prevent. ``"ignore"`` and
``"error"`` behave as they do for any other warning.

The vendored Ipopt interface (``yapss._backend.mseipopt``) is written to stand on its own and
keeps its own categories, such as ``IpoptVerificationWarning``; they are not part of this
hierarchy.

Reference
---------

.. autoexception:: yapss.YapssWarning
.. autoexception:: yapss.YapssDeprecationWarning
.. autoexception:: yapss.YapssError
.. autoexception:: yapss.LargeSegmentWarning
.. autoexception:: yapss.IpoptConvergenceWarning

The other categories are documented where the behavior they report is described:
:class:`~yapss.IpoptOptionSettingWarning` in :doc:`ipopt_options`, and
:class:`~yapss.UnsupportedMathFunctionError` in :doc:`callbacks` (defined in ``yapss.math``
and re-exported from ``yapss``).
