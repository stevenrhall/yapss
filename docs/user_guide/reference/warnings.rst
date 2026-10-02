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
          ├── yapss.LargeSegmentWarning          a mesh segment has more than 20 points
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
callback output left unassigned is an example: the row would be zero, which almost always
means a missing line, so the solve does not start. Where a value that looks like a mistake is
what you mean, say so explicitly --- assign ``0.0`` to the row.

YAPSS **warns** only when what you supplied is valid but something deserves your attention:
the outcome of the solve (Ipopt did not converge), the environment (an Ipopt option your build
does not provide, an environment variable that no longer has an effect), a choice with a cost
(a very large mesh segment), or a notice that a behavior will change.

Solves that stop without a solution
-----------------------------------

A solve returns a ``Solution`` only when Ipopt has an iterate to report. For the statuses where
it has none, ``solve()`` raises instead of returning a solution made of placeholder values:
``ValueError`` for too few degrees of freedom (``-10``), inconsistent bounds (``-11``), an
invalid option (``-12``), or a NaN or Inf returned by a callback or its derivative during the
solve (``-13``); ``MemoryError`` when Ipopt runs out of memory (``-102``); and ``RuntimeError``
for a failure inside Ipopt itself, or for a status this version of YAPSS does not recognize.
:doc:`solution` describes the statuses that do return a solution, and :doc:`ipopt_backend`
explains why these leave nothing to report.

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

The vendored Ipopt interface (``yapss._private.mseipopt``) is written to stand on its own and
keeps its own categories, such as ``IpoptVerificationWarning``; they are not part of this
hierarchy.

Reference
---------

.. autoexception:: yapss.YapssWarning
.. autoexception:: yapss.YapssDeprecationWarning
.. autoexception:: yapss.YapssError
.. autoexception:: yapss.LargeSegmentWarning

The other categories are documented where the behavior they report is described:
:class:`~yapss.IpoptConvergenceWarning` in :doc:`solution`,
:class:`~yapss.IpoptOptionSettingWarning` in :doc:`ipopt_options`, and
``UnsupportedMathFunctionError`` in :doc:`math` (defined in ``yapss.math`` and
re-exported from ``yapss``).
