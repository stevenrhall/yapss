Warnings and Errors
===================

YAPSS warns rather than raises whenever a run can still produce a result the user may want to
look at --- an unconverged solve, an option Ipopt refused, an output a callback never assigned.
Every warning points at the line in *your* code that caused it, which is what makes the
categories below worth knowing: a filter written as ``module="yapss"`` matches none of them,
because the module recorded is yours.

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
callback output left unassigned is an example: the row would be zero, which almost always
means a missing line, so the solve does not start. Where a value that looks like a mistake is
what you mean, say so explicitly --- assign ``0.0`` to the row.

YAPSS **warns** only when what you supplied is valid but something deserves your attention:
the outcome of the solve (Ipopt did not converge), the environment (an Ipopt option your build
does not provide, an environment variable that no longer has an effect), a choice with a cost
(a very large mesh segment), or a notice that a behavior will change.

Filtering
---------

To turn every YAPSS warning into an error, which is the strictest way to run a script::

    import warnings
    import yapss

    warnings.simplefilter("error", yapss.YapssWarning)

Or one category at a time::

    warnings.filterwarnings("error", category=yapss.IpoptConvergenceWarning)
    warnings.filterwarnings("ignore", category=yapss.LargeSegmentWarning)

The same on the command line, using the fully qualified name::

    python -W error::yapss.IpoptConvergenceWarning my_problem.py

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
``UnsupportedMathFunctionError`` in :doc:`callbacks` (defined in ``yapss.math`` and
re-exported from ``yapss``).
