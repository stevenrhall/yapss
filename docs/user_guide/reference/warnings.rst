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
          ├── yapss.IpoptOptionSettingWarning    Ipopt refused an option value
          ├── yapss.LargeSegmentWarning          a mesh segment has very many points
          ├── yapss.UnsetOutputWarning           a callback never assigned an output row
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

Filtering
---------

To turn every YAPSS warning into an error, which is the strictest way to run a script::

    import warnings
    import yapss

    warnings.simplefilter("error", yapss.YapssWarning)

Or one category at a time::

    warnings.filterwarnings("error", category=yapss.IpoptConvergenceWarning)
    warnings.filterwarnings("ignore", category=yapss.UnsetOutputWarning)

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
.. autoexception:: yapss.UnsetOutputWarning
.. autoexception:: yapss.LargeSegmentWarning

The other categories are documented where the behavior they report is described:
:class:`~yapss.IpoptConvergenceWarning` in :doc:`solution`,
:class:`~yapss.IpoptOptionSettingWarning` in :doc:`ipopt_options`, and
``UnsupportedMathFunctionError`` in :doc:`callbacks` (defined in ``yapss.math`` and
re-exported from ``yapss``).
