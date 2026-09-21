Type Annotations
================

Typing is optional in YAPSS. Nothing requires an annotation, the examples use none, and every
check YAPSS makes at run time is made without one. But a program that is annotated should be
able to pass ``mypy --strict``, and this page says how.

What is checked without annotations
-----------------------------------

A type checker reads the declarations -- the vector classes, the phase shapes, and the
``Phases`` class -- because they are class bodies, and that is enough to check nearly all of a
problem's setup with no annotation written anywhere:

.. code-block:: python

    problem = yapss.Problem("Brachistochrone", phases=Phases)
    ph = problem.phases.slide

    ph.state.x.bounds = (0, 10)     # fine
    ph.state.xx.bounds = (0, 10)    # "State" has no attribute "xx"
    ph.state.x.bond = (0, 10)       # "ScalarField" has no attribute "bond"; maybe "bounds"?
    ph.state.x.bounds = 5.0         # a bound is a pair, not a float
    problem.phases.slid             # "Phases" has no attribute "slid"

A shape that puts a vector in the wrong role is reported too, since `yapss.Phase` annotates
each slot with its role: ``state: Control`` is an incompatible override.

Annotating callbacks
--------------------

A callback is checked once its arguments are annotated. Each argument type is generic in what
the callback can see:

``yapss.ContinuousArg[Shape, Parameter]``
    What a continuous callback reads. ``Shape`` is the phase's shape, from which ``arg.state``
    and ``arg.control`` are typed; ``Parameter`` is the problem's parameter class, and may be
    left out if the callback reads no parameters.

``yapss.ContinuousOut[Shape]``
    What a continuous callback fills: ``out.dynamics`` is typed as the shape's state,
    ``out.path`` as its path constraints, and ``out.integrand`` as its integrals.

``yapss.EndpointArg[Parameter]``
    What the objective and discrete callbacks read. ``arg[ph]`` is typed from the phase handle
    itself, so ``arg[ph].integral`` has that phase's integrals.

``yapss.DiscreteOut[Discrete]``
    What the discrete callback fills: ``out.discrete`` is typed as the discrete class.

The shape is written once, and every vector is recovered from its annotations. Here is a
continuous callback annotated in full, from a problem whose phase has the shape ``Slide``:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: Slide

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: setup
   :lines: 1-20

A helper that fills one output can take the output vector itself, typed as the declaration it
is an instance of:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: dynamics

The objective returns a float when the problem is evaluated and a symbol when YAPSS traces it
for automatic differentiation, so its honest return type is ``Any``. A value read from a
continuous callback is typed as a numpy array, which it is: of floats, or of symbols.

To type the problem itself, give `yapss.Problem` the classes it was built from --
``yapss.Problem[Phases, Discrete, Parameter]`` -- as the ``setup`` above does. The bare
``yapss.Problem`` is also valid, and answers any phase name, since it does not say which phases
were declared.

What is not checked
-------------------

These are the limits, stated so that a silent checker is not mistaken for a passing one.

- ``arg[ph].initial`` and ``arg[ph].final`` are not typed. Each holds the phase's states *and*
  its independent variable -- ``arg[ph].final.time`` and ``arg[ph].final.x`` are both valid --
  and a type that is one class plus one more name cannot be written in Python's type system.
- A misspelled name at the top of a continuous argument, such as ``arg.stat``, is caught only at
  run time. The independent variable is reached by the name the phase gave it, ``arg.r`` for a
  phase declaring ``r: yapss.Independent``, and no type parameter can carry that name, so the
  argument accepts any name at its top level. Below it, ``arg.state.xx`` is checked.
- The solution, and the arguments of user-supplied derivative callbacks, are not typed yet.

The file the examples on this page come from, ``tests/typed/typed_problem.py``, is checked in
strict mode on every change to YAPSS, together with a list of mistakes each of which must be
reported.
