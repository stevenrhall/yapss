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

``yapss.ContinuousArg[State, Control, Parameter]``
    What a continuous callback reads: ``arg.state``, ``arg.control``, and ``arg.parameter``, typed
    as the phase's state and control classes and the problem's parameter class.

``yapss.ContinuousOut[State, Path, Integral]``
    What a continuous callback fills: ``out.dynamics`` is typed as the phase's state class,
    ``out.path`` as its path constraints, and ``out.integrand`` as its integrals.

``yapss.EndpointArg[Parameter]``
    What the objective and discrete callbacks read. ``arg[ph]`` is typed from the phase handle
    itself, so ``arg[ph].integral`` has that phase's integrals.

``yapss.DiscreteOut[Discrete]``
    What the discrete callback fills: ``out.discrete`` is typed as the discrete class.

Each parameter is bounded by its role, so naming a control where the state belongs is
reported. Trailing parameters may be left out -- ``yapss.ContinuousArg[State, Control]`` for a
callback that reads no parameters -- and a class left bare checks nothing at all. A callback
used by several phases of one shape needs the annotation written once, and a long one can be
named once and reused: ``SlideArg = yapss.ContinuousArg[State, Control, Parameter]``.

Here is a problem's setup with every callback annotated in full:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: setup
   :lines: 1-21

A helper that fills one output can take the output vector itself, typed as the declaration it
is an instance of:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: dynamics

The vectors are named, rather than the phase's shape, because that is what every checker and
every editor can follow: the shape carries the same information, but recovering it takes a
match that some editors, PyCharm among them, do not make, and an annotation they cannot follow
gives no completion there at all.

The objective returns a float when the problem is evaluated and a symbol when YAPSS traces it
for automatic differentiation, so its honest return type is ``Any``. A value read from a
continuous callback is typed as a numpy array, which it is: of floats, or of symbols.

To type the problem itself, give `yapss.Problem` the classes it was built from --
``yapss.Problem[Phases, Discrete, Parameter]`` -- as the ``setup`` above does. The bare
``yapss.Problem`` is also valid, and answers any phase, parameter or discrete name, since it does
not say what was declared; so does the solution it returns.

Reading a solution
------------------

A solution is typed from its problem, with nothing written: ``problem.solve()`` returns
``yapss.Solution[Discrete, Parameter]``, from the classes the problem was built with, so
``solution.parameter.g`` and ``solution.multiplier.discrete.landing`` are checked, and so are
their positions in the solver's record, ``solution.nlp.index``.

A phase's solution is ``yapss.PhaseSolution[State, Control, Path, Integral]``, and every tree
on it is typed from those four: ``ps.state``, ``ps.dynamics``, ``ps.costate`` and
``ps.multiplier.dynamics`` as the state class, ``ps.control`` and ``ps.multiplier.control`` as
the control class, and so on into the solver's record, where ``ps.nlp.index.variable.state`` is
the state class too. mypy types ``solution[ph]`` from the handle, as it types ``arg[ph]``, so
this needs no annotation:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: solve_and_report

PyCharm's engine does not follow that match. There, annotate the variable, most simply through
an alias written once:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :start-at: SlideSolution =
   :end-before: def solve_and_report

What is not checked
-------------------

These are the limits, stated so that a silent checker is not mistaken for a passing one.

- ``arg[ph].initial`` and ``arg[ph].final`` are not typed, nor are a solution's ``ps.initial``,
  ``ps.final``, ``ps.multiplier.initial`` and ``ps.multiplier.final``. Each holds the phase's
  states *and* its independent variable -- ``arg[ph].final.time`` and ``arg[ph].final.x`` are
  both valid -- and a type that is one class plus one more name cannot be written in Python's
  type system. In a solution the typed read is the trajectory's end: ``ps.state.x[-1]`` and
  ``ps.time[-1]``.
- A misspelled name at the top of a continuous argument or a phase's solution, such as
  ``arg.stat`` or ``ps.stat``, is caught only at run time. The independent variable is reached by
  the name the phase gave it, ``arg.r`` for a phase declaring ``r: yapss.Independent``, and no
  type parameter can carry that name, so both accept any name at their top level. Below it,
  ``arg.state.xx`` and ``ps.state.xx`` are checked.
- Nothing checks that a callback's annotation matches the phase it is registered on. The
  classes named are what the checker uses; the runtime gives the callback the phase's own.

In VS Code, completion and navigation work as they are, but misspellings are underlined only
once Pylance's type checking is turned on: set ``python.analysis.typeCheckingMode`` to
``"standard"``.

The file the examples on this page come from, ``tests/typed/typed_problem.py``, is checked in
strict mode on every change to YAPSS, together with a list of mistakes each of which must be
reported.
