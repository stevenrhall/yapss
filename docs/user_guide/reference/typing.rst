Type Annotations
================

Python does not check types when it runs a program,
but a static type checker, such as mypy or pyright, checks them without running it.
It finds errors that raise no exception or warning when the program runs,
and errors on a path the program takes only in some circumstances,
such as a branch that runs only when a solve fails.
Editors use the same information.
VS Code and PyCharm can complete names as they are typed and highlight mistakes as they are made,
but only where they can tell what type each value is.

Typing is optional in YAPSS, and nothing requires an annotation.
But the class declarations YAPSS uses already provide many of those benefits:
with no annotation written, an editor completes a problem's phases and fields
and highlights a misspelled name before the problem is solved.
A modest number of annotations,
on the callbacks and on the functions that take a problem or its solution,
extends the checking to nearly everything.
The :doc:`example scripts <../scripts/index>` are annotated in full,
and each passes ``mypy --strict``, mypy's strictest checks:
they are the larger illustration of what this page describes.

Declaring a Problem
-------------------

A type checker reads class bodies,
so a problem declared as a class is checked as written.
Its annotations name the classes the problem is made of,
whichever it has of phases, discrete constraints, and parameters.
From the :doc:`dynamic soaring example <../scripts/dynamic_soaring>`:

.. literalinclude:: ../../../src/yapss/examples/dynamic_soaring.py
   :language: python
   :start-at: class Phase(yapss.Phase):
   :end-at: parameter: Parameter

``State``, ``Control``, and the other vector classes are declared above these the same way,
each field a class attribute.
Every phase's independent variable is ``time``, declared by ``yapss.Phase`` itself,
so a phase class names only its vectors.
The name is ``time`` whatever the variable measures, a radius or an arc length as well as a time,
so that every phase, callback, and solution reads it the same way.

The problem class is its own annotation.
The example's functions name it where they return or take the problem:

.. code-block:: python

    def setup() -> DynamicSoaring:
        """Set up the dynamic soaring problem."""
        ...

    def plot_solution(problem: DynamicSoaring, solution: yapss.Solution) -> None:
        """Plot the circuit in three dimensions, and the quantities along it."""
        ...

If the problem class leaves out ``phases``, ``discrete``, or ``parameter``,
the type checker does not know what it contains, and checks nothing read from it.
Leave out ``parameter``, and a misspelled ``problem.parameter.gg`` goes unreported.
A problem that has no parameters can say so, with ``parameter: yapss.Parameter``,
and then reading any parameter from it, such as ``problem.parameter.g``, is reported as a mistake.

A function whose argument is annotated with plain ``yapss.Problem``,
rather than with the problem's own class, accepts any problem.
The checker still checks what every problem has,
such as ``problem.ipopt_options``, ``problem.spectral_method``, and ``solution.objective``,
so this is the right annotation for a function meant to work on any problem,
such as one that sets the Ipopt options a project always uses.
But it cannot check the problem's own names:
its phases, discrete constraints, and parameters go unchecked,
in the problem and in its solution.

What Is Checked Without Annotations
-----------------------------------

With no annotation written anywhere,
a type checker checks nearly all of a problem's setup, the Ipopt options included.
Here is the setup of the :doc:`brachistochrone example <../scripts/brachistochrone_minimal>`,
with some mistakes added.
The comment above each line is the message mypy prints for it:

.. literalinclude:: ../../../tests/typed_messages/unannotated_setup.py
   :language: python
   :start-after: # -- shown on the page
   :end-before: # -- end of what the page shows

Other checkers word their messages differently; see `Editors`_.

One mistake in a declaration is caught by YAPSS rather than by the checker.
A phase class that puts a vector in the wrong role, ``state: Control``,
raises an error as soon as the class is defined, naming the role the vector belongs in.

Annotating Callbacks
--------------------

A callback is checked once its arguments are annotated.
Each argument type names the classes the callback reads or fills (written in square brackets):

``yapss.ContinuousArg[State, Control, Parameter]``
    What a continuous callback reads: ``arg.state``, ``arg.control``, and ``arg.parameter``,
    typed as the phase's state and control classes and the problem's parameter class.

``yapss.ContinuousOut[State, Path, Integral]``
    What a continuous callback fills: ``out.dynamics`` is typed as the phase's state class,
    ``out.path`` as its path constraints, and ``out.integrand`` as its integrals.

``yapss.DiscreteArg[Parameter]``
    What the objective and discrete callbacks read.
    A phase's values are selected with the phase's *handle*, ``ph = problem.phases.phase``,
    as ``arg[ph]``, which is typed from the handle itself,
    so ``arg[ph].integral`` has that phase's integrals.

``yapss.DiscreteOut[Discrete]``
    What the discrete callback fills: ``out.discrete`` is typed as the discrete class.

Each class in the brackets must have the right role,
so naming a control where the state belongs is reported.
Classes at the end may be left out,
as in ``yapss.ContinuousArg[State, Control]`` for a callback that reads no parameters.
A class in the middle is skipped by writing its role's base class,
``yapss.ContinuousOut[State, yapss.Path, Integral]``,
which declares no names, so reading one is reported.
An argument type written with no brackets, ``yapss.ContinuousArg``, checks nothing at all.

The example scripts name each phase shape's argument types once, after the declarations:

.. literalinclude:: ../../../src/yapss/examples/brachistochrone.py
   :language: python
   :start-at: # Names for the types the annotations below use.
   :end-at: PhaseOut =

and annotate the callbacks with the names:

.. literalinclude:: ../../../src/yapss/examples/brachistochrone.py
   :language: python
   :start-at: @ph.register.continuous
   :end-at: return arg[ph].final_time
   :dedent: 4

A callback used by several phases of one shape needs the names written only once.

A callback that fills its outputs returns nothing, and is annotated ``-> None``;
for one annotated to return something,
the type checker reports an error where it is registered.
The objective returns a float when the problem is evaluated
and a symbol when YAPSS traces it for automatic differentiation,
so its return type must be the catch-all type ``Any``.
A value read in a continuous callback is typed as a numpy array, which it is:
of floats, or of symbols.

A helper that fills one output can take the output vector itself,
and should be annotated with that vector's own class,
``State`` for the dynamics:

.. literalinclude:: ../../../tests/typed/typed_problem.py
   :language: python
   :pyobject: dynamics

Using a Solution
----------------

A solution is typed from its problem, with nothing written.
``problem.solve()`` returns a solution whose type,
``yapss.Solution[Discrete, Parameter]``, comes from the problem class's annotations,
so the whole :doc:`solution tree <solution>` is typed with no other action by the user:
``solution.parameter.g`` and ``solution.multiplier.discrete.landing`` are checked,
and so is everything else in it.
A function that takes the solution is annotated with that type.

Each phase of the solution is typed from the phase's own classes,
``yapss.PhaseSolution[State, Control, Path, Integral]``:
``ps.state`` and ``ps.costate`` as the state class, ``ps.control`` as the control class,
and so on.
The checker knows which phase it is when the phase is selected with its handle,
as in the callbacks:

.. code-block:: python

    problem = setup()
    solution = problem.solve()
    ps = solution.phases[problem.phases.phase]
    ps.state.x   # checked
    ps.state.xx  # reported: "State" has no attribute "xx"

When a phase is selected by name, ``solution.phases.phase`` or ``solution.phases["phase"]``,
the checker does not know which phase it is.
It checks the names every phase solution has,
so a misspelling such as ``ps.hamiltonain`` for ``ps.hamiltonian`` is reported,
but not the phase's own state and control names.
So select a phase of a solution with its handle,
or annotate the result (see `Editors`_).

Option Values
-------------

``problem.spectral_method``, ``problem.derivatives.method``, ``problem.derivatives.order``,
and ``problem.objective.sense`` are typed with the values they accept,
so a misspelled value is reported where it is assigned.
A value that arrives through a variable typed ``str`` is reported too,
because the checker cannot tell which string it holds.
Annotate the variable with the value's type, which YAPSS exports as
``yapss.SpectralMethod``, ``yapss.DerivativeMethod``,
``yapss.DerivativeOrder``, and ``yapss.ObjectiveSense``:

.. code-block:: python

    from typing import Final

    import yapss

    METHOD: Final = "lgr"  # Final keeps the value's own type (PEP 586)


    def setup(method: yapss.SpectralMethod = "lgl") -> Brachistochrone:
        problem = Brachistochrone("Brachistochrone")
        problem.spectral_method = method
        ...
        return problem


    problem = setup(METHOD)

.. py:data:: yapss.SpectralMethod
   :type: typing.TypeAlias
   :value: Literal["lgl", "lgr", "lg"]

   The collocation method, ``problem.spectral_method``.

.. py:data:: yapss.DerivativeMethod
   :type: typing.TypeAlias
   :value: Literal["auto", "central-difference", "central-difference-full"]

   How derivatives are computed, ``problem.derivatives.method``.

.. py:data:: yapss.DerivativeOrder
   :type: typing.TypeAlias
   :value: Literal["first", "second"]

   The order of the derivatives Ipopt is given, ``problem.derivatives.order``.

.. py:data:: yapss.ObjectiveSense
   :type: typing.TypeAlias
   :value: Literal["minimize", "maximize"]

   Whether the objective is minimized or maximized, ``problem.objective.sense``.

Editors
-------

Developing a problem with completion and checking turned on in the editor
is noticeably more pleasant, and worth trying if you have not worked that way.
Names are offered as you type them,
and a misspelled name is underlined as soon as it is written,
rather than raising an error when the script runs,
which for a mistake in the plotting code is only after the solve has finished.
YAPSS is particularly well suited for code completion:
a problem and its solution are trees of attribute names, such as ``ph.state.x.bounds``,
and after each dot the editor can list every name that may come next.

VS Code and PyCharm each work out types their own way,
and differ in how far they follow YAPSS's.
mypy, run from the command line, reports all of the mistakes on this page,
with its suggestions, whatever the editor.

**VS Code**, through Pylance, completes everything on this page as it is,
including a phase selected with its handle.
Mistakes are underlined only once Pylance's type checking is turned on:
set ``python.analysis.typeCheckingMode`` to ``"basic"`` in the workspace settings.
Every mistake in `What Is Checked Without Annotations`_ is then reported,
as ``Cannot access attribute "xx" for class "State"``,
though without mypy's suggestion of the name that was meant.

**PyCharm**'s own inspections report few of these mistakes;
install the mypy plugin, which reports what mypy does.
Completion comes from PyCharm's own engine,
which follows the declarations and annotations
but does not work out which phase a handle selects.
Completion on a phase of a solution is enabled with an annotation,
most simply through a name for the phase solution's type, written once:

.. code-block:: python

    Solved = yapss.PhaseSolution[State, Control, Path, Integral]

    ps: Solved = solution.phases[ph]

What Is Not Checked
-------------------

These are the limits, stated so that a silent checker is not mistaken for a passing one.

- A phase of a solution selected by name, as `Using a Solution`_ describes.
- A phase selected by iterating over ``problem.phases`` is one the checker cannot identify,
  so its state and control names are not checked.
  A loop over phases named by their handles is checked against each of them.
  In a problem with phases ``boost`` and ``coast``:

  .. code-block:: python

      for ph in problem.phases:
          ph.state.hh.bounds = (0, 1)  # not reported

      for ph in (problem.phases.boost, problem.phases.coast):
          ph.state.hh.bounds = (0, 1)  # reported: "State" has no attribute "hh"

- ``yapss.interp`` accepts any arguments, as far as a type checker can tell;
  what it is given is checked when it is called.
- If a callback is annotated with another phase's classes,
  the checker checks against those classes,
  while YAPSS gives the callback the phase's own.
