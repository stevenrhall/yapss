
Problem Definition
==================

This section outlines the general structure of an optimal control problem and its
implementation in YAPSS. See the following sections for details of how to define callback
functions, set variable constraint bounds, define the initial guess, select options for
evaluating derivatives, select solver options, understand how YAPSS connects to Ipopt, and
access the solution.

General Formulation
-------------------

YAPSS implements in Python a pseudospectral method for solving optimal control problems,
similar to the MATLAB algorithm GPOPS-II :footcite:`Patterson:2014` described by Patterson
and Rao. YAPSS generalizes the GPOPS-II algorithm by allowing Legendre-Gauss (LG) and
Legendre-Gauss-Lobatto (LGL) collocation points in addition to the Legendre-Gauss-Radau
(LGR) points used by GPOPS-II. In this formulation, the optimal control problem is defined
over multiple phases, each with its own dynamics, path constraints, and integrals. In
addition, the problem may depend on a vector of static parameters that are to be optimized
over as well.

The general multistage optimal control problem is defined as follows: There are
:math:`n_{p}` phases, each with a state vector :math:`x^{(p)}` with dimension
:math:`n_{x}^{(p)}`, and a control vector :math:`u^{(p)}` with dimension
:math:`n_{u}^{(p)}`. In addition, there is a static parameter vector :math:`s` with
dimension :math:`n_{s}` that applies to all phases. The dynamics of the problem for each
phase are then given by

.. math::
   :label: dynamics

   \dot{x}^{(p)}
      = f^{(p)}(x^{(p)}, u^{(p)}, t, s), \quad p=0, \ldots, n_{p}-1

over the interval :math:`t \in [t_{0}^{(p)}, t_{f}^{(p)}]`. The trajectory of the system over
each phase is subject to the path constraints

.. math::
   :label: path

   h_{\text{min}}^{(p)} \leq h^{(p)} ( x^{(p)}, u^{(p)}, t, s ) \leq h_{\text{max}}^{(p)},
      \quad p=0, \ldots, n_{p}-1

where :math:`h^{(p)}` is a vector-valued function with dimension :math:`n_{h}^{(p)}`.
In addition, each phase may have integrals associated with it of the form

.. math::
   :label: integral

   q^{(p)} = \int_{t_{0}^{(p)}}^{t_{f}^{(p)}} g^{(p)}( x^{(p)}, u^{(p)}, t, s ) \,dt,
      \quad p=0, \ldots, n_{p}-1

where :math:`g^{(p)}` is a vector-valued function with dimension :math:`n_{q}^{(p)}`. The
integrals may appear as a term in the cost function (a `Lagrangian` term) or as an
isoperimetric constraint.

The cost to be minimized is given by a function of all the discrete variables in the
problem:

.. math::
   :label: cost

   \begin{aligned}
      J=\phi\Big[
         & x^{(0)} (t_{0}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{0}^{(n_{p}-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(n_{p}-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{f}^{(n_{p}-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(n_{p}-1)},{q}^{(0)},\ldots,{q}^{(n_{p}-1)},{s}\Big]
   \end{aligned}

subject to additional constraints on the discrete variables,

.. math::
   :label: discrete

   \begin{aligned}
      d_\text{min} \le d
         \Big[
         & x^{(0)} (t_{0}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{0}^{(n_{p}-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(n_{p}-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{f}^{(n_{p}-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(n_{p}-1)},{q}^{(0)},\ldots,{q}^{(n_{p}-1)},{s}
         \Big]
         \le d_\text{max}
   \end{aligned}

where :math:`d` is a vector-valued function with dimension :math:`n_{d}`.

In addition, upper and lower bounds may be specified on all the decision variables:

- The state :math:`x^{(p)}` and the control :math:`u^{(p)}` vectors
- The initial and final state vectors :math:`x^{(p)}(t_{0}^{(p)})` and :math:`x^{(p)}(t_{f}^{(p)})`
- The initial and final times :math:`t_{0}^{(p)}` and :math:`t_{f}^{(p)}`
- The parameter vector :math:`s`
- The integrals :math:`q^{(p)}`

(In this formulation, the integrals are treated as decision variables, subject to the
constraint that they are the integrals of the integrand over the phase.)

The *shape* of the optimal problem is defined by the dimensions of all the equations above

Problem Instantiation
---------------------

.. note::

   **Reading the attribute paths in this guide.** A path such as
   :samp:`{problem}.phases.{flight}.state.{h}.bounds` mixes two kinds of name. The upright
   names -- ``phases``, ``state``, ``bounds`` -- are YAPSS's, and are typed exactly as they
   appear. The *italic* names are yours: the variable you assigned the problem to, the phase
   you declared, the field you named. Only the italic parts differ from one program to the
   next, and a path is as short as the variable you start it from --
   :samp:`{ph}.state.{h}.bounds` says the same thing once
   :samp:`{ph} = {problem}.phases.{flight}` has been written.

Given a formulation of the problem as described above, the problem can be implemented in YAPSS
and solved. To do so, we need to define the shape of the problem, meaning a  description of the
decision variables* and the *constraint functions*. Decision variables are the discrete
parameters and functions of time that are optimized over, and constraint functions are
functions of the decision variables that must satisfy equality or inequality constraints. The
description includes the names which refer to the decision variables and functions. Because
YAPSS can solve multiphase problems, the description also includes names for each phase, and
which decision variables and constraint functions are associated with each phase.

Decision variables include:

- The state vectors for each phase of the problem
- The control vectors for each phase
- A parameter vector that defines parameters that can be applied to every phase
- The initial and final times of each phase
- The vector of values of the integrals evaluated over each phase

The last item is surprising, since the integrand for each integral is expressed in terms of the state, control, and parameter vectors. However, it turns out that the nonlinear program (NLP) that results from the psudospectral transcription is much sparser if the integral values are treated as decision variables.

Constraint functions include:

- The vector equations of motion describing the dynamics for each phase
- A vector path constraint function for each phase
- A vector of inequality constraints for each phase that requires that the vector of integral values
  is the same as the value calculated by integranding the vector integrand
- A vector of discrete constraints that relate all the discrete quantites in the problem, including the
  parameter vector, vector of integral values, and initial and final times for each phase

Defining the shape of the optimal control problem requires naming each phase, each decison variable vector, each constraint function vector, and the elements of each of those vectors. In addition, the independent variable for each phase (usually time, but not always) must be named. Once the shape of the problem is defined and instantiated, the only way to reference decision variables and constraint functions is by name. The is a big departure from the YAPSS API prior to version 0.4.0, in which the only way to refer to them was by bare numerical index.

Consider, for example, the Goddard problem, a classic optimal control problem to maximize the
altitude of a sounding rocket launched vertically from the surface of the Earth, taking into
account the forces of gravity, drag, and thrust. (See the JupyterLab notebooks for the
`one phase Goddard problem <../notebooks/goddard_problem_1_phase.ipynb>`_ and the
`three phase Goddard problem <../notebooks/goddard_problem_3_phase.ipynb>`_ with a singular arc.)
For the one phase problem, there are three states (altitude, velocity, and mass) and one control
(thrust).

Problem instantiation yields an initialized `yapss.Problem` object that describes the shape of
the optimal of an optimal control, at more or less the level of detail that can be gleaned from
a description like the description given in the paragraph above. The names for the elements of
the state and control vectors, as well as a name for the phase. For more general problems, the
instantiation includes descriptions and names of path constraints, integral cost functions,
integral constraints, linkage constraints, etc.

We begin by defining the important vectors in the problem, in this case the state vector and
the control vector:

.. doctest:: example

    >>> import yapss
    >>>
    >>> class RocketState(yapss.State):
    ...     """Define the rows of the state vector."""
    ...     h = yapss.scalar()
    ...     """Altitude."""
    ...     v = yapss.scalar()
    ...     """Velocity."""
    ...     m = yapss.scalar()
    ...     """Mass."""
    ...
    >>> class Thrust(yapss.Control):
    ...     """Define the rows of the control vector."""
    ...     thrust = yapss.scalar()
    ...     """Thrust."""

The vector classes can be named anything you like, and in a one-phase problem it would be
natural to default to "State" and "Control". But for multi-phase problems, the states in
different phases don't have to have the same shape, so naming vectors more descriptively can be
helpful.

Once the vector definitions are complete, the phases of the problem can be defined, in two
steps. First, the shape of a phase is a subclass of `yapss.Phase`, whose annotations say which
vector class plays each role. A ``state`` is required; ``control``, ``path``, and ``integral``
are optional, and a phase that omits one has none of it. One more annotation, of type
`yapss.Independent`, names the phase's independent variable. This problem has no path
constraints or integrals, and runs over time:

.. doctest:: example

    >>> class Flight(yapss.Phase):
    ...     """The shape of the flight: the rocket's state, its thrust, and time."""
    ...
    ...     state: RocketState
    ...     control: Thrust
    ...     time: yapss.Independent

Then the phases themselves are named in a subclass of `yapss.Phases`, each annotated with its
shape. This problem has only one:

.. doctest:: example

    >>> class Phases(yapss.Phases):
    ...     """Define the single phase of this problem."""
    ...
    ...     flight: Flight

A shape is not a phase, so two phases may share one: in the three-phase version of this
problem, the boost and coast arcs are both annotated with one shape, and only the singular arc
between them, which adds a path constraint, has its own.

The independent variable is named like everything else about a phase, and YAPSS does not
choose the name: ``time`` is what most problems call it, but a phase that runs over a radius
declares ``r: yapss.Independent``, and its extent is then set as :samp:`{ph}.{r}.initial` and
:samp:`{ph}.{r}.final`. A shape that names none is refused, because a phase has one whether or
not it is written down, and a shape a reader cannot see whole is worth less than a line saved.

Then instantiate the problem, by providing the name of the problem and the phase information:

.. doctest:: example

    >>> problem = yapss.Problem("Goddard rocket, one phase", phases=Phases)

With that, the shape and name definition of the problem is complete. The string representation
of the problem object provides a summary of the problem:

.. doctest:: example

    >>> print(problem)
    <Problem 'Goddard rocket, one phase' phases=(flight)>

The `yapss.Problem` constructor takes the problem name as its only positional argument,
followed by three keyword arguments:

- `name` (positional): The name of the optimal control problem, which may be used in messages
  and printed output.
- `phases`: The phases of the problem, defined as a subclass of `yapss.Phases`. This keyword is
  optional, and defaults to `yapss.Phases` itself, which declares no phases. Zero phases is a
  count like any other: a problem with none reduces to a parameter optimization problem, and is
  written by leaving the keyword out.
- `discrete`: The discrete constraints of the problem, defined as a subclass of
  `yapss.Discrete`. (Path constraints belong to a phase, and are declared there.) This keyword is
  optional, and defaults to `yapss.Discrete` itself, which declares no fields.
- `parameter`: The static parameters of the problem, defined as a subclass of `yapss.Parameter`.
  This keyword is optional, and defaults to `yapss.Parameter` itself, which declares no fields.

See the :ref:`Problem Class Reference <problem-class-reference>` section below for the full API.

Other sections of this reference describe the remaining steps required to solve an optimal control
problem using YAPSS:

- Defining vectors
- Defining the :doc:`callback functions <callbacks>` that define the objective, dynamics,
  path constraints, integrals, and discrete constraints.
- Setting :doc:`bounds <bounds>` on decision variables and constraints.
- Setting the :doc:`initial guess <guess>` for the decision variables.
- Setting options for evaluating :doc:`derivatives <derivatives>`.
- :doc:`Scaling <scaling>` the problem for improved numerical conditioning.
- Defining the :doc:`mesh structure <mesh_structure>` for the problem.
- Setting :doc:`Ipopt options <ipopt_options>`.
- :doc:`How YAPSS connects to Ipopt <ipopt_backend>`. (background; nothing to configure)

.. _problem-class-reference:

``Problem`` Class Reference
---------------------------

.. autoclass:: yapss.Problem(name, *, phases=Empty, discrete=Empty, parameter=Empty)
   :members:
   :no-special-members:
   :no-undoc-members:

References
----------

.. footbibliography::
