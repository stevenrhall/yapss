
Problem Definition
==================

This section outlines the general structure of an optimal control problem and its
implementation in YAPSS. See the following sections for details of how to define callback
function, set variable constraint bounds, define the initial guess, select options for
evaluating derivatives, select solver options, configure the Ipopt NLP solver source, and
access the solution.

General Formulation
-------------------

YAPSS implements in Python a pseudospectral method for solving optimal control problems,
similar to the MATLAB algorithm GPOPS-II :footcite:`Patterson:2014` described by Patterson
and Rao. YAPSS generalizes the GPOPS-II algorithm by allowing Legendre-Gauss (LG) and
Legendre-Gauss-Loabatto (LGL) collocation points in addition to the Legendre-Gauss-Radau
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

   \dot{x}^{(p)}
      = f^{(p)}(x^{(p)}, u^{(p)}, t, s), \quad p=0, \ldots, n_{p}-1

over the interval :math:`t \in [t_{0}^{(p)}, t_{f}^{(p)}]`. The trajectory of the system over
each phase is subject to the path constraints

.. math::

   h_{\text{min}}^{(p)} \leq h^{(p)} ( x^{(p)}, u^{(p)}, t, s ) \leq h_{\text{max}}^{(p)},
      \quad p=0, \ldots, n_{p}-1

where :math:`h^{(p)}` is a vector-valued function with dimension :math:`n_{h}^{(p)}`.
In addition, each phase may have integrals associated with it of the form

.. math::

   q^{(p)} = \int_{t_{0}^{(p)}}^{t_{f}^{(p)}} g^{(p)}( x^{(p)}, u^{(p)}, t, s ) \,dt,
      \quad p=0, \ldots, n_{p}-1

where :math:`g^{(p)}` is a vector-valued function with dimension :math:`n_{q}^{(p)}`. The
integrals may appear as a term in the cost function (a `Lagrangian` term) or as a
perimetric constraint.

The cost to be minimized is given by a function of all the discrete variables in problem:

.. math::

   \begin{aligned}
      J=\phi\Big[
         & x^{(0)} (t_{0}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{0}^{(n_{p}-1)}),
            t_{0}^{(0)},\ldots,t_{0}^{(n_{p}-1)}, \\
         & x^{(0)}(t_{f}^{(0)}),\ldots,x^{(n_{p}-1)}(t_{f}^{(n_{p}-1)}),
            t_{f}^{(0)},\ldots,t_{f}^{(n_{p}-1)},{q}^{(0)},\ldots,{q}^{(n_{p}-1)},{s}\Big]
   \end{aligned}

subject to additional constraints on the discrete variables,

.. math::

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

where the function :math:`d` is a vector-valued with dimension :math:`n_{d}`.

In addition, upper and lower bounds may be specified on all the decision variables:

- The state :math:`x^{(p)}` and the control :math:`u^{(p)}` vectors
- The initial and final state vectors :math:`x^{(p)}(t_{0}^{(p)})` and  :math:`x^{(p)}(t_{f}^{(p)})`
- The initial and final times :math:`t_{0}^{(p)}` and :math:`t_{f}^{(p)}`
- The parameter vector :math:`s`
- The integrals :math:`q^{(p)}`

(In this formulation, the integrals are treated as decision variables, subject to the
constraint that they are the integrals of the integrand over the phase.)

Problem Instantiation
---------------------

Given a formulation of the problem as described above, the problem can be implemented in
YAPSS and solved. In this section, we describe the instantiation of a YAPSS problem object.

Consider for example, the Goddard problem, a classic optimal control problem to maximize the
altitude of a sounding rocket launched vertically from the surface of the Earth, taking into
account the forces of gravity, drag, and thrust. (See the JupyterLab notebooks for the
`one phase Goddard problem <../notebooks/goddard_problem_1_phase.ipynb>`_ and the
`three phase Goddard problem <../notebooks/goddard_problem_3_phase.ipynb>`_ with a singular arc.)
For the three phase problem, there are three states (altitude, velocity, and mass), one control
(thrust), and one path constraint (the singular arc constraint). In addition, there are eight
discrete constraints that ensure the continuity of the state variables across phases, and that
the final time of each phase is equal to the initial time of the next phase.

The problem is instantiated as an instance of the :class:`~yapss.Problem` class. To
initialize, the user specifies the name of the problem; the relevant dimensions of the
state, control, and path constraint for each phase; and the dimensions of the static
parameters and discrete constraints. In this case, the problem is instantiated as follows:

.. doctest:: example

    >>> import yapss.numpy as np
    >>> from yapss import Problem, Solution
    >>>
    >>> problem = Problem(
    ...     name="Goddard Rocket Problem with Singular Arc",
    ...     nx=[3, 3, 3],  # Number of states in each phase
    ...     nu=[1, 1, 1],  # Number of controls in each phase
    ...     nh=[0, 1, 0],  # Number of path constraints in each phase
    ...     nd=8,          # Number of discrete constraints
    ... )

Note that all arguments are keyword arguments, and the ``name`` and ``nx`` arguments are
required. The number of phases is determined by the length of the ``nx`` argument.
If an optional argument is not provided, the assumed value is either zero or a
tuple of zeros, as appropriate.

The string representation of the problem object provides a summary of the problem:

.. doctest:: example

    >>> print(problem)
    Problem(
        name='Goddard Rocket Problem with Singular Arc',
        nx=(3, 3, 3),
        nu=(1, 1, 1),
        nq=(0, 0, 0),
        nh=(0, 1, 0),
        nd=8,
        ns=0
    )

Other sections of this reference describe the remaining steps required to so solve an optimal control
problem using YAPSS:

- Defining the `callback functions <callbacks.rst>`_ the define the objective, dynamics,
  path constraints, integrals, and discrete constraints.
- Setting `bounds <bounds.rst>`_  on decisions variables and constraints.
- Setting the `initial guess <guess.rst>`_ for the decision variables.
- Setting options for evaluating `derivatives <derivatives.rst>`_.
- Specifying `user-defined derivatives <user_derivatives.rst>`_. (rarely needed)
- Scaling the problem for improved numerical conditioning
- Defining the `mesh structure <mesh.rst>`_ for the problem.
- Setting `Ipopt options <ipopt_options.rst>`_.
- `Configuring the Ipopt binary source <configuration.rst>`_. (usually not needed)

``Problem`` Class Reference
---------------------------

.. autoclass:: yapss.Problem
   :members:
   :no-special-members:
   :no-undoc-members:

References
----------

.. footbibliography::
