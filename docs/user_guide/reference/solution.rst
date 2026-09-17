The Solution Object
===================

The :class:`yapss.Solution` class stores the solution to an optimal control problem. An
instance of this class contains detailed information about the optimal decision variables,
Lagrange multipliers, and additional data relevant to the problem.

For example, consider the Goddard rocket problem, whose trajectory includes a singular
arc and therefore requires a three-phase solution. You can solve this problem and obtain a
`Solution` object with the following Python code:

.. doctest:: example

   >>> from yapss.examples.goddard_problem_3_phase import setup
   >>> problem = setup()
   >>> problem.ipopt_options.print_level = 0  # Suppress output
   >>> problem.ipopt_options.sb = "yes"       # Silent mode
   >>> problem.ipopt_options.tol = 1e-8       # Set solver tolerance
   >>> solution = problem.solve()
   >>> solution
   <yapss._private.solution.Solution: 'Goddard Rocket Problem with Singular Arc'>

Checking That the Solve Converged
---------------------------------

``problem.solve()`` returns a :class:`~yapss.Solution` regardless of the status reported by
Ipopt. A run that reaches its iteration limit or stops because the step size collapses still
produces a full set of trajectories --- they simply do not satisfy any convergence criterion. Nothing about
the returned object looks different.

Since version 0.2.0, YAPSS emits an :class:`~yapss.IpoptConvergenceWarning` when that
happens, so the outcome is at least not silent. For example, forcing Ipopt to stop
immediately by setting ``max_iter = 0`` reliably produces an unconverged solve, and the
warning appears as soon as ``solve()`` returns:

.. code-block:: pycon

   >>> problem.ipopt_options.max_iter = 0
   >>> solution = problem.solve()
   IpoptConvergenceWarning: Ipopt did not converge. Status -1: "Maximum Number of Iterations
   Exceeded." The returned solution does not satisfy Ipopt's convergence criteria and should
   not be treated as an optimal trajectory. Check solution.status and the Ipopt output
   before using these results.

This illustration is not itself doctested, since the warning text goes to ``stderr``
rather than ``stdout`` and capturing it would need the same ``warnings`` bookkeeping this
example is trying to avoid. The behavior it depicts is covered by
``tests/modules/test_convergence_warning.py``.

Three Ipopt statuses are treated as success and do not warn: ``0`` (optimal solution
found), ``1`` (solved to acceptable level), and ``6`` (feasible point found for a square
problem). Status ``1`` is included deliberately. It is a normal outcome when tolerances are
pushed hard --- the answer is routinely correct to far more digits than requested --- and
warning on it would train users to disregard the warning, which would destroy its value for
the cases that matter.

The warning class is public, so it can be silenced or escalated in the usual way:

.. code-block:: python

    import warnings
    import yapss

    warnings.filterwarnings("ignore", category=yapss.IpoptConvergenceWarning)
    warnings.filterwarnings("error", category=yapss.IpoptConvergenceWarning)

Projects that run their test suites with ``-W error`` will newly see failures on
unconverged solves.

.. note::

    **Every unconverged solve warns**, including repeated solves from the same line in a
    loop. Python normally reports a repeated warning from one line only once, but that
    deduplication does not carry over from one ``solve()`` to the next, so the ``"default"``
    and ``"once"`` filter actions do not reduce the count. The ``"ignore"`` and ``"error"``
    actions shown above work as usual. To act on only some failures, branch on the status
    instead, as described below.

What the warning does not tell you
..................................

**A cancelled solve carries no guarantee at all.** Interrupting with Ctrl-C stops Ipopt
at whatever iterate it had reached, reported as status ``5``. The warning tells you it did
not converge, and that is the only thing it tells you.

**Branch on the status, not on the warning.** Warnings are for people reading output.
Code that needs to know should test ``solution.converged``, or ``solution.status`` to tell
the failure modes apart. Both are described under `Information from Ipopt Solver`_ below,
and neither is affected by warning filters.

**Only** ``Problem.solve()`` **warns.** The check is deliberately placed at the public
boundary rather than inside the solver, so that the reported source location is your own
call. Internal routines that solve repeatedly do not warn on each attempt.

Incomplete and Unverified Multipliers
--------------------------------------

.. warning::

    **State bound multipliers are not returned.** Ipopt computes a Lagrange multiplier
    for every bound on ``bounds.phase[p].state``, ``initial_state``, and
    ``final_state``, but none of these are currently exposed on `SolutionPhase`.
    Reporting them correctly is subtler than it looks: at a point where a continuous
    state bound is also active --- always possible for LGL at either endpoint, and for
    LGR at the initial endpoint --- the raw Ipopt multiplier isn't a usable value on its
    own; it needs to be divided by the quadrature weight and folded into the costate
    instead. Working out that logic is planned for a future release. See :doc:`bounds`
    for the related discussion of state bounds used as path constraints.

    **The multipliers YAPSS returns have limited test coverage.** This is a research code in
    active development, and Lagrange multipliers are the least exercised part of it.
    ``control_multiplier`` was wrong in every release through 0.1.1, and ``control_multiplier``
    and ``path_multiplier`` were scaled wrongly on any phase whose duration was not 2 through
    0.2.2; both were caught by inspection. Since 0.2.3 a test checks them against the costate
    on a problem with a known solution. Treat multiplier values as provisional until you have
    checked them against a known solution for your problem.

Multiplier and Costate Scaling
------------------------------

The costate, ``control_multiplier``, and ``path_multiplier`` are approximations to the
continuous-time multipliers of the optimal control problem: densities in time, so that
the Hamiltonian is :math:`H = \lambda^T f + \mu_q^T g` and stationarity in the control
reads :math:`\partial H / \partial u + \mu_u = 0` with :math:`\mu_u` the control-bound
multiplier. The raw Ipopt multipliers are on the discrete rows and bounds; YAPSS divides
out the quadrature weight and the phase half-duration :math:`(t_f - t_0)/2` to report
them per unit time. On a phase of zero duration these densities are undefined, and
``control_multiplier`` and ``path_multiplier`` are NaN there.

Multiplier and Costate Sign
----------------------------

The Lagrange multipliers and costates in a `Solution` -- ``parameter_multiplier``,
``discrete_multiplier``, ``control_multiplier``, ``costate``, ``path_multiplier``,
``duration_multiplier``, ``integral_multiplier``, and the time-bound multipliers --
are all reported with correct sign relative to the objective :math:`J` as written in
the objective callback, :math:`\mu = dJ/dc`, regardless of whether
``problem.sense`` is ``"minimize"`` or ``"maximize"``. See :doc:`callbacks` for why
this is only true when ``problem.sense`` is used to select maximization, rather than
negating the objective directly.

Representation of Solution Objects
----------------------------------

The `repr()` output of a `Solution` object confirms that it is an instance of
:class:`~yapss.Solution` and displays the problem name.

The `str()` representation provides additional information, including the Ipopt status
code and message (indicating the success of the optimization) and the objective value at
the optimal solution:

.. doctest:: example

   >>> print(solution)
   <yapss._private.solution.Solution> object
       Name: Goddard Rocket Problem with Singular Arc
       Ipopt Status Code: 0
       Status Message: Optimal Solution Found.
       Objective Value: 18550.87...

Structure of a :class:`~yapss.Solution` Instance
------------------------------------------------

The `Solution` object contains various attributes stored in a relatively flat structure, each representing a key element of the solution:

-  **name** (*str*): The name of the optimal control problem.
-  **problem** (*Problem*): A deep copy of the original problem definition.
-  **objective** (*float*): The value of the objective function at the optimal solution.
-  **parameter** (*np.ndarray*): An array of the optimal parameter values.
-  **parameter_multiplier** (*np.ndarray*): Lagrange multipliers for parameter bounds.
-  **discrete** (*np.ndarray*): An array of the discrete constraint functions, evaluated at the optimal solution.
-  **discrete_multiplier** (*np.ndarray*): Lagrange multipliers corresponding to the discrete constraint functions.
-  **phase** (*yapss.SolutionPhases*): A tuple of `SolutionPhase` objects, each containing information
   specific to a phase in the solution.
-  **nlp_info** (*yapss.NLPInfo*): A dataclass container with information returned from the Ipopt NLP solver.

Attributes of a `SolutionPhase` Instance
------------------------------------------------------

The `phase` attribute is a tuple of `SolutionPhase` objects, one for each phase in the optimal
control problem. Each `SolutionPhase` object includes:

-  **index** (*int*): The phase index, useful for functions that further process phase
   solutions (e.g., plotting).
-  **time** (*np.ndarray*): An array of time values at interpolation points, where only the
   state trajectory is evaluated.
-  **time_c** (*np.ndarray*): An array of time values at collocation points, where dynamics
   and path constraints are enforced. If LGL collocation points are used, `time` and `time_c`
   match, simplifying post-processing.
-  **initial_time**, **final_time** (*float*): The initial and final time of the phase
   (`time[0]` and `time[-1]`).
-  **initial_time_multiplier**, **final_time_multiplier** (*float*): Lagrange multipliers
   for the initial- and final-time bounds.
-  **duration** (*float*): The phase duration, `time[-1] - time[0]`.
-  **state** (*np.ndarray*): Optimal state values at interpolation points.
-  **initial_state**, **final_state** (*np.ndarray*): Initial and final state of the phase,
   `state[:, 0]` and `state[:, -1]`.
-  **control** (*np.ndarray*): Optimal control values at collocation points.
-  **control_multiplier** (*np.ndarray*): Lagrange multipliers for control bounds, as a
   density in time (see below).

Results of user-defined functions (dynamics, path, etc.) are stored in each `SolutionPhase`:

-  **dynamics** (*np.ndarray*): Values of the dynamics function, which represents the time
   derivatives of state variables, evaluated at collocation points.
-  **path** (*np.ndarray*): Values of the path constraint function at collocation points.
-  **integrand** (*np.ndarray*): Values of the integrand function, evaluated at collocation
   points.
-  **integral** (*np.ndarray*): Values of the integral functions at the optimal solution.

Lagrange multipliers for constraints are also stored:

-  **costate** (*np.ndarray*): Optimal costate values at collocation points.
-  **path_multiplier** (*np.ndarray*): Lagrange multipliers for the path constraint function,
   as a density in time (see below).
-  **duration_multiplier** (*float*): Lagrange multiplier associated with the duration constraint.
-  **integral_multiplier** (*np.ndarray*): Lagrange multipliers associated with the integral
   constraints, enforcing equality of integrals over each phase.

The Hamiltonian function, derived from the costates, dynamics, integrands, and integral
multipliers, is also available:

-  **hamiltonian** (*np.ndarray*): Values of the Hamiltonian function, evaluated at collocation
   points.

Information from Ipopt Solver
-----------------------------

.. versionadded:: 0.3.0

    ``solution.status``, ``solution.converged``, and :class:`yapss.IpoptStatus`.

Two attributes summarize how the solve ended:

-  **status** (:class:`~yapss.IpoptStatus`): The status Ipopt reported. ``IpoptStatus`` is an
   ``IntEnum`` naming Ipopt's return codes, so ``solution.status == 0`` and
   ``solution.status == yapss.IpoptStatus.SOLVE_SUCCEEDED`` are the same test.
   ``solution.status.message`` is Ipopt's own description, as printed on its ``EXIT:``
   line.
-  **converged** (*bool*): Whether Ipopt reported a converged solution: status ``0``,
   ``1``, or ``6``.

A solve returns a ``Solution`` only when Ipopt has an iterate to report. For the statuses
where it has none, ``solve()`` raises instead of returning a solution made of placeholder
values: ``ValueError`` for too few degrees of freedom (``-10``), inconsistent bounds
(``-11``), an invalid option (``-12``), or a NaN or Inf returned by a callback or its
derivative during the solve (``-13``); ``MemoryError`` when Ipopt runs out of memory
(``-102``); and ``RuntimeError`` for a failure inside Ipopt itself. With status ``-13``,
Ipopt reports the point it had reached but sets every constraint value and multiplier to
zero, so a solution built from it would show costates and multipliers that are not.

The `nlp_info` attribute provides detailed information from the Ipopt solver, including:

-  **ipopt_status** (:class:`~yapss.IpoptStatus`): The Ipopt status code, the same object
   as ``solution.status``.
-  **ipopt_status_message** (*str*): The corresponding status message.
-  **obj_val** (*float*): Objective function value at the optimal solution.
-  **x** (*np.ndarray*): Optimal values of the NLP decision variables, combining all
   decision variables into a single array.
-  **g** (*np.ndarray*): Values of the NLP constraint functions at the optimal solution.

Lagrange multipliers associated with variable bounds and constraints:

-  **mult_x_L**, **mult_x_U** (*np.ndarray*): Lagrange multipliers for the lower and upper
   bounds on decision variables.
-  **mult_g** (*np.ndarray*): Lagrange multipliers for the NLP constraint functions.

``Solution`` Class Reference
----------------------------

.. autoclass:: yapss.Solution
    :members:

``IpoptStatus`` Class Reference
-------------------------------

.. autoclass:: yapss.IpoptStatus
    :members: message, converged
    :undoc-members:

``IpoptConvergenceWarning`` Class Reference
-------------------------------------------

.. autoexception:: yapss.IpoptConvergenceWarning
