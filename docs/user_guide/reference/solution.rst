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

``problem.solve()`` returns a :class:`~yapss.Solution` whenever Ipopt stops at an iterate,
whether or not it converged; the statuses at which it has none raise instead. A run that
reaches its iteration limit or stops because the step size collapses still produces a full
set of trajectories --- they simply do not satisfy any convergence criterion. Nothing about
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

Multiplier Coverage and Verification
------------------------------------

.. versionadded:: 0.3.0

    The multipliers of a state's bounds, ``state_multiplier``, ``initial_state_multiplier``
    and ``final_state_multiplier``, and of an integral's bounds,
    ``integral_bound_multiplier``.

Every bound and constraint of a problem has a multiplier on the solution. The multipliers of
a state's bounds need the most care in reading, because two bounds can apply to the state at
an end of a phase: its bound over the phase, ``bounds.phase[p].state``, and the end's own,
``bounds.phase[p].initial_state`` or ``final_state``. The NLP holds the state there by the
tighter of the two, so it has one multiplier for the pair.

-  ``state_multiplier`` is the multiplier of the bound on a state over the phase, as a
   density in time at the collocation points, like ``control_multiplier``.
-  ``initial_state_multiplier`` and ``final_state_multiplier`` are the multipliers of the
   bound on the state at each end, one number per state, whichever of the two bounds is the
   active one. Each is a single value at a point, not a density.

At an end that is not a collocation point, which is the final point under LGR and both ends
under LG, ``state_multiplier`` has no value, and the end multiplier stands alone. At an end
that is a collocation point, which is both ends under LGL and the initial point under LGR,
the same multiplier can be read either way, as a point value or as a sample of a density,
and which is right depends on the problem: a bound that is active over an interval reaching
the end has a density there, and a bound active only at the end has a point value. YAPSS
does not judge that. It reports the end multiplier always, and reports the density there as
follows, where the active side is the upper bounds if the multiplier is positive and the
lower bounds otherwise:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - On the active side, at a collocated end
     - ``state_multiplier`` there
     - End multiplier
   * - The state bound is the tighter, or the end has no bound
     - the multiplier, as a density
     - the multiplier
   * - The two bounds are equal
     - the multiplier, as a density
     - the multiplier
   * - The end's bound is the tighter
     - zero
     - the multiplier

So at a collocated end, ``state_multiplier`` is either zero or the same multiplier as the
end's, divided by the weight of that point in an integral over the phase. **The two are
one number and are never added.** The sensitivity of the objective to a state bound is the
integral of ``state_multiplier`` over the phase, plus the end multipliers at the ends that
are not collocation points, where the state bound is the active one. Where a state bound
and an end's bound are equal, the multiplier is the sensitivity to moving both together.

To tell a point value from a density sample, refine the mesh. The weight of an end point
shrinks as the mesh is refined, so a point value keeps its size while ``state_multiplier``
at the end grows, and a density sample settles while the end multiplier shrinks.

.. TODO (Steve): why the pointwise values of a state bound's multiplier from a collocation
   method are unreliable, the endpoint caveat, and when to use a path constraint instead
   (see the "Special Considerations for State Bounds" section of the bounds page).

An integral has two multipliers. ``integral_bound_multiplier`` is the multiplier of the
bounds on the integral's value, ``bounds.phase[p].integral``. ``integral_multiplier`` is the
multiplier of the constraint that makes the integral equal to its quadrature, and is the
one that appears in the Hamiltonian.

.. TODO (Steve): why there is an integral constraint at all: the integral is a decision
   variable, which keeps the problem sparse.

The multipliers YAPSS *does* return are checked against objective sensitivities. A
multiplier is a derivative of the optimal objective with respect to the constraint or
bound it belongs to, so perturbing that constraint and re-solving measures it
independently of the transcription; for a density the prediction is its integral over
the phase, :math:`\sum_k h w_k \mu_k`. Before the 0.3.0 release every reported
multiplier kind was measured this way across the three spectral methods, single and
multi-segment meshes, unit and non-unit ``problem.scale.*``, and both senses, agreeing
to a worst relative error of :math:`10^{-8}`. The costate was also checked against a
closed form, where it converges spectrally, and the identities were confirmed on the
brachistochrone, isoperimetric, and three-phase Goddard examples. A subset of these
checks is pinned in the test suite.

The checks exist because the history warranted them: ``control_multiplier`` was wrong in
every release through 0.1.1, and ``control_multiplier`` and ``path_multiplier`` were
scaled wrongly on any phase whose duration was not 2 through 0.2.2. Both were found by
inspection rather than by a test.

Three situations remain in which a reported multiplier is correct but is *not* an
objective sensitivity. All three are properties of the problem, not of YAPSS:

-  **A degenerate optimum.** Where the optimum is a manifold rather than a point, or the
   gradients of the active constraints are linearly dependent, the split among the
   multipliers is not determined, and Ipopt's choice among the valid ones is arbitrary.
   Only quantities invariant under that choice --- typically a sum, or the norm of a
   group --- are meaningful. The isoperimetric example is one: its optimum is invariant
   under rotation, and the two closure multipliers vary with the initial guess while
   their norm does not.
-  **A one-sided sensitivity.** Where a perturbation is feasible in one direction only,
   as when a state sits exactly on a bound, the optimal objective has a kink and only
   the one-sided derivative agrees with the multiplier.
-  **A collocated endpoint.** How a multiplier at an endpoint divides between the
   costate, a path constraint, and an endpoint condition is a matter of convention.
   ``costate[:, 0]`` is the costate at ``time_c[0]``, which under LG is not
   :math:`t_0`.

Multiplier and Costate Scaling
------------------------------

The costate, ``control_multiplier``, and ``path_multiplier`` are approximations to the
continuous-time multipliers of the optimal control problem: densities in time, so that
the Hamiltonian is :math:`H = \lambda^T f + \mu_q^T g` and stationarity in the control
reads :math:`\partial H / \partial u + \mu_u + \mu_h^T \partial h / \partial u = 0`,
with :math:`\mu_u` the control-bound multiplier and :math:`\mu_h` the path multiplier.
The last term is absent wherever no path constraint is active. The raw Ipopt multipliers are on the discrete rows and bounds; YAPSS divides
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
-  **problem** (*Problem*): A deep copy of the original problem definition. A pickled solution
   keeps it only when the problem can be pickled; see `Saving a Solution`_.
-  **objective** (*float*): The value of the objective function at the optimal solution.
-  **parameter** (*np.ndarray*): An array of the optimal parameter values, one per parameter.
-  **parameter_multiplier** (*np.ndarray*): Lagrange multipliers for parameter bounds, one per
   parameter.
-  **discrete** (*np.ndarray*): An array of the discrete constraint functions, evaluated at the
   optimal solution, one per constraint.
-  **discrete_multiplier** (*np.ndarray*): Lagrange multipliers corresponding to the discrete
   constraint functions, one per constraint.
-  **phase** (*tuple*): A tuple of `SolutionPhase` objects, each containing information
   specific to a phase in the solution.
-  **nlp_info** (*NLPInfo*): A dataclass container with information returned from the Ipopt NLP solver.

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
-  **state_multiplier** (*np.ndarray*): Lagrange multipliers for the state bounds, at
   collocation points, as a density in time (see `Multiplier Coverage and Verification`_).
-  **initial_state_multiplier**, **final_state_multiplier** (*np.ndarray*): Lagrange
   multipliers for the bounds on the initial and final state.
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
-  **integral_bound_multiplier** (*np.ndarray*): Lagrange multipliers for the bounds on the
   integrals.

The Hamiltonian function, derived from the costates, dynamics, integrands, and integral
multipliers, is also available:

-  **hamiltonian** (*np.ndarray*): Values of the Hamiltonian function
   :math:`H = \lambda^T f + \mu_q^T g`, evaluated at collocation points.

There is no path-constraint term in :math:`H`, by construction rather than by omission.
Written in the standard form :math:`h - h_\text{bound} \le 0`, the augmented Hamiltonian
adds :math:`\mu_h^T (h - h_\text{bound})`, and complementary slackness makes that term
zero on-shell: :math:`\mu_h` vanishes wherever the constraint is inactive, and the
residual vanishes wherever it is active. So the reported :math:`H` *is* the augmented
Hamiltonian in value along the solution, and it is the quantity that is constant along an
autonomous trajectory and that vanishes at a free, interior final time --- with or without
an active path constraint.

The two agree in value, not in derivative. Off the solution the term is a function of
:math:`u` like any other, which is why the stationarity condition above keeps its
:math:`\mu_h` term on a constrained arc.

Array Shapes
------------

A phase's arrays have one row per variable and one column per point, on one of two sets of
points. Write ``Nt`` for ``len(time)`` and ``Nc`` for ``len(time_c)``, and ``nx``, ``nu``,
``nh`` and ``nq`` for the phase's numbers of states, controls, path constraints and
integrals. A phase with none of a kind has an array with no rows.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Attribute
     - Shape
   * - ``time``
     - ``(Nt,)``
   * - ``state``
     - ``(nx, Nt)``
   * - ``initial_state``, ``final_state``, ``initial_state_multiplier``,
       ``final_state_multiplier``
     - ``(nx,)``
   * - ``time_c``, ``hamiltonian``
     - ``(Nc,)``
   * - ``dynamics``, ``costate``, ``state_multiplier``
     - ``(nx, Nc)``
   * - ``control``, ``control_multiplier``
     - ``(nu, Nc)``
   * - ``path``, ``path_multiplier``
     - ``(nh, Nc)``
   * - ``integrand``
     - ``(nq, Nc)``
   * - ``integral``, ``integral_multiplier``, ``integral_bound_multiplier``
     - ``(nq,)``

For a mesh of ``m`` segments with ``n`` collocation points each, the counts are:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Spectral method
     - ``Nt``
     - ``Nc``
   * - ``"lgl"``
     - ``m (n - 1) + 1``
     - ``m (n - 1) + 1``
   * - ``"lgr"``
     - ``m n + 1``
     - ``m n``
   * - ``"lg"``
     - ``m n + m + 1``
     - ``m n``

So under LGR and LG the state has more columns than the control, and a control is plotted
against ``time_c``, not ``time``.

Saving a Solution
-----------------

A solution can be pickled, and copied with :func:`copy.copy` or :func:`copy.deepcopy`.
Every solved quantity is data and is always kept.

The one part of a solution that is not data is ``problem``, which holds the callbacks.
Python pickles a function by its name, so a problem can be pickled only when its callbacks
are functions defined at the top level of a module, and everything in ``problem.auxdata``
is picklable. A callback defined inside another function, as in the examples' ``setup()``,
or a lambda, cannot be pickled. So:

-  When the problem can be pickled, the solution is pickled with it, and the solution loaded
   from the pickle has its ``problem``.
-  When it cannot, the solution is pickled without it. Reading ``problem`` on the solution
   loaded from the pickle raises ``AttributeError``, with Python's reason for refusing the
   problem. Nothing else is affected.

To load a solution together with its problem, the module that defines the callbacks must be
importable where the pickle is loaded, as for any pickled function. A copy always keeps its
problem, since copying does not go through pickling.

.. versionchanged:: 0.3.0

    A solution whose problem cannot be pickled is pickled without it; before, pickling the
    solution raised. The phases of a copied or pickled solution are now its phases; before,
    ``phase`` came back as a tuple of one tuple.

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
