The Solution Object
===================

The :class:`yapss.Solution` class stores the solution to an optimal control problem. An
instance of this class contains detailed information about the optimal decision variables,
Lagrange multipliers, and additional data relevant to the problem.

For example, consider the Goddard rocket problem with a trajectory that includes a
singular arc, requiring a three-phase solution. You can solve this problem and obtain a
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
       Status Message: Optimal Solution Found
       Objective Value: 18550.87...

Structure of a :class:`~yapss.Solution` Instance
------------------------------------------------

The `Solution` object contains various attributes stored in a relatively flat structure, each representing a key element of the solution:

-  **name** (*str*): The name of the optimal control problem.
-  **problem** (*Problem*): A deep copy of the original problem definition.
-  **objective** (*float*): The value of the objective function at the optimal solution.
-  **parameter** (*np.ndarray*): An array of the optimal parameter values.
-  **discrete** (*np.ndarray*): An array of the discrete constraint functions, evaluated at the optimal solution.
-  **discrete_multiplier** (*np.ndarray*): Lagrange multipliers corresponding to the discrete constraint functions.
-  **phase** (*yapss.SolutionPhases*): A tuple of :class:`~yapss.SolutionPhase` objects, each containing information 
   specific to a phase in the solution.
-  **nlp_info** (*yapss.NLPInfo*): A data class container with information returned from the Ipopt NLP solver.

Attributes of a :class:`~yapss.SolutionPhase` Instance
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
-  **duration** (*float*): The phase duration, `time[-1] - time[0]`.
-  **state** (*np.ndarray*): Optimal state values at interpolation points.
-  **initial_state**, **final_state** (*np.ndarray*): Initial and final state of the phase,
   `state[:, 0]` and `state[:, -1]`.
-  **control** (*np.ndarray*): Optimal control values at collocation points.

Results of user-defined functions (dynamics, path, etc.) are stored in each `SolutionPhase`:

-  **dynamics** (*np.ndarray*): Values of the dynamics function, which represents the time
   derivatives of state variables, evaluated at collocation points.
-  **path** (*np.ndarray*): Values of the path constraint function at collocation points.
-  **integrand** (*np.ndarray*): Values of the integrand function, evaluated at collocation
   points.
-  **integral** (*float*): Value of the integral function at the optimal solution.

Lagrange multipliers for constraints are also stored:

-  **costate** (*np.ndarray*): Optimal costate values at collocation points.
-  **path_multiplier** (*np.ndarray*): Lagrange multipliers for the path constraint function.
-  **duration_multiplier** (*float*): Lagrange multiplier associated with the duration constraint.
-  **integral_multiplier** (*float*): Lagrange multiplier associated with the integral constraint,
   enforcing equality of integrals over each phase.

The Hamiltonian function, derived from dynamics, integrand, and integral multipliers, is also
available:

-  **hamiltonian** (*np.ndarray*): Values of the Hamiltonian function, evaluated at collocation
   points.

Information from Ipopt Solver
-----------------------------

The `nlp_info` attribute provides detailed information from the Ipopt solver, including:

-  **ipopt_status** (*int*): The Ipopt status code.
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
