Ipopt Options
=============

YAPSS uses `Ipopt <https://coin-or.github.io/Ipopt/index.html>`_ :footcite:`Wachter:2006` (Interior
Point Optimizer) to solve the NLP problem formulated by YAPSS to represent the optimal control
problem. Ipopt is an open-source software package for large-scale nonlinear optimization. As noted
in the Ipopt documentation,

    Ipopt has many (maybe too many) options that can be adjusted for the algorithm.

    Options are all identified by a string name, and their values can be of one of three
    types: Number (real), Integer, or String. Number options are used for things like
    tolerances, integer options are used for things like maximum number of iterations, and
    string options are used for setting algorithm details, like the NLP scaling method.
    Options can be set through the code that interfaces Ipopt (have a look at the examples
    to see how this is done) or by creating a ipopt.opt file in the directory you are
    executing Ipopt.

Users can set Ipopt options by setting attributes of the ``ipopt_options`` attribute of
instances of the ``Problem`` class. So for example, to set the print level of the Ipopt output, the
user would set the ``ipopt_options`` attribute as follows:

.. code-block:: python

    >>> problem.ipopt_options.print_level = 5

An attribute is used instead of a dictionary to allow for tab completion in an interactive
environment such as the PyCharm IDE. For example, in the PyCharm IDE, typing
``problem.ipopt_options.tol`` will show a list of available options including "tol" as part of
the option name. (There are more than 30 such options!)

A complete description of the Ipopt options is available in the `Ipopt options documentation
<https://coin-or.github.io/Ipopt/OPTIONS.html>`_. For most problems, the default Ipopt options will
be sufficient. The most common options that users may want to change are:

``max_iter``
    Maximum number of iterations. (``max_iter`` :math:`\ge` 0, default: 3000).

``tol``
    Desired convergence tolerance (relative). (``tol`` > 0, default: 1e-8). Determines the maximum
    (scaled) NLP error required for convergence.

``linear_solver``
    Linear solver used for step computations. Determines which linear algebra package is to be used
    for the solution of the augmented linear system (for obtaining the search directions). The Ipopt
    default is "ma27", but for most installations, the MA27 is not available, and Ipopt falls back
    to the "mumps" solver. The available options are:

    - "ma27": use the Harwell routine MA27
    - "ma57": use the Harwell routine MA57
    - "ma77": use the Harwell routine HSL_MA77
    - "ma86": use the Harwell routine HSL_MA86
    - "ma97": use the Harwell routine HSL_MA97
    - "pardiso": use the Pardiso package from pardiso-project.org
    - "pardisomkl": use the Pardiso package from Intel MKL
    - "spral": use the Spral package
    - "wsmp": use the Wsmp package
    - "mumps": use the Mumps package
    - "custom": use custom linear solver (expert use)

    MA27 is part of the Harwell Subroutine Library (HSL), and requires a separate license.

``hsllib``
    Name of library (possibly including path information) containing HSL routines for load at runtime.

    In some installations, Ipopt is built without the HSL linear solvers (ma27, ma57, ma77, ma86, ma97),
    which are compiled separately into a dynamic library that can be loaded at runtime. This option
    allows Ipopt to access those libraries. The default value for this string option is "libhsl.so"
    ("libhsl.dylib" on macOS, "libhsl.dll" on Windows)

``pardisolib``
    Name of library (possibly including path information) containing Pardiso routines
    (from pardiso-project.org) for load at runtime.

``print_user_options``
    Print all options set by the user. ("yes" or "no", default: "no") It can be helpful to print
    the options set by the user to verify that the options are as intended.

``sb``
    Suppress banner. ("yes" or "no", default: "yes") (Not documented in the Ipopt documentation.)
    Suppresses the Ipopt banner at the beginning of the output.

``print_level``
    Output verbosity level. (``print_level``:math:`\ge` 0, default: 5)

.. Note::

    Changing some Ipopt options may not have the expected affect. In particular, YAPSS sets the
    ``hessian_approximation``, ``nlp_scaling_method``, and ``obj_scaling_factor`` options based on
    the problem definition, and setting these options manually will have no effect.

``IpoptOptions`` Class Reference
--------------------------------

.. autoclass:: yapss._private.ipopt_options.IpoptOptions
   :members:
   :no-special-members:
   :no-undoc-members:
