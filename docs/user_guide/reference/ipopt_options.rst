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

    problem.ipopt_options.print_level = 5

An attribute is used instead of a dictionary to allow for tab completion in an interactive
environment such as the PyCharm IDE. For example, in the PyCharm IDE, typing
``problem.ipopt_options.tol`` will show a list of available options including "tol" as part of
the option name. (There are more than 30 such options!)

Each option is of one of Ipopt's three kinds, and the value is checked against the kind when it
is assigned: an Integer option takes an ``int`` (a NumPy integer is accepted and converted), a
Number option takes a ``float`` or an ``int``, and a String option takes a ``str``. A ``bool`` is
refused everywhere, since no Ipopt option is boolean; the yes/no options take the strings
``"yes"`` and ``"no"``. A value of the wrong kind raises ``TypeError`` at the assignment, and a
NaN for a Number option raises ``ValueError``, since no comparison with NaN is true and Ipopt
checks values against a range.

Which options exist, and which values they take, depends on the Ipopt build: the pip wheel's
Ipopt and conda-forge's are different builds of different versions. So YAPSS passes every
option to Ipopt, which is the only authority on what it accepts and validates them when the
problem is solved. If Ipopt refuses one, YAPSS warns with
:class:`~yapss.IpoptOptionSettingWarning`: the option is not applied, and the solve continues
with Ipopt's default. Ipopt reports only that it refused the option, and its console output says
why; the warning adds a hint from a table generated from Ipopt's documentation for one release
--- for a name close to a documented one, the likely intent (``max_iters`` suggests
``max_iter``), and for a documented option, that this build might not provide it or might not
accept the value.

A type checker reports a misspelled option name before the script runs, since every documented
option is annotated. The warning can be silenced or turned into an error with
:func:`warnings.filterwarnings`, as for :class:`~yapss.IpoptConvergenceWarning`.

.. versionchanged:: 0.3.0

    A value Ipopt refuses now raises when it is outside what Ipopt documents for that option;
    before, every refusal was a warning and the solve continued with the default.

Ipopt can write its own log to a file, which is the way to keep solver output when
``print_level`` is 0::

    problem.ipopt_options.output_file = "ipopt.log"
    problem.ipopt_options.file_print_level = 5

Ipopt's documentation says these work only when read from an ``ipopt.opt`` file. That is not
true of the interface YAPSS uses: both take effect when set here.

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
    Suppress banner. ("yes" or "no", default: "no") (Not documented in the Ipopt documentation.)
    Suppresses the Ipopt banner at the beginning of the output.

``print_level``
    Output verbosity level. (``print_level``:math:`\ge` 0, default: 5)

YAPSS otherwise tries not to be opinionated about Ipopt options, but makes two exceptions.
First, the default value of ``mu_strategy`` is ``"adaptive"`` rather than Ipopt's own
default of ``"monotone"``. The YAPSS test suite runs about 30% slower using the Ipopt
default, and we have found that Ipopt sometimes fails to converge on difficult problems
with the monotone strategy. Second, the default value of ``check_derivatives_for_naninf``
is ``"yes"`` rather than ``"no"``. Without the check, Ipopt passes a NaN or infinite
Jacobian or Hessian entry to its linear solver, which can crash the Python process; with
it, Ipopt stops with status -13 ("Invalid number in NLP function or derivative
detected"), and ``problem.solve()`` raises ``ValueError`` saying so. The check costs one pass
over the derivative values per evaluation.
Separately, ``problem.solve()`` raises ``ValueError`` before starting Ipopt if the
objective, constraints, or their first derivatives are not finite at the initial guess,
naming the quantities involved. Unlike the reserved options below, both are normal
options and can still be set to any value through ``problem.ipopt_options``.

The following options are reserved: YAPSS determines them from the problem configuration,
and attempting to set them directly through ``ipopt_options`` raises a ``ValueError``
immediately, rather than being silently overridden later.

``hessian_approximation``
    Controlled by ``problem.derivatives.order``. When the derivative order is
    ``"first"``, YAPSS uses ``"limited-memory"``; otherwise, it provides the
    Hessian needed by Ipopt. Set the derivative order through
    ``problem.derivatives`` instead.

``warm_start_init_point``
    YAPSS does not yet support warm starts, because it does not provide the multiplier
    data that Ipopt needs for one. It therefore sets this option to ``"no"`` for every
    solve.

``nlp_scaling_method``
    YAPSS supplies NLP scaling data through Ipopt's scaling interface and sets this
    option to ``"user-scaling"``. Set scaling through ``problem.scale`` instead. This
    restriction may be relaxed in a future YAPSS release.

``obj_scaling_factor``
    YAPSS manages objective scaling internally. Set the sign through ``problem.sense``
    and the magnitude through ``problem.scale.objective`` instead.

For example, trying to set ``hessian_approximation`` directly raises an error:

.. doctest:: group1
    :options: +IGNORE_EXCEPTION_DETAIL

    >>> from yapss import Problem
    >>> problem = Problem(name="Test", nx=[1])
    >>> problem.ipopt_options.hessian_approximation = "exact"
    Traceback (most recent call last):
        ...
    ValueError: 'hessian_approximation' is managed by YAPSS and cannot be set directly.
    YAPSS chooses this based on 'problem.derivatives.order'; set that instead.

``IpoptOptions`` Class Reference
--------------------------------

.. autoclass:: yapss._private.ipopt_options.IpoptOptions
   :members:
   :no-special-members:
   :no-undoc-members:

``IpoptOptionSettingWarning`` Class Reference
----------------------------------------------

.. autoexception:: yapss.IpoptOptionSettingWarning
