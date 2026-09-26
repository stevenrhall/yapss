How YAPSS Connects to Ipopt
===========================

.. note::

    Most users can skip this page. YAPSS connects to Ipopt automatically, and there is
    nothing you need to configure. This page explains how, and answers the questions that
    tend to raise.

YAPSS solves optimal control problems by converting them into nonlinear programs (NLPs)
and solving them with `Ipopt <https://coin-or.github.io/Ipopt/>`_, a software package for
large-scale nonlinear optimization. YAPSS calls Ipopt through its own interface, and it
uses the Ipopt library that CasADi, a YAPSS dependency, already uses:

* **In a pip install** (a virtual environment, a system Python): the Ipopt library bundled
  inside the CasADi package.
* **In a Conda environment**: conda-forge's Ipopt package, which conda-forge's CasADi links
  against.

If CasADi already includes Ipopt, why not use CasADi's solver interface?
------------------------------------------------------------------------

It's true that this is what CasADi is for. Its own documentation is explicit about it:

    Finally, CasADi is not an "optimal control problem solver", that allows the user to
    enter an OCP and then gives the solution back. Instead, it tries to provide the user
    with a set of "building blocks" that can be used to implement general-purpose or
    specific-purpose OCP solvers efficiently with a modest programming effort.

    -- `CasADi user guide <https://web.casadi.org/docs/>`_

YAPSS is one of those specific-purpose solvers. An early development version of YAPSS used
CasADi entirely to formulate the optimal control problem as an NLP that could be solved by
passing it to the ``casadi.nlpsol`` function. Under the hood, CasADi differentiates the
NLP to form the required Jacobians (first derivatives) and Hessians (second derivatives),
and passes the resulting functions to Ipopt to solve. Unfortunately, that approach was far
too slow to be usable, especially for complex problems with multiple phases or dense
discretization meshes. The underlying difficulty is that the expression graph for the
Jacobian grows rapidly with the size of the mesh, and the Hessian graph grows faster than
the Jacobian.

Instead, YAPSS uses the building blocks that CasADi provides to perform automatic
differentiation of the user's functions (at very low computational cost), and vectorizes
the evaluation of those functions at the mesh interpolation points using the
``casadi.Function`` evaluators built from them. But the derivatives of the user functions
alone are not enough --- the NLP derivatives also depend on the collocation method used and
the mesh structure. (For calculus fans, this is just an application of the chain rule for
differentiation.) That calculation is performed by YAPSS internally, to produce the final
callback functions to be passed to Ipopt.

The ``"central-difference"`` and ``"central-difference-full"`` differentiation methods
differentiate the user functions differently, but the additional chain-rule step is
shared by all of the differentiation methods, and so is everything downstream of it.

The difficulty is then that CasADi doesn't have a direct interface to Ipopt itself, and the
``nlpsol`` function is an abstraction that can be used to call Ipopt, but also many other
solvers. With some effort, it's possible to use ``nlpsol`` with the YAPSS-generated callback
functions, but features of Ipopt that YAPSS relies on are then no longer reachable. It
scales the NLP through Ipopt's own scaling interface, using the scale factors the problem
declares, rather than rescaling the problem itself; it installs an intermediate callback,
which is what makes interrupting a solve with Ctrl-C work and records Ipopt's final
convergence measures; and it reports Ipopt's return status directly rather than a
normalized subset of it. So ultimately, YAPSS needs to access Ipopt directly.

How YAPSS calls Ipopt
---------------------

There are several ways to call a C library such as Ipopt from Python. YAPSS's interface is
written in pure Python using ``ctypes``, the standard library's foreign function interface,
so it needs no compiler. What it does need is a compiled Ipopt library to call, and one is
already present wherever CasADi is installed. That makes installation a single step, with
pip or with Conda. The interface is derived from
`mseipopt <https://github.com/cea-ufmg/mseipopt>`_ and is bundled with YAPSS rather than
installed separately.

A ``ctypes`` interface has to agree with the library about the sizes of the values that
cross between them, which a compiled wrapper gets automatically. So before its first solve,
YAPSS checks the Ipopt library's compile-time configuration against the ``IpoptConfig.h``
header installed alongside it, and refuses to run if the two disagree --- for instance, if
Ipopt was built with 64-bit integer indices or in single precision. It also confirms that
exactly one Ipopt library is loaded in the process, and runs a small test problem.

When Ipopt stops without a solution
-----------------------------------

For most statuses Ipopt stops at an iterate and reports it, and YAPSS returns a
``Solution`` built from it, with an ``IpoptConvergenceWarning`` if the status is not a
converged one. For the rest --- too few degrees of freedom (status ``-10``), inconsistent
bounds (``-11``), an invalid option (``-12``), a NaN or Inf from a callback or derivative
during the solve (``-13``), running out of memory (``-102``), and a failure inside Ipopt
itself (``-100``, ``-101``, ``-199``) --- Ipopt has no constraint values or multipliers to
report: it leaves the output arrays as they were passed in, or fills them with zeros. A
solution built from those would look like one and be nothing of the kind, so ``solve()``
raises instead, with Ipopt's own description of the status and, for most, what usually
causes it.

Only one Ipopt per process
--------------------------

Two independently built copies of Ipopt cannot safely share a process. Each brings its own
OpenMP runtime, and the two can collide and crash the process part-way through a solve,
with no Python error to say why.

So when YAPSS first loads Ipopt, it checks that exactly one copy is loaded, and if there
are more it raises ``RuntimeError`` naming every copy it found, rather than risk the crash.
A second copy comes from another package that brings its own Ipopt into the process:
usually cyipopt on a pip install, where it links an Ipopt of its own and CasADi's bundled
Ipopt is a separate copy. Having such a package installed does no harm; importing it into
the same process as a YAPSS solve is what matters. To use both, run them in separate
processes --- separate scripts, or separate notebook kernels.

The check runs once, when YAPSS first loads Ipopt, so it catches a copy imported before
YAPSS's first solve but not one imported after. The hazard is the same either way.

In a Conda environment, conda-forge's cyipopt and CasADi link the *same* Ipopt package, so
there is only one copy, reached two ways, and nothing to refuse.

Why is CasADi required if I use central differences?
----------------------------------------------------

CasADi provides the automatic differentiation behind the default ``derivatives.method`` of
``"auto"``, so it is needed for that. But it is also how YAPSS finds the Ipopt library,
which means YAPSS depends on CasADi even for a problem solved by central differences, which
never uses automatic differentiation.

Can I choose which Ipopt library YAPSS uses?
--------------------------------------------

No. Versions before 0.3.0 let you select the interface, or point YAPSS at an Ipopt library
of your own, through the ``ipopt_source`` attribute of ``yapss.Problem`` or the
``YAPSS_IPOPT_SOURCE`` environment variable.

.. versionchanged:: 0.3.0

    ``ipopt_source`` and ``YAPSS_IPOPT_SOURCE`` were removed, after being deprecated in
    0.2.0. Setting ``problem.ipopt_source`` now raises ``AttributeError``, and a
    ``YAPSS_IPOPT_SOURCE`` still set in the environment has no effect, apart from a
    one-time warning saying so. Deleting either is all that is required.

The option was withdrawn because YAPSS cannot verify a library it did not find through
CasADi. A library supplied by you has no matching header to check against, and a mismatch of
the kind the check catches does not produce a Python exception. It may or may not crash the
process, and if it does, there is nothing to indicate why. Rather than offer an option that
could not be made safe, YAPSS removed it.

The linear solver
-----------------

Ipopt spends most of its time in a sparse linear solver, so which one it uses matters more
than any other setting. Different builds of Ipopt offer different solvers, and each
prefers its own:

* **Conda**, every platform: MUMPS.
* **pip, macOS**: MUMPS --- CasADi's bundled Ipopt has no other option there.
* **pip, Windows and Linux**: SPRAL, because that build includes it and Ipopt selects it
  when it is present.

YAPSS selects **MUMPS** on the pip path unless you have chosen a solver yourself, so every
installation behaves the same way.

The reason is specific to how CasADi builds SPRAL. SPRAL is designed around shared-memory
parallelism, and CasADi compiles it with OpenMP disabled --- a reasonable choice on their
part, since mixing OpenMP runtimes is the same hazard that shapes the rest of this page.
The result is a solver running without the shared-memory parallelism it is designed to use.
In every configuration tested --- three problem sizes on both Windows and Linux, and against
Conda's separately built Ipopt --- MUMPS solved the same problem in less time.

This is not a claim that SPRAL is the weaker solver. Its more careful pivoting often results
in a better solution path, and on the larger problems tested it converged in noticeably fewer
iterations. However, the reduction in iterations was more than offset by the additional
computation required for pivoting. On a properly parallel build, or on problems much larger
than those tested,
the trade may well go the other way.

To use SPRAL, or any other solver your Ipopt provides::

    problem.ipopt_options.linear_solver = "spral"

or place a file named ``ipopt.opt`` in the working directory::

    linear_solver spral

.. warning::

    If the build does not have the solver you asked for, Ipopt does not raise an error. It
    warns and continues with its own choice, so a comparison made without checking can end
    up measuring one solver against itself. Ipopt echoes the options it received when
    ``print_user_options`` is set to ``"yes"``.

The fill-reducing ordering on macOS
-----------------------------------

MUMPS begins by computing a fill-reducing ordering, chosen through ``mumps_pivot_order``
--- Ipopt's name for MUMPS's ``ICNTL(7)``. The values are MUMPS's own:

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - Value
     - Ordering
     - Note
   * - 0
     - AMD
     -
   * - 1
     - user-supplied
     - Requires a permutation YAPSS does not provide.
   * - 2
     - AMF
     -
   * - 3
     - SCOTCH
     - Not present in the builds YAPSS uses.
   * - 4
     - PORD
     - In CasADi's build this is a synonym for METIS; see below.
   * - 5
     - METIS
     - Crashes on macOS in CasADi's bundled Ipopt --- that is, on a pip install. See
       below.
   * - 6
     - QAMD
     - What YAPSS selects on macOS, on a pip install. See below.
   * - 7
     - automatic
     - MUMPS decides, and selects METIS on larger problems.

On macOS, YAPSS sets ``mumps_pivot_order`` to ``6`` (QAMD) unless you have chosen an
ordering yourself. This is not a performance adjustment. The METIS library in CasADi's
macOS build crashes the process --- at any problem size when METIS is requested
explicitly, and above roughly 5000 variables under the default of ``7``, where MUMPS
selects METIS on its own. The defect is in CasADi's build and is fixed in CasADi 3.8.0;
until YAPSS is able to require that version, it avoids the orderings that reach METIS.

Note that ``4`` (PORD) is not an alternative on macOS: in this build it is a synonym for
METIS and fails identically. YAPSS makes no ordering choice on any other platform, where
METIS runs correctly.

.. tip::

    MUMPS reports the ordering it actually used, rather than the one requested, and Ipopt
    prints it at a sufficiently detailed print level::

        MUMPS used permuting_scaling 7 and pivot_order 6.

    That line is the reliable way to confirm an ordering took effect.

In a Conda environment
----------------------

YAPSS sets neither the linear solver nor the ordering in a Conda environment. There, Ipopt
is a package you installed rather than one bundled inside CasADi, and it may be built against
solvers YAPSS cannot detect --- HSL's MA27, for instance, which would usually be the best
choice available. Overriding a solver YAPSS knows nothing about would be presumptuous, and
Conda's Ipopt already defaults to MUMPS in any case.
