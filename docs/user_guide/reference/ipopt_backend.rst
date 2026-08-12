How YAPSS Connects to Ipopt
===========================

.. note::

    Most users can skip this page. YAPSS connects to Ipopt automatically, and there is
    nothing you need to configure. This page explains how it chooses, and answers the
    questions that choice tends to raise.

YAPSS solves optimal control problems by converting them into nonlinear programs (NLPs)
and solving them with `Ipopt <https://coin-or.github.io/Ipopt/>`_, a software package for
large-scale nonlinear optimization. How YAPSS reaches Ipopt depends entirely on the
environment it is running in:

* **In a Conda environment**: YAPSS uses cyipopt.
* **Everywhere else** (a pip install, a virtual environment, a system Python): YAPSS uses
  its own interface to the Ipopt library that the CasADi package already ships.

The choice is made automatically by default. The formerly supported overrides are deprecated
and will be removed in version 0.3.0. The rest of this page explains that choice and answers
common questions about it.

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

The ``"central-difference"``, ``"central-difference-full"``, and ``"user"`` differentiation
methods differentiate the user functions differently, but the additional chain-rule step is
shared by all of the differentiation methods, and so is everything downstream of it.

The difficulty is then that CasADi doesn't have a direct interface to Ipopt itself, and the
``nlpsol`` function is an abstraction that can be used to call Ipopt, but also many other
solvers. With some effort, it's possible to use ``nlpsol`` with the YAPSS-generated callback
functions, but features of Ipopt that YAPSS relies on are then no longer reachable. It
scales the NLP through Ipopt's own scaling interface, using the scale factors given in
``problem.scale``, rather than rescaling the problem itself; it installs an intermediate
callback, which is what carries the user's own callback and makes interrupting a solve with
Ctrl-C work; and it reports Ipopt's return status directly rather than a normalized subset
of it. So ultimately, YAPSS needs to access Ipopt directly.

Why are there two interfaces to Ipopt?
--------------------------------------

There are several ways to call a C library such as Ipopt from Python, and YAPSS uses two of
them, because no single one works well in both packaging ecosystems.

**cyipopt** is a Cython wrapper around Ipopt. In a Conda environment it installs with a
single command, along with the Ipopt binary itself, so it is the natural choice there. In a
pip environment it is considerably harder: building the extension needs a C compiler, and
you have to install the Ipopt library on your system yourself. How difficult that is depends
on the operating system — on macOS it is fairly straightforward with Homebrew — but there is
no single-command installation.

**YAPSS's own interface** is written in pure Python using ``ctypes``, the standard library's
foreign function interface, so it needs no compiler. What it does need is a compiled Ipopt
library to call, and one is already present: the CasADi package, which YAPSS depends on in
any case, ships Ipopt inside its distribution. That makes the pip path work with no
additional installation steps at all. This interface is derived from
`mseipopt <https://github.com/cea-ufmg/mseipopt>`_ and, as of version 0.2.0, is bundled
with YAPSS rather than installed separately.

Exception handling
------------------

.. versionchanged:: 0.2.0

    Exceptions raised while Ipopt is calling into YAPSS are now preserved and re-raised
    once ``solve()`` returns, instead of being lost. This applies whether the exception
    comes from a user-supplied callback or from a function YAPSS constructs internally,
    and to both interfaces described above, though the two previously failed
    differently. On the Conda/cyipopt path, an exception raised during a Hessian
    evaluation was discarded silently: Ipopt kept iterating on stale Hessian values and
    reported the run as unconverged, or even as successful, with nothing printed to
    indicate that anything had gone wrong. On the pip path, every callback exception was
    caught, its traceback printed to the console, and status ``-13`` ("Invalid number in
    NLP function or derivative detected") returned to Ipopt -- a real but misleading
    status, since nothing was actually numerically invalid. Either way, no exception
    ever reached ``solve()``'s caller, and an apparently normal ``Solution`` came back
    regardless. Exceptions now surface with their original traceback as a normal,
    catchable Python error on both paths.

Why doesn't YAPSS use the cyipopt I installed?
----------------------------------------------

Outside a Conda environment, YAPSS ignores cyipopt even when it is installed, and uses its
own interface to CasADi's Ipopt instead. This is deliberate.

Loading two independently built copies of Ipopt into one process is unsafe. Each copy
brings its own private OpenMP runtime, and the two can collide and crash the process
part-way through a solve. Outside Conda, a pip-installed cyipopt and CasADi's bundled Ipopt
are exactly two such copies, so YAPSS makes certain that only one is ever loaded.

Conda is the exception, and for a more specific reason than it might appear. It is not that
Conda's runtimes are better behaved, it is that conda-forge's ``cyipopt`` and
conda-forge's ``casadi`` both link against the *same* installed Ipopt and OpenMP packages,
resolved once by Conda's dependency solver. There is only one Ipopt binary in the process,
reached two ways — not two copies coexisting politely. Indeed, one of the reasons that Conda
exists is to solve the problem of setting up binaries in scenarios just like this.

Having cyipopt installed alongside YAPSS in a pip environment does no harm. It simply goes
unused.

Why is CasADi required if I supply my own derivatives?
------------------------------------------------------

CasADi provides the automatic differentiation behind the default ``derivatives.method`` of
``"auto"``, so it is needed for that. But outside a Conda environment it is also where the
Ipopt library itself comes from, which means YAPSS depends on CasADi even for problems that
never use automatic differentiation — with user-supplied derivatives, or with central
differences.

Can I choose which Ipopt library YAPSS uses?
--------------------------------------------

.. deprecated:: 0.2.0

    The ``ipopt_source`` attribute of ``yapss.Problem`` and the ``YAPSS_IPOPT_SOURCE``
    environment variable are deprecated, and will be **removed in version 0.3.0**. After
    that, the interface is determined solely by whether YAPSS is running in a Conda
    environment. Both continue to work in 0.2.x, and now emit a warning when set.

Earlier versions of YAPSS let you select the interface, or point it at an Ipopt library of
your own. That option is being withdrawn because YAPSS cannot verify a library it did not
ship.

For its own interface, YAPSS checks the Ipopt library's compile-time configuration against
the header file that CasADi ships alongside it, and refuses to run if the two disagree —
for instance if Ipopt was built with 64-bit integer indices or in single precision. A
library supplied by you has no matching header to check against, and a mismatch of that
kind does not produce a Python exception. It may or may not crash the process, and if it
does, there is nothing to indicate why. Rather than offer an option that cannot be made safe,
YAPSS is removing it.

If you used previous versions of YAPSS, and were setting ``ipopt_source`` in a program,
deleting that line is all that is required. For ``"default"``, ``"casadi"``, and for
``"cyipopt"`` in a Conda environment, the behavior is identical to what you already had.
The one case that changes is ``"cyipopt"`` requested outside a Conda environment, which is
the unsafe configuration described above.

The linear solver
-----------------

Ipopt spends most of its time in a sparse linear solver, so which one it uses matters more
than any other setting. Different builds of Ipopt offer different solvers, and before
version 0.2.0 YAPSS inherited whatever each build happened to prefer:

* **Conda**, every platform: MUMPS.
* **pip, macOS**: MUMPS --- CasADi's bundled Ipopt has no other option there.
* **pip, Windows and Linux**: SPRAL, because that build includes it and Ipopt selects it
  when it is present.

.. versionchanged:: 0.2.0

    On pip installs, YAPSS now selects MUMPS unless you have chosen a linear solver
    yourself. Previously it accepted whatever the Ipopt build preferred, which was SPRAL
    on Windows and Linux.

As of version 0.2.0, YAPSS selects **MUMPS** on the pip path unless you have chosen a
solver yourself, so every installation now behaves the same way.

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

.. versionchanged:: 0.2.0

    On macOS pip installs, YAPSS now sets ``mumps_pivot_order`` to QAMD unless you have
    chosen an ordering yourself, to avoid a crash in CasADi's METIS library.

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
is a package you installed rather than one YAPSS shipped, and it may be built against
solvers YAPSS cannot detect --- HSL's MA27, for instance, which would usually be the best
choice available. Overriding a solver YAPSS knows nothing about would be presumptuous, and
Conda's Ipopt already defaults to MUMPS in any case.
