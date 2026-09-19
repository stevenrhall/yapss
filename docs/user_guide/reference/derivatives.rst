Derivatives
===========

Users have a choice of how YAPSS calculates derivatives required for the solution of the
optimal control problem. The ``derivatives`` attribute of a :class:`~yapss._private.problem.Problem` object controls
how derivatives are calculated.

The ``derivatives.method`` Attribute
------------------------------------

The ``derivatives.method`` option can take on one of four values:

*  "auto" (default), for automatic differentiation using the CasADi package. With this method,
   YAPSS calls each callback function a single time, passing symbolic placeholders in place of
   the problem variables it reads --- states, controls, times, integrals, and parameters --- and
   records the expressions the callback builds from them. A placeholder stands for any value the
   variable might take, so it cannot answer a yes-or-no question: a Python ``if``, ``and``,
   ``or``, or ``not`` that tests one raises ``TypeError``, whose message points to
   ``yapss.math.where`` and related functions. Use those functions under every derivative
   method, not only ``"auto"``. Under ``"central-difference"``, an ``if`` in the objective or
   discrete callback runs, but the sparsity of the derivatives is found by evaluating the
   callback at the initial guess, so a variable used only in a branch not taken there is
   treated as having no effect, and the solve can stall or converge to the wrong point; in the
   continuous callback the same ``if`` raises ``ValueError``. Only
   ``"central-difference-full"``, which does not detect sparsity, differentiates an ``if``
   correctly, and even then not at the point where the branch switches.

*  "central-difference" or "central-difference-full", for derivatives calculated using central-
   difference techniques. For the "central-difference" method, the sparsity pattern of the
   derivatives is found automatically, by passing numerical arguments to the user-defined
   callback functions that include the ``nan`` (not a number) floating point value in specific
   elements of the decision variable arrays. The ``nan`` values propagate through the calculations
   and can be used to determine the sparsity patterns of the derivatives. This process works well
   in many cases; however, in some cases this method is unreliable for finding the sparsity pattern.
   (For example, if a C, C++, or Fortran function is called from Python, the ``nan`` values may not
   propagate through the function call as expected.) In these cases, the "central-difference-full"
   method can be used, which calculates all the derivatives by central difference, without
   attempting to find the sparsity pattern. This method is slower than the "central-difference"
   method but is more reliable.

*  "user", in which case the user must supply the first and perhaps second derivatives.

.. warning::

    Under ``"central-difference"``, the sparsity pattern is found by setting one variable to
    ``nan`` and seeing which outputs come back ``nan``, so any function that *absorbs* a
    ``nan`` hides a dependency and yields an incomplete Jacobian --- a wrong answer that Ipopt
    reports as optimal. NumPy's ``where``, ``fmax``, ``fmin`` and the ``nan*`` reductions
    (``nanmax``, ``nansum``, ...) all discard a ``nan`` in the branch they do not select, and
    so do the builtin ``min`` and ``max`` on scalars. Use the ``yapss.math``
    versions, which keep a ``nan`` from either branch, or the ``"central-difference-full"``
    method, which does not detect sparsity. (``numpy.clip`` and ``numpy.maximum`` propagate
    ``nan`` and are safe.)

It's usually best to use the "auto" method, as it is typically faster and more accurate than the
central-difference methods. If central-difference methods are required because the CasADi package
is unable to calculate the derivatives, it is safer (but slower) to start with the
"central-difference-full" method. Once the problem is working, the "central-difference" method can
be tried to see if it produces the same solution.

The ``derivatives.order`` Attribute
-----------------------------------

Users can also choose whether YAPSS calculates first or second derivatives. The
``derivatives.order`` option can take on one of two values, "first" or "second". When using
automatic differentiation, it’s almost always better to use "second". When using the central-
difference method, it can sometimes be advantageous to use only first-order derivatives, because
taking second derivatives is computationally expensive, and numerical second derivatives are
less accurate than numerical first derivatives.

Example
-------

Consider the minimum time-to-climb problem, where the objective is to minimize the time to climb.
Because tabular data is used to represent the aerodynamic performance, automatic differentiation
is not an option. So the differentiation method chosen is "central-difference". Even though
numerical differentiation is used, the derivative order is set to "second", as that turns
out to be (a little bit) faster than using first-order derivatives.

   >>> from yapss._legacy import Problem
   >>> ocp = Problem(name="Bryson Minimum Time to Climb", nx=[4], nu=[1])
   >>> ocp.derivatives.method = "central-difference"
   >>> ocp.derivatives.order = "second"


``Derivatives`` Class Reference
-------------------------------

Below is a complete reference of the ``Derivatives`` class attributes.

.. autoclass:: yapss._private.problem.Derivatives
    :members:
    :no-special-members:
    :no-undoc-members:
