Derivatives
===========

Users have a choice of how YAPSS calculates derivatives required for the solution of the
optimal control problem. The ``derivatives`` attribute of a :class:`~yapss.Problem` optimal control
problem object controls how derivatives are calculated.

The ``derivatives.method`` Attribute
------------------------------------

The ``derivatives.method`` option can take on one of four values:

*  "auto" (default), for automatic differentiation using the casadi package.

*  "central-difference" or "central-difference-full", for derivatives calculated using central
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

It's usually best to use the "auto" method, as it is typically faster and more accurate than the
central difference methods. If central difference methods are required because the casadi package
is unable to calculate the derivatives, it is safer (but slower) to start with the
"central-difference-full" method. Once the problem is working, the "central-difference" method can
be tried to see if it produces the same solution.

The ``derivatives.order`` Attribute
-----------------------------------

Users can also choose whether YAPSS calculates first or second derivatives. The
``derivatives.order`` option can take on one of two values, "first" or "second". When using
automatic differentiation, it’s almost always better to use "second". When using the central
difference method, it can sometimes be advantageous to use only first-order derivatives, because
taking second derivatives is computationally expensive, and numerical second derivatives are
less accurate than numerical first derivatives.

Example
-------

Consider the minimum time-to-climb problem, where the objective is to minimize the time to climb.
Because tabular data is used to represent the aerodynamic performance, automatic differentiation
is not an option. So the differentiation method chosen is "central-difference". Even though
numerical differentiation is used, the derivatives order is set to "second", as that turns
out to be (a little bit) faster than using first-order derivatives.

   >>> from yapss import Problem
   >>> ocp = Problem(name="Bryson Minimum Time to Climb", nx=[4], nu=[1])
   >>> ocp.derivatives.method = "central-difference"
   >>> ocp.derivatives.order = "second"


``Derivatives`` Class Reference
-------------------------------

Below is a complete reference of the ``Derivatives`` class attributes.

.. autoclass:: yapss._private.problem.Derivatives
    :members: