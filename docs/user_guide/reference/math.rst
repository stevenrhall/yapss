The ``yapss.math`` Module
=========================

Import the math functions your callbacks use from ``yapss.math``, by name:

.. code-block:: python

    from yapss.math import cos, exp, pi, sin

A function that can be imported from ``yapss.math`` works in callbacks under every differentiation method,
and gives the **same result under each**.
So a problem formulated using only the functions in ``yapss.math`` and standard mathematical operators
represents the same optimal control problem, no matter the method you choose to calculate derivatives.
The exceptions are negative zero and NaN, which a working callback does not produce (see `NaN`_).
On numbers, each function is NumPy's, so the same import serves the rest of a script as well.
That rule is all most problems need; the rest of this page is for when it is not.

``yapss.math`` and NumPy
------------------------

Under ``"auto"``, YAPSS calls each callback with symbolic values, CasADi expressions wrapped by YAPSS,
and traces it to compute exact derivatives.
Otherwise a callback receives numbers: NumPy floats and arrays of them.

Almost every math function that works under ``"auto"``
works whether it is imported from ``yapss.math`` or from ``numpy``.
The one exception is ``where``, which must be imported from ``yapss.math`` for ``"auto"``.
Most of the functions in ``yapss.math`` are NumPy's own:
``from yapss.math import sin`` imports ``numpy.sin``.
The others are YAPSS's versions of NumPy's functions,
and the symbolic values YAPSS passes under ``"auto"``
hand themselves to YAPSS when a NumPy function is applied to them.
``where`` is not a NumPy ufunc,
and NumPy's ``where`` asks the condition for a truth value before anything else,
which a symbolic value does not have.
Under ``"central-difference"``,
NumPy's own ``fmax``, ``fmin``, and ``where`` are a trap of a different kind:
they drop a NaN, which hides a dependency from the sparsity probe (see `NaN`_).
``yapss.math``'s versions do not.

Importing from ``yapss.math`` gives better information: the import succeeding is YAPSS's guarantee
that the function works under every differentiation method.
Asking ``yapss.math`` for another NumPy name raises an ``AttributeError``
that says to import the name from ``numpy``,
as you would for ``linspace`` or ``linalg`` outside a callback.

A NumPy function that ``yapss.math`` does not provide may still work in a callback,
but YAPSS does not promise it.
Under ``"auto"`` most do not.
Under ``"central-difference-full"``, anything that works on NumPy arrays works.
So does it under ``"central-difference"``, unless it drops a NaN or raises on one;
then ``"central-difference-full"``, which does not probe with NaN, is needed (see `NaN`_).

Python's ``math`` module works only on single floats,
so its functions fail on symbols and on arrays; don't use it in callbacks.

Available Functions
-------------------

**Constants**
    - ``pi``

**Trigonometric functions**
    - ``cos``, ``sin``, ``tan``

**Inverse trigonometric functions**
    - ``arccos``, ``arcsin``, ``arctan``, ``arctan2``

    NumPy's shorter names ``acos``, ``asin``, ``atan``, and ``atan2`` are the same functions.

**Hyperbolic functions**
    - ``cosh``, ``sinh``, ``tanh``

**Inverse hyperbolic functions**
    - ``arccosh``, ``arcsinh``, ``arctanh``

    NumPy's shorter names ``acosh``, ``asinh``, and ``atanh`` are the same functions.

**Angular conversion**
    - ``degrees``, ``radians``, ``deg2rad``, ``rad2deg``

**Exponentials and logarithms**
    - ``exp``, ``exp2``, ``expm1``, ``log``, ``log1p``, ``log2``, ``log10``, ``logaddexp``,
      ``logaddexp2``

**Powers and roots**
    - ``cbrt``, ``float_power``, ``hypot``, ``pow``, ``power``, ``reciprocal``, ``sqrt``,
      ``square``

**Arithmetic**
    - ``add``, ``subtract``, ``multiply``, ``divide``, ``true_divide``, ``matmul``

**Modular arithmetic**
    - ``mod``, ``remainder``, ``fmod``, ``floor_divide``

    ``mod`` and ``remainder`` take the sign of the divisor;
    ``fmod`` truncates and takes the sign of the dividend.
    This follows NumPy, and the two conventions differ whenever the operands have opposite signs.

**Sign and magnitude**
    - ``abs``, ``absolute``, ``fabs``, ``copysign``, ``negative``, ``positive``, ``sign``,
      ``conj``, ``conjugate``

    Under ``"auto"``, ``copysign`` treats a ``y`` of ``-0.0`` as positive:
    NumPy reads the floating-point sign bit, which a symbolic expression cannot represent.

**Rounding**
    - ``ceil``, ``floor``, ``trunc``, ``rint``, ``round``

    ``rint`` and ``round`` round half to even, exactly as NumPy does,
    including at the ties and their floating-point neighbors; ``round`` accepts ``decimals``.
    Python's builtin ``round`` is a different function ---
    ``round(x, n)`` rounds the exact decimal value of the float,
    so ``round(2.675, 2)`` is ``2.67`` where ``numpy.round`` gives ``2.68`` ---
    and it does not work on a row of values in a continuous callback under any method.
    Use ``yapss.math.round``.
    Like ``floor``, all of these are piecewise constant:
    zero derivative everywhere and a jump at each switch.

**Extrema**
    - ``maximum``, ``minimum``, ``fmax``, ``fmin``

    ``fmax`` and ``fmin`` are synonyms for ``maximum`` and ``minimum``,
    rather than NumPy's NaN-ignoring versions of them.
    See `NaN`_.

**Conditionals and bounds**
    - ``clip``, ``where``

    ``where`` returns NaN wherever either branch is NaN, unlike NumPy's ``where``; see `NaN`_.

**Comparison functions**
    - ``equal``, ``not_equal``, ``less``, ``less_equal``, ``greater``, ``greater_equal``

**Logical functions**
    - ``logical_and``, ``logical_or``, ``logical_not``, ``logical_xor``, ``invert``

    ``invert`` is the ``~`` operator; see `Comparisons`_.

**Step function**
    - ``heaviside``

**Reductions**
    - ``max``, ``min``, ``amax``, ``amin``, ``all``, ``any``, ``sum``

    On a symbolic argument the reductions fold over every element
    (``max`` is a chain of ``maximum``); only ``sum`` supports the ``axis`` argument there.

    ``abs``, ``all``, ``any``, ``max``, ``min``, ``pow``, ``round``, and ``sum``
    share their names with Python's builtins,
    so import them by name only if you mean to replace the builtins.
    They are rarely needed: for the larger of two values use ``maximum`` and ``minimum``,
    and Python's own ``abs``, ``pow``, and ``sum`` work on symbols.
    Python's ``max``, ``min``, ``all``, and ``any`` raise under ``"auto"`` (see `Conditionals and Bounds`_),
    so to reduce a sequence, write ``yapss.math.max(values)`` after ``import yapss.math``,
    or ``ym.max(values)`` after ``import yapss.math as ym``.

Comparisons
-----------

Comparisons are supported, and so are the corresponding **operators** ---
a comparison in a callback is evaluated exactly under every differentiation method.
Because Python cannot overload ``and``, ``or``, and ``not``,
combine conditions with ``&``, ``|``, and ``~``:

.. code-block:: python

    inside = (y >= -1) & (y <= 1)
    outside = ~inside

Conditionals and Bounds
-----------------------

A symbolic value has no truth value.
A Python ``if``, ``and``, ``or``, or ``not`` on one,
the builtins ``max``, ``min``, and ``sorted``, and the ``in`` operator all ask for one,
and all raise ``TypeError`` under ``"auto"`` --- exactly as CasADi's own symbols do.
Under the other methods they would evaluate the real condition,
so the same callback would transcribe two different problems.
Write the condition as an expression instead:

.. code-block:: python

    from yapss.math import clip, maximum, where

    thrust = clip(u, 0.0, t_max)          # not: max(0.0, min(u, t_max))
    drag = where(v > 0, k * v**2, 0.0)    # not: k * v**2 if v > 0 else 0.0
    speed = maximum(v, v_min)

``clip`` is ``minimum(maximum(x, lo), hi)`` and ``where`` is CasADi's ``if_else``;
both are evaluated exactly under every differentiation method.
Like the comparisons they are built from, neither is differentiable at the switch.

Non-Smooth Functions
--------------------

Note that "supported" is not the same as "smooth" or even "advisable".
``abs``, ``sign``, ``floor``, ``maximum``, ``heaviside``, and the comparisons
are all supported and all non-differentiable somewhere.
They are legitimate modelling tools, but a solver that assumes smoothness may struggle with them.
``equal`` and ``not_equal`` are rarely meaningful for a continuous variable at all.

Comparisons are not differentiable,
and multiplying an expression by a comparison to switch it on and off
makes the discretized problem depend on where the collocation points fall relative to the switch.
Piecewise behavior is normally better expressed with phases,
which is what the multi-phase machinery is for:
put the switch at a phase boundary,
where it becomes an endpoint condition rather than a discontinuity inside a phase.

NaN
---

In ``yapss.math``, NaN contaminates every path it touches:
no function absorbs or selects away a NaN in any of its arguments, with the exceptions below.
Most NumPy functions already behave this way; ``fmax``, ``fmin``, and ``where`` do not,
and ``yapss.math`` overrides them so that they do.
The reason is the sparsity structure under ``"central-difference"``,
which is found by setting one variable to NaN and recording which outputs come back NaN.
A function that could drop the NaN would hide a genuine dependency ---
``where(u > 0, u, 0.0)`` at a point where ``u <= 0`` selects the constant branch,
and so does the probe --- and the Jacobian would be silently incomplete.
A callback should not be producing NaN in the first place;
if it does, write the guard so the invalid branch is never evaluated
(``sqrt(maximum(x, 0.0))`` rather than ``where(x > 0, sqrt(x), 0.0)``,
which evaluates ``sqrt`` at every point either way).
``"central-difference-full"`` does not probe, and under ``"auto"`` the structure is exact,
so none of this applies to them;
under ``"auto"``, ``fmax`` and ``fmin`` return the other operand, following CasADi,
which has no NaN-propagating maximum.

The exceptions:

- The comparison and logical functions return a boolean, which has no derivative to hide.
- ``copysign`` uses only the sign of its second argument,
  and ``heaviside`` uses its second argument only where the first is exactly zero.
- ``power`` follows IEEE arithmetic,
  in which ``power(1.0, nan)`` and ``power(nan, 0.0)`` are ``1.0``, and so does the ``**`` operator.
  The probe runs at the initial guess,
  so if the guess gives a base of exactly 1 or an exponent of exactly 0 at every point,
  the dependency on the other operand is not found.
  Start from a guess that avoids those exact values, or use ``"central-difference-full"``.

When ``"auto"`` Is Not Available
--------------------------------

Not every problem can be formulated using only ``yapss.math`` functions.
A callback that interpolates tabular data, or that calls into external compiled code,
cannot be traced symbolically at all, so ``"auto"`` is unavailable.
The :ref:`minimum time to climb problem </notebooks/minimum_time_to_climb.ipynb>` is such an example:
its aerodynamic and thrust data are irregular tables wrapped in SciPy interpolators,
so it selects ``"central-difference"``.
Such a problem is solved with ``"central-difference"`` or ``"central-difference-full"``,
or with derivatives supplied by hand under ``"user"``.

Unsupported Functions
---------------------

A few NumPy ufuncs inspect the floating-point representation of a number rather than its value,
and have no symbolic equivalent:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Reason
   * - ``nextafter``
     - Steps between adjacent floating-point values.
   * - ``signbit``
     - Reads the floating-point sign bit, including the sign of negative zero.
   * - ``spacing``
     - Returns the distance to the adjacent floating-point value.

``yapss.math`` provides these three names only so that using one says why it is refused:
each raises ``yapss.UnsupportedMathFunctionError``, a subclass of the built-in ``TypeError``,
whatever the argument:

>>> from yapss.math import spacing
>>> spacing(1.0)
Traceback (most recent call last):
    ...
yapss.math.wrapper.UnsupportedMathFunctionError: 'spacing' is not supported in YAPSS callback functions because it returns the distance to the adjacent floating-point value, which has no symbolic equivalent. Callback functions must give the same result under every derivative method. Use 'numpy.spacing' directly if you need it outside a callback.

If one of these worked under central differences and failed under automatic differentiation,
a formulation could come to depend on the differentiation method,
which is exactly what ``yapss.math`` exists to prevent.

.. versionchanged:: 0.3.0
    A real argument raises, as a symbolic one already did. Through 0.2.x it emitted
    ``UnsupportedMathFunctionWarning`` and evaluated as NumPy does.

``yapss.math`` does not provide NumPy's integer functions, such as ``gcd`` and ``left_shift``,
or those with two outputs, such as ``frexp`` and ``modf``.
NumPy's own work on real arguments, under the central-difference methods;
under ``"auto"``, applying one to a callback argument raises ``yapss.UnsupportedMathFunctionError``,
as does any NumPy function that has no symbolic equivalent.

.. versionchanged:: 0.3.0
    ``yapss.math`` provides only the functions listed on this page. Through 0.2.x it also
    re-exported the rest of NumPy.

.. autoexception:: yapss.UnsupportedMathFunctionError
