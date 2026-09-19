Brachistochrone with User-Defined Derivatives
=============================================

The same problem as the :doc:`brachistochrone <brachistochrone>`, with the first and second
derivatives of the objective and continuous functions supplied by hand rather than obtained by
automatic differentiation. Everything but the four derivative callbacks is the same, and the
answer is the same, so the two can be read side by side.

User-defined derivatives are almost never necessary. They can be faster to compute than
derivatives obtained by automatic differentiation, but not by a large factor, and because for
most problems as much time is spent in the Ipopt solver as in derivative evaluation, even a
substantial speedup in derivative evaluation may not produce a significant speedup overall.

The Python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.brachistochrone_user_derivatives

Functions
---------

.. automodule:: yapss.examples.brachistochrone_user_derivatives
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/brachistochrone_user_derivatives.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/brachistochrone_user_derivatives.txt
   :language: none

Plots
-----

Optimal Trajectory
..................

.. figure:: plots/brachistochrone_user_derivatives_plot_1.png
   :width: 400pt
   :align: center

State Vector
............

.. figure:: plots/brachistochrone_user_derivatives_plot_2.png
   :width: 400pt
   :align: center

Control
.......

.. figure:: plots/brachistochrone_user_derivatives_plot_3.png
   :width: 400pt
   :align: center

Costate Vector
..............

.. figure:: plots/brachistochrone_user_derivatives_plot_4.png
   :width: 400pt
   :align: center

Hamiltonian
...........

.. figure:: plots/brachistochrone_user_derivatives_plot_5.png
   :width: 400pt
   :align: center
