Brachistochrone
===============

A bead slides without friction from the origin to :math:`x = 1` under gravity. The control is
the slope angle of the path, and the objective is the time taken; the answer is a cycloid.

This is the example to read first. It states a complete optimal control problem -- a state
vector, a control, dynamics, bounds, a guess and an objective -- and nothing in it is there
for any reason but the problem. For the shortest complete statement of the same problem, see
the :doc:`minimal implementation <brachistochrone_minimal>`.

The Python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.brachistochrone

Functions
---------

.. automodule:: yapss.examples.brachistochrone
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/brachistochrone.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/brachistochrone.txt
   :language: none

Plots
-----

Optimal Trajectory
..................

.. figure:: plots/brachistochrone_plot_1.png
   :width: 400pt
   :align: center

State Vector
............

.. figure:: plots/brachistochrone_plot_2.png
   :width: 400pt
   :align: center

Control
.......

.. figure:: plots/brachistochrone_plot_3.png
   :width: 400pt
   :align: center

Costate Vector
..............

.. figure:: plots/brachistochrone_plot_4.png
   :width: 400pt
   :align: center

Hamiltonian
...........

.. figure:: plots/brachistochrone_plot_5.png
   :width: 400pt
   :align: center
