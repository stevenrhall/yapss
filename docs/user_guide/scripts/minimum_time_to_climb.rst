Minimum Time to Climb Problem
=============================

For a description of the minimum time to climb problem, see the
`JupyterLab notebook documentation <../notebooks/minimum_time_to_climb.ipynb>`_ for this
problem.

The python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.minimum_time_to_climb

Functions
---------

.. automodule:: yapss.examples.minimum_time_to_climb
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/minimum_time_to_climb.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/minimum_time_to_climb.txt
   :language: none

-----
Plots
-----

Optimal Trajectory
..................

Also shown are contours of energy height (kinetic plus potential energy, divided by mass), and
excess power (power available minus power required for level flight).

.. figure:: plots/minimum_time_to_climb_plot_1.png
   :width: 400pt
   :align: center

Altitude
........

.. figure:: plots/minimum_time_to_climb_plot_2.png
   :width: 400pt
   :align: center

Velocity
........

.. figure:: plots/minimum_time_to_climb_plot_3.png
   :width: 400pt
   :align: center

Flight Path Angle
.................

.. figure:: plots/minimum_time_to_climb_plot_4.png
   :width: 400pt
   :align: center

Mass
....

.. figure:: plots/minimum_time_to_climb_plot_5.png
   :width: 400pt
   :align: center

Angle of Attack
...............

.. figure:: plots/minimum_time_to_climb_plot_6.png
   :width: 400pt
   :align: center

Hamiltonian
...........

.. figure:: plots/minimum_time_to_climb_plot_7.png
   :width: 400pt
   :align: center
