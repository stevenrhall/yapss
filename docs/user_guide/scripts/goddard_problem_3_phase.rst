The Goddard Problem (Three Phases)
==================================

For a description of the one-phase Goddard rocket problem, see the `JupyterLab notebook
documentation <../notebooks/goddard_problem_3_phase.ipynb>`_ for this problem.

This problem has three phases because there is a singular arc in the solution. In the
middle phase, the singular arc conditions are imposed as a path constraint. This results
in a significantly better solution than the one-phase solution. See the one phase solution
as a `Python script <goddard_problem_1_phase.rst>`_ or as a `JupyterLab notebook
<../notebooks/goddard_problem_1_phase.ipynb>`_.

This example script has user-defined methods for computing the first and second derivatives of the
objective and continuous functions. User-defined derivatives can be faster to compute than
derivatives computed by automatic differentiation, but not by a large factor. Because for most
problems as much time is spent in the Ipopt solver as in derivative functions evaluation, even
a substantial speedup in derivative evaluation may not result in a significant speedup in the
overall solution time, and so it's almost never worth the effort to implement user-defined
derivatives.

The python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.goddard_problem_3_phase

Functions
---------

.. automodule:: yapss.examples.goddard_problem_3_phase
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/goddard_problem_3_phase.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/goddard_problem_3_phase.txt
   :language: none

Plots
-----

Thrust
......

.. figure:: plots/goddard_problem_3_phase_plot_1.png
   :width: 400pt
   :align: center

Altitude
........

.. figure:: plots/goddard_problem_3_phase_plot_2.png
   :width: 400pt
   :align: center

Velocity
........

.. figure:: plots/goddard_problem_3_phase_plot_3.png
   :width: 400pt
   :align: center

Mass
....

.. figure:: plots/goddard_problem_3_phase_plot_4.png
   :width: 400pt
   :align: center

Hamiltonian
...........

.. figure:: plots/goddard_problem_3_phase_plot_5.png
   :width: 400pt
   :align: center
