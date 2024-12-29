Brachistochrone
===============

For a description of the brachistochrone problem with wall constraints as in this script,
see the `JupyterLab notebook Tutorial Example <../notebooks/tutorial.ipynb>`_.

This script provides a detailed implementation of the brachistochrone problem that includes
user-defined derivatives. (User-defined derivatives are almost never necessary.) For a
script that implements the brachistochrone problem without user-defined derivatives, see the
`example of a minimal implementation of the brachistochrone <brachistochrone_minimal.rst>`_.

This example script has user-defined methods for computing the first and second derivatives of the
objective and continuous functions. User-defined derivatives can be faster to compute than
derivatives computed by automatic differentiation, but not by a large factor. Because for most
problems as much time is spent in the Ipopt solver as in derivative functions evaluation, even
a substantial speedup in derivative evaluation may not result in a significant speedup in the
overall solution time, and so it's almost never worth the effort to implement user-defined
derivatives.

The python script in this example can be executed from the command line with:

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
