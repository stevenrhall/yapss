Brachistochrone (Minimal Implementation)
========================================

For a description of the brachistochrone problem with constraints as in this script, see the
`JupyterLab notebook Tutorial Example <../notebooks/tutorial.ipynb>`_.

This script provides a minimal implementation of the brachistochrone problem, that is, without
providing user-defined derivatives.  (User-defined derivatives are almost never necessary.)
For a script that implements the brachistochrone problem with user-defined derivatives, see the
`example implementation of the brachistochrone problem with user defined derivatives. <brachistochrone.rst>`_.

The python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.brachistochrone_minimal

Functions
---------

.. automodule:: yapss.examples.brachistochrone_minimal
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/brachistochrone_minimal.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/brachistochrone_minimal.txt
   :language: none

Plots
-----

State Trajectory
................

.. figure:: plots/brachistochrone_minimal_plot_1.png
   :width: 400pt
   :align: center
