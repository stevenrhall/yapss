Rosenbrock Function
===================

For a description of the Rosenbrock function minimization problem, see the
`JupyterLab notebook documentation <../notebooks/rosenbrock.ipynb>`_ for this problem.

The python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.rosenbrock

Functions
---------

.. automodule:: yapss.examples.rosenbrock
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/rosenbrock.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/rosenbrock.txt
   :language: none

Plots
-----

Plotted below is a contour plot of the Rosenbrock function, with the minimum indicated by a red dot.
Note that the minimum lies is a long, narrow valley, which makes optimization difficult.

.. figure:: plots/rosenbrock_plot_1.png
   :width: 400pt
   :align: center
