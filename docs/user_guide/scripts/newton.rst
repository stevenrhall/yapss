Newton's Minimal Resistance Problem
===================================

For a description of Newton's minimal resistance problem, see the
`JupyterLab notebook documentation <../notebooks/newton.ipynb>`_ for this problem.

This example script has user-defined methods for computing the first and second derivatives of the
objective and continuous functions. User-defined derivatives can be faster to compute than
derivatives computed by automatic differentiation, but not by a large factor. Because for most
problems as much time is spent in the Ipopt solver as in derivative functions evaluation, even
a substantial speedup in derivative evaluation may not result in a significant speedup in the
overall solution time, and so it's almost never worth the effort to implement user-defined
derivatives.

The python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.newton

Functions
---------

.. automodule:: yapss.examples.newton
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/newton.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/newton.txt
   :language: none

Plots
-----

Nosecone Shape for First Formulation
....................................

.. figure:: plots/newton_plot_1.png
   :width: 400pt
   :align: center

Nosecone Shape for Second Formulation
.....................................

.. figure:: plots/newton_plot_2.png
   :width: 400pt
   :align: center

Nosecone Shape for Various Heights
...................................

.. figure:: plots/newton_plot_3.png
   :width: 400pt
   :align: center
