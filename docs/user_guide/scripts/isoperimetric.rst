The Isoperimetric Problem
=========================

For a description of the isoperimetric problem, see the
`JupyterLab notebook documentation <../notebooks/isoperimetric.ipynb>`_ for this problem.

The Python script in this example can be executed from the command line with:

.. code-block:: console

   $ python -m yapss.examples.isoperimetric

Vectors
-------

The classes that name the rows of each vector. A field is listed here when it carries a
docstring of its own, written on the line below it; the ``doc=`` argument of ``field()`` is
metadata for YAPSS and is not read by Sphinx. The empty parentheses in each directive
suppress the inherited ``__init__`` signature, which exists only to refuse construction.

.. autoclass:: yapss.examples.isoperimetric.State()
   :members:

.. autoclass:: yapss.examples.isoperimetric.Control()
   :members:

.. autoclass:: yapss.examples.isoperimetric.Path()
   :members:

.. autoclass:: yapss.examples.isoperimetric.Integral()
   :members:

.. autoclass:: yapss.examples.isoperimetric.Discrete()
   :members:

Functions
---------

.. automodule:: yapss.examples.isoperimetric
   :members:

Code
----

.. literalinclude:: ../../../src/yapss/examples/isoperimetric.py
   :language: python

Text Output
-----------

.. literalinclude:: plots/isoperimetric.txt
   :language: none

Plots
-----

Optimal Solution
................

.. figure:: plots/isoperimetric_plot_1.png
   :width: 400pt
   :align: center

Hamiltonian
...........

.. figure:: plots/isoperimetric_plot_2.png
   :width: 400pt
   :align: center
