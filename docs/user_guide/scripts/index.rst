Python Scripts
==============

This section contains a collection of examples that demonstrate how to use YAPSS to solve optimal
control problems using a Python script. All the examples are part of the YAPSS package examples
module, and can be run from the command line or a Python console. For example, to run the dynamic
soaring example, run the following command in the terminal:

.. code-block:: bash

    python -m yapss.examples.dynamic_soaring

To run the example from a Python console, use the following commands::

    >>> from yapss.examples import dynamic_soaring
    >>> dynamic_soaring.main()

Most of the examples here are presented without much commentary. More detail for each example
can be found in the corresponding JupyterLab notebook example in the
`JupyterLab Notebooks <../notebooks/index.rst>`_. section.

.. toctree::
   :maxdepth: 1

   rosenbrock.rst
   hs071.rst
   brachistochrone_minimal.rst
   brachistochrone.rst
   isoperimetric.rst
   newton.rst
   goddard_problem_1_phase.rst
   goddard_problem_3_phase.rst
   orbit_raising.rst
   dynamic_soaring.rst
   minimum_time_to_climb.rst
   delta_iii_ascent.rst
