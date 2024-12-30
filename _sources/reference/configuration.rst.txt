Configuring the Ipopt Binary Source
===================================

.. note::

    This section is intended for advanced users who need to configure the interface
    between YAPSS and the Ipopt library. Most users can safely ignore this section.

YAPSS solves optimal control problems by first converting them into nonlinear programs
(NLPs) and then using the Ipopt software package for large-scale nonlinear optimization to
solve. For reasons explained below, YAPSS uses two different interfaces to connect to
Ipopt, but for most users, no configuration of the interface is necessary. For users who
need to change the default behavior, or want to get the best possible performance, YAPSS
provides options for configuring the Ipopt interface.

Why Two Interfaces to Ipopt?
----------------------------

Understanding why two interfaces are necessary requires a bit of background. There are
multiple ways to connect C/C++ libraries to Python. For our purposes, we consider two
available Python packages that connect to Ipopt: cyipopt and mseipopt.

**cyipopt** is a Cython wrapper around the Ipopt library. In principle, it works in both
pip and Conda environments. In a Conda environment, cyipopt can be installed with a single
command. Therefore, cyipopt is the default interface to Ipopt in YAPSS when using Conda.

In a pip environment, however, cyipopt can be challenging to install, since it requires
a C/C++ compiler, and the user must install Ipopt on their system. The process is
more or less difficult depending on the operating system. On MacOS, it's actually
fairly straightforward using Homebrew. However, there is no way to install cyipopt
in a pip environment with a single command.

**mseipopt** is a pure Python interface to the Ipopt library using the ctypes package,
which is a foreign function library that allows Python code to call C functions in shared
libraries. Because no compiler is required, mseipopt can be installed in a pip
environment, so long as a compiled Ipopt library is available. As it turns out, the CasADi
package used by YAPSS includes the Ipopt library in its distribution package, which allows
YAPSS to connect to Ipopt using mseipopt without any additional installation steps. This
is the default interface to Ipopt in YAPSS in a pip environment.

Unfortunately, mseipopt is not available as a Conda package. While it's easy to install
mseipopt in a Conda environment using pip install, including mseipopt as a dependency
would make YAPSS incompatible with Conda packaging. As a result, both interfaces are
necessary to make YAPSS available to both Conda and pip users.

Note that in either a pip or Conda environment, it's possible to install both the cyipopt
and mseipopt interfaces. In pip, the installation would be

.. code-block:: bash

    $ pip install yapss cyipopt

with the understanding that the user would have to ensure that the Ipopt library and a Cython
compiler is available on the system. In a Conda environment, the installation would be

.. code-block:: bash

    $ conda install -c conda-forge yapss
    $ pip install mseipopt

Configuration Options
---------------------

The configuration of the Ipopt source and interface is controlled by the ``ipopt_source``
attribute of a ``yapss.Problem`` instance, and by the YAPSS_IPOPT_SOURCE environment
variable. The ``ipopt_source`` attribute can take the following values:

* **"default"**: The default behavior. YAPSS will check for the presence of the
  environment variable YAPSS_IPOPT_SOURCE. If the environment variable matches one of
  the options below, YAPSS will configure the Ipopt interface based on that setting. If the
  environment variable is not set, YAPSS will connect to Ipopt using the cyipopt interface
  if possible, otherwise connect to the Ipopt library in the CasADi package directory
  using the mseipopt interface.

* **"cyipopt":** Connect to Ipopt using the cyipopt interface. If it is unavailable, an
  exception is raised.

* **"casadi":** Connect to Ipopt using the mseipopt interface to the Ipopt library in the
  CasADi package directory. If it is unavailable, an exception is raised. Note that while
  this option is available in a Conda environment if mseipopt is installed, it won't change
  performance by much if at all, since the same Ipopt library is used in both cases.

* **Custom path:** Provide a fully qualified path to the desired Ipopt library as a
  string. YAPSS will attempt to connect to the specified library using the mseipopt
  interface, and an exception is raised if it is unable to do so.

For example:

.. doctest::

    >>> from yapss import Problem
    >>>
    >>> # Instantiate a problem
    >>> problem = Problem(name="Brachistochrone", nx=[3], nu=[1])
    >>>
    >>> # Example 1: Use cyipopt explicitly
    >>> problem.ipopt_source = "cyipopt"
    >>>
    >>> # Example 2: Use Ipopt from CasADi package
    >>> problem.ipopt_source = "casadi"
    >>>
    >>> # Example 3: Use a custom Ipopt library
    >>> problem.ipopt_source = "/path/to/custom/ipopt.so"

Tips for Improving Performance
------------------------------

Based on limited testing, default performance in pip and Conda environments is generally
similar on a given machine (within about 30% on a test suite of example problems). If
performance is an issue, testing both Conda and pip is recommended. In my testing, I found
the pip environment to be faster for MacOS (M1 processor) and Windows (Intel i7), while
the Conda environment was faster for Linux (Intel i7). The difference in performance is
likely due to the different compilers used to build the Ipopt library in the two
environments.

For Windows, pre-built Ipopt binaries are available from the
`Ipopt GitHub site <https://github.com/coin-or/Ipopt/releases>`_. Using these binaries
is about 30% faster than the default performance in either a pip or Conda environment.
To use these binaries, download and unzip the latest release
(Ipopt-3.14.17-win64-msvs2022-md.zip as of this writing), and set the environment
variable

.. code-block:: bash

    $env:YAPSS_IPOPT_SOURCE="C:/path/to/Ipopt-3.14.17-win64-msvs2022-md/bin/ipopt.dll"

and make sure the ``ipopt_source`` attribute of the problem is set to "default".

For both Windows and Linux in the pip environment, the CasADi package Ipopt binary is
compiled with the optional SPRAL linear solver, which in those cases is the default linear
solver. In limited testing, the MUMPS linear solver was about 10% faster than the SPRAL.
To use the MUMPS linear solver, either set the ``ipopt_options.linear_solver`` attribute
of the problem to "mumps", or to make MUMPS the default linear solver, create a file named
``ipopt.opt`` in the working directory with the following content:

.. code-block:: text

    linear_solver mumps