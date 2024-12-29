# YAPSS: Yet Another Pseudo-Spectral Solver

YAPSS is a Python package for numerically solving optimal control problems using
pseudospectral methods. Features include:

- Computational approach based on the GPOPS-II algorithm of
  [Patterson and Rao (2014)](https://dl.acm.org/doi/pdf/10.1145/2558904)
- Support for multiple differentiation methods: automatic differentiation via the CasADi
  package, user-defined derivatives, and central difference numerical differentiation for
  problems not amenable to automatic differentiation.
- Choice of collocation method, including Legendre-Gauss (LG), Legendre-Gauss-Radau (LGR),
  and Legendre-Gauss-Lobatto (LGL) options.
- Segmented mesh support, enabling mesh refinement in specific regions. (Automatic mesh
  refinement is not yet available.)
- An API for defining optimal control problems designed to catch common errors
  and provide helpful messages.
- Documentation covering installation, setup, and example usage.
- Examples available as both Python scripts and Jupyter notebooks.

## Quickstart

To get started, install YAPSS  and verify the installation using pip:

```console
$ python -m venv yapss-env
$ source yapss-env/bin/activate
(yapss-env) $ pip install yapss
(yapss-env) $ python -m yapss.examples.isoperimetric
```

If the console output shows a small relative error and a Matplotlib window displays 
a circle, the installation is successful!

For more detailed installation instructions, see the next section.

## Installation

YAPSS supports installation via Conda or pip. It requires Python 3.9 or later..

### Option 1: Using Conda

Create and activate a virtual environment, and install YAPSS:

```console
$ conda create -n yapss-env python=3.9
$ conda activate yapss-env
(yapss-env) $ conda install -c conda-forge yapss
```

If you encounter the following error during installation

```text
PackagesNotFoundError: The following packages are not available from current channels:
  - yapss
```

then YAPSS is not yet available on conda-forge. In this case, install from source as follows:

```console
(yapss-env) $ conda install -c conda-forge numpy casadi scipy mpmath matplotlib cyipopt -y
(yapss-env) $ pip install git+https://github.com/stevenrhall/yapss.git@v0.1.0 --no-deps
```

The ``--no-deps`` flag is important — it prevents pip from reinstalling dependencies that
Conda has already installed, avoiding conflicts. You can delete the tag ``@v0.1.0`` to
install the latest version, or specify a different version tag.

### Option 2: Using Pip

Create and activate a virtual environment, then install YAPSS:

```console
$ python -m venv yapss-env
$ source yapss-env/bin/activate
(yapss-env) $ pip install yapss
```

To install from source:

```console
(yapss-env) $ pip install git+https://github.com/stevenrhall/yapss.git@v0.1.0
```

### Verify the Installation

To verify the installation, run the isoperimetric example:

```console
(yapss-env) $ python -m yapss.examples.isoperimetric
```

The result should be a matplotlib window with a plot of the optimal curve (a circle), and
console output that concludes with something similar to

```text
Maximum area = 0.07957747154594766 (Should be 1 / (4 pi) = 0.07957747154594767)
Relative error in solution = 1.743934249004316e-16
```

If the plot does not display, add `%matplotlib inline` in a Jupyter notebook or set the 
backend with `matplotlib.use('Agg')` for headless environments.

The console output may differ slightly depending on machine precision. Minor deviations in
the final digits are normal, and the relative error should be on the order of machine
precision. If it is, the installation is correct.

Additional examples are available in 

- the examples/notebooks directory
- the src/yapss/examples directory
- the Examples section of the documentation.

## License

YAPSS is licensed under the MIT License. See the LICENSE file for more information.

## Documentation

The [documentation](https://yapss.readthedocs.io/) is available on Read the Docs.

## Contributing

YAPSS is open source — contributions are not only welcome but encouraged. See
[CONTRIBUTING.md](https://github.com/stevenrhall/yapss/blob/main/CONTRIBUTING.md) for details.

