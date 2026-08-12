# YAPSS: Yet Another Pseudo-Spectral Solver

YAPSS is a Python library for formulating and solving optimal-control problems with
pseudospectral methods.

YAPSS provides:

- Legendre-Gauss, Legendre-Gauss-Radau, and Legendre-Gauss-Lobatto collocation methods, using
  a computational approach based on the GPOPS-II algorithm of [Patterson and
  Rao (2014)](https://dl.acm.org/doi/pdf/10.1145/2558904)
- Multiple differentiation approaches: automatic differentiation via CasADi, user-supplied
  derivatives, and central-difference numerical differentiation for problems not amenable to
  automatic differentiation
- Multi-phase problems and segmented meshes
- An interface designed to identify common formulation errors early
- Worked examples as both Python scripts and Jupyter notebooks

## Start here

1. [Install YAPSS](#installation).
2. Work through the [tutorial](https://github.com/stevenrhall/yapss/blob/main/examples/notebooks/tutorial.ipynb) to define and solve a first problem.
3. Browse the [examples](https://github.com/stevenrhall/yapss/tree/main/examples/notebooks) for complete applications.

## Installation

YAPSS requires Python 3.10 or later. Install it into a virtual environment with pip:

```console
$ python -m venv yapss-env
$ source yapss-env/bin/activate
(yapss-env) $ python -m pip install --upgrade pip
(yapss-env) $ python -m pip install yapss
```

Alternatively, install it with Conda:

```console
$ conda create -n yapss-env python=3.10
$ conda activate yapss-env
(yapss-env) $ conda install -c conda-forge yapss
```

### Verify the Installation

Run the HS071 example, a small constrained optimization problem:

```console
(yapss-env) $ python -m yapss.examples.hs071
```

YAPSS is installed correctly if the run finishes, and the output ends with

```text
Objective value
f(x*) = 1.701402e+01

YAPSS solution is correct.
```

The value of the final digits of the objective may vary between platforms and solver versions.

## Where to go next

- **New to YAPSS?** Start with the [tutorial](https://github.com/stevenrhall/yapss/blob/main/examples/notebooks/tutorial.ipynb).
- **Looking for a pattern to adapt?** Browse the [notebook examples](https://github.com/stevenrhall/yapss/tree/main/examples/notebooks) or
  [script examples](https://github.com/stevenrhall/yapss/tree/main/src/yapss/examples).
- **Need API details?** See the reference documentation in the navigation sidebar.
- **Want to contribute?** Contributions are welcome! Read
  [Contributing to YAPSS](https://github.com/stevenrhall/yapss/blob/main/CONTRIBUTING.md).

## License

YAPSS is licensed under the MIT License. See the 
[License](https://github.com/stevenrhall/yapss/blob/main/LICENSE) page
for more information.

## Documentation

The [documentation](https://yapss.readthedocs.io/) is available on Read the Docs.

