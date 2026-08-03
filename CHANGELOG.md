# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project
adheres to [PyPA versioning
guidelines](https://packaging.python.org/en/latest/specifications/version-specifiers/).
YAPSS is currently in initial development (0.x), and hence the public API should not be
considered stable. YAPSS will follow a predictable versioning policy during 0.x development:

- Breaking changes or major new features may occur at minor version changes (e.g., from 
  0.3.x to 0.4.0).
- Patch releases (e.g., 0.3.1, 0.3.2) within the same minor version are backwards-compatible
  and will not introduce breaking changes.
- Users can pin to a specific minor version (e.g., yapss>=0.3.0,<0.4.0) to avoid unexpected 
  changes, but should expect significant updates when upgrading to a new minor version.

## [Unreleased]

### Fixed

- Fixed the root cause of the NumPy 2.5 import failure (previously only worked around via
  a `numpy<2.5` ceiling in 0.1.1). NumPy 2.5 rewrote `numpy.typing.NDArray` using PEP 695
  `type` statements, which cannot be subclassed directly; `ContinuousArray` now subclasses
  `np.ndarray` directly instead. The `numpy<2.5` ceiling remains for now because NumPy 2.5
  requires Python 3.12+, which conflicts with YAPSS's `>=3.10` support — raising the floor
  is deferred to a future release.

### Changed

- Advanced the `mypy` dev/CI tooling from 1.13.0 to 1.20.2, and its `numpy` constraint from
  `<2.0.0` to an exact `2.4.6` pin. Also moved the `mypy` tox environment's `basepython` from
  Python 3.10 to 3.13 to match the pre-commit hook, since NumPy dropped Python 3.10 wheels
  entirely as of 2.4.6 was unreachable from a 3.10-hosted environment (it caps at 2.2.6). No
  user-facing effect; `[tool.mypy] python_version = "3.10"` still governs the actual
  type-checking target regardless of which Python hosts the checker.
- Switched `plt.xlim(...)`/`plt.ylim(...)` calls in the example scripts from list to tuple
  arguments (e.g. `plt.xlim([0, 1])` → `plt.xlim((0, 1))`), matching matplotlib's type stubs.
  No behavior change.

---

## 0.1.0 - 2024-12-28

### Added

Initial release of the software package. Features include:

- Computational approach based on the GPOPS-II algorithm of Patterson and Rao (2014). 
- Support for multiple differentiation methods: automatic differentiation via the CasADi package,
  user-defined derivatives, and central difference numerical differentiation for problems not
  amenable to automatic differentiation.
- Choice of collocation method, including Legendre-Gauss (LG), Legendre-Gauss-Radau (LGR),
  and Legendre-Gauss-Lobatto (LGL) options.
- Segmented mesh support, enabling mesh refinement in specific regions. (Automatic mesh refinement is
  not yet available.)
- An API for defining optimal control problems designed to catch common errors and
  provide helpful messages.
- Documentation covering installation, setup, and example usage.
- Examples available as both Python scripts and Jupyter notebooks.
- Nearly complete test coverage for all modules.

## 0.1.1 - 2026-08-02

### Removed

- Stopped publishing documentation to GitHub Pages. The Pages copy was unreferenced by the
  README, package metadata, and PyPI listing (all of which already pointed at
  [readthedocs.io](https://yapss.readthedocs.io/)), so it had gone stale without anyone noticing.
  The `gh-pages` branch now redirects to Read the Docs instead of serving old content.

### Changed

- Updated the supported Python versions to 3.10 and newer in `pyproject.toml` and `tox.ini` to match
  the supported test matrix. This change is consistent with Python 3.9 having reached its end-of-life
  phase in October 2025.
- The `notebook` extra no longer installs `black`, `isort`, or `jupyterlab_code_formatter` by
  default. These remain available via the `dev` extra for contributors; installing `yapss[notebook]`
  no longer forces an opinionated formatting setup on end users.

### Fixed

- Restricted NumPy to versions earlier than 2.5. NumPy 2.5 causes YAPSS 0.1.0 to fail on import due
  to changes in NumPy's typing implementation.
- Restricted CasADi to versions 3.6.0 through 3.7.2. `mseipopt` hard-wires a path into CasADi's
  bundled IPOPT library, which is not part of CasADi's public interface and could change in any
  release; pinning avoids a known crash risk until `mseipopt` is no longer the default solver
  backend for pip installations.
- Corrected outdated installation instructions in the README and documentation. A conda-forge
  workaround for a `PackagesNotFoundError`, needed only before YAPSS was published on
  conda-forge, no longer applies and has been removed; the example `conda create` command now
  specifies Python 3.10 instead of the no-longer-supported 3.9.
- Fixed a spurious matplotlib warning ("Ignoring fixed x/y limits to fulfill fixed data aspect
  with adjustable data limits") produced when running the `brachistochrone`,
  `brachistochrone_minimal`, and `newton` example scripts. `axis("equal")` can silently override
  explicitly set axis limits; switched to `axis("scaled")`, which respects them.
