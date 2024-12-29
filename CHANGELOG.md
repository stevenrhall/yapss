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

<!-- 
## [Unreleased]

### Added

### Changed

### Fixed
-->

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
