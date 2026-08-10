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

### Added

- `Problem.solve()` now emits an `IpoptConvergenceWarning` when Ipopt does not report a
  converged solution. A `Solution` is still returned in every case, but an unconverged
  run — one that hit `max_iter`, for example — previously produced a plausible-looking
  trajectory with nothing to indicate that it satisfied no convergence criteria. Ipopt
  statuses 0 (optimal), 1 (solved to acceptable level) and 6 (feasible point for a
  square problem) are treated as success and remain silent; status 1 in particular is a
  normal outcome when tolerances are pushed hard. The warning category is exported as
  `yapss.IpoptConvergenceWarning`, so it can be silenced with
  `warnings.filterwarnings("ignore", category=yapss.IpoptConvergenceWarning)` or
  escalated to an exception with `"error"` in place of `"ignore"`. Projects that run
  their test suites with `-W error` will newly see failures on unconverged solves.
- Deterministic resolution of the Ipopt library on pip installs. YAPSS now asks the
  dynamic loader which Ipopt file CasADi actually loaded and opens exactly that one.
  After loading it verifies that only one Ipopt is mapped into the process, and
  checks the library's compile-time configuration against the `IpoptConfig.h` header
  that ships in the CasADi wheel, refusing to run against a 64-bit-index or single-precision
  build.
- The documentation build now generates `llms.txt` and `llms-full.txt` alongside the
  HTML output, giving LLM tools a Markdown-formatted index and a single-file version
  of the full user guide.

### Changed

- `bounds.phase[p].zero_mode` is now private (`_zero_mode`). It was never meant for
  users to set — it exists for internal testing and research use only.
- For pip installs, YAPSS now selects the MUMPS linear solver unless the user has set
  `ipopt_options.linear_solver`. This makes every YAPSS installation behave the same
  way: Conda's Ipopt already defaults to MUMPS on every platform, and CasADi's bundled
  Ipopt does so on macOS, but on Windows and Linux that build enables SPRAL and selects
  it by default. CasADi compiles SPRAL with OpenMP disabled, so the solver runs without
  the shared-memory parallelism it is designed around; in every configuration measured
  — three problem sizes on both Windows and Linux, and against Conda's own separately
  built Ipopt — MUMPS completed the same problem in less time. This is a consequence of
  how CasADi builds SPRAL rather than a judgment about the solver: SPRAL's pivoting earns
  a better solution path, converging in fewer iterations on the larger problems tested,
  but without OpenMP it cannot amortize the cost of producing it. This is a change in
  behavior for pip installs on Windows and Linux. Setting
  `ipopt_options.linear_solver` to `"spral"` restores the previous selection.
- For pip installs, the `mseipopt` Ipopt interface is now bundled with YAPSS rather
  than installed as a dependency. The now-unused `mseipopt` package can be uninstalled.
- Outside a Conda environment, `cyipopt` is no longer imported even when installed.
  Importing it opens a second, independently built Ipopt binary alongside the one
  CasADi bundles. Having `cyipopt` installed is harmless; YAPSS simply does not use it
  there.
- In a Conda environment, a missing `cyipopt` is now reported at import with the
  `conda install` command needed to fix it, rather than a bare `ModuleNotFoundError`.
- Ipopt status messages are now transcribed verbatim from the `EXIT:` lines Ipopt itself
  prints, so that `solution.nlp_info.ipopt_status_message` — and the `Status Message:`
  line of `print(solution)` — match the solver output directly above them. Five entries
  change: statuses 0 and 3 gain the trailing period Ipopt prints; -12 and -100 were the
  `ApplicationReturnStatus` enumeration names rather than messages, and are now
  "Invalid option encountered." and "Some uncaught Ipopt exception encountered."; and
  -101, for which Ipopt prints no `EXIT:` line at all, is now described as an exception
  not raised by Ipopt. Code that compares these strings exactly will need updating;
  `solution.nlp_info.ipopt_status` is the stable way to test the outcome.
- The bundled examples no longer request `tol = 1e-20`. That value cannot be met, so the
  solves were terminating through Ipopt's "acceptable level" fallback rather than on the
  tolerance requested, at accuracies between 1e-9 and 1e-14 depending on the problem.
  They now use Ipopt's default tolerance. `isoperimetric` keeps a tight tolerance of
  1e-14, where it is attainable and demonstrates the accuracy of the method; its script
  and notebook, which had drifted into solving different formulations, now agree.

### Deprecated

- `Problem.ipopt_source` and the `YAPSS_IPOPT_SOURCE` environment variable, both to be
  **removed in 0.3.0**, after which the Ipopt backend is determined solely by whether
  YAPSS is running in a Conda environment or not. Both methods of setting the Ipopt source
  continue to function through 0.2.x, and now emit a `DeprecationWarning` or `FutureWarning`.
  In the case of an explicit library path, `FutureWarning` is used to ensure that a warning
  is emitted before a potential crash; `DeprecationWarning` is suppressed by default
  outside `__main__`. Removing any line setting the `ipopt_source` attribute and removing
  the `YAPSS_IPOPT_SOURCE` environment variable eliminates the warning and will avoid an
  `AttributeError` in 0.2.0.

### Fixed

- Fixed synchronization of the physical time vector passed to numeric continuous
  callbacks. Some internal derivative evaluations copied a new NLP decision vector
  without rebuilding time from the phase endpoints and mesh nodes, leaving callbacks
  with stale time values. For time-dependent dynamics and integrands, this could produce
  incorrect derivative sparsity or central-difference derivatives and, in turn, incorrect
  solutions. Numeric continuous arguments now update decision variables and time together
  for the LG, LGR, and LGL spectral methods.
- Fixed `solution.phase[p].control_multiplier`, which was not a Lagrange multiplier at
  all: it was computed from the primal control value rather than from Ipopt's bound
  multiplier, and was missing the `1 / mesh.w[p]` quadrature-weight factor applied to
  the other continuous multipliers. This was wrong in every previous release.
- Fixed a race in the documentation build. The new `llms.txt`/`llms-full.txt`
  generation runs a second Sphinx build in a subprocess, in parallel with the primary
  one by default; both load `conf.py`, and without a guard, the notebook- and
  plot-regenerating build steps would run a second time concurrently with the primary
  build reading their output.
- Fixed a crash risk on pip installs. On Linux and macOS, YAPSS could load a second copy
  of Ipopt alongside the one CasADi bundles, and two independently built copies can fail
  mid-solve. See the resolution change under Added.
- Fixed the root cause of the NumPy 2.5 import failure (previously only worked around via
  a `numpy<2.5` ceiling in 0.1.1). NumPy 2.5 rewrote `numpy.typing.NDArray` using PEP 695
  `type` statements, which cannot be subclassed directly, so `ContinuousArray` now
  subclasses `np.ndarray` directly instead. The ceiling has been removed, and users on
  Python 3.12+ can now install NumPy 2.5.
- Fixed a segmentation fault on macOS for pip installs. CasADi's bundled Ipopt crashes
  inside its METIS 4.0 library, which MUMPS calls to compute the fill-reducing ordering.
  The crash occurs at any problem size when METIS is requested explicitly, and above
  roughly 5000 variables by default, where MUMPS selects METIS on its own. The defect is
  in the CasADi build and is fixed upstream in CasADi 3.8.0; until YAPSS can require that
  version, YAPSS sets `mumps_pivot_order` to QAMD on macOS, and only when the user has
  not selected an ordering. Note that `pord` is not an alternative: in this build it is a
  synonym for METIS and crashes identically. Linux, Windows, and Conda installations are
  unaffected and are left alone.
- Fixed logging configuration, which had been setting the level and handler on a single
  module's logger rather than on the `yapss` package logger. Messages from other modules
  were therefore discarded, and `YAPSS_LOGGING=DEBUG` silently reported nothing from the
  Ipopt library resolver.
- Fixed the `Bool` type width in the bundled Ipopt interface for pip installs. Ipopt
  changed `Bool` from `int` to `bool` in 3.14, and the interface still declared it four bytes
  wide. The width is now derived from the Ipopt version in the shipped header rather
  than assumed. In practice this had not misfired, but a failed option-setting call
  in principle could have been read as success and silently ignored.
- Fixed `Guess.from_solution`, which assigned a solution's `control` array directly onto
  the guess's `time` grid. `control` is defined on `time_c`, which only has the same
  length as `time` for the lgl spectral method; for lgr and lg, `from_solution` either
  raised on `guess.validate()` or produced a misaligned guess. `control` is now
  interpolated (with extrapolation at the endpoints) from `time_c` onto `time` before
  being assigned, so `from_solution` works for all three spectral methods.

---

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
