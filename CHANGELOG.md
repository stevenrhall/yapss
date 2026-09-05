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

### Changed

- The user's continuous functions are evaluated once per Ipopt iterate. The constraint,
  Jacobian, and Hessian callbacks previously each evaluated them independently at the same
  point, so the function ran three times per iterate and its Jacobian twice; they now share
  one evaluator. No change in results; the golden tests pin every callback value. The
  example suite runs about 6% faster, and central-difference solves save one Jacobian stencil
  per iterate (Delta III: 6.2 s to 5.8 s).

### Fixed

- `yapss.math` now gives the same answer under every derivative method for the functions that
  numpy evaluates by taking a truth value. Before this release, `clip`, `where`, `all`, `any`,
  and the `max`/`min` reductions silently returned their first argument on a symbolic value —
  numpy's object-dtype loops compare and then call `bool()`, and a symbol had no `bool()` to
  refuse — so a callback using any of them transcribed a different problem under `"auto"` than
  under the finite-difference methods, and nothing raised. Each is now implemented symbolically
  (`clip` as `minimum(maximum(x, lo), hi)`, `where` as CasADi's `if_else`, the reductions as
  folds), and the transcription is pinned to agree between `"auto"` and `"central-difference"`
  for every one of them. `where` is newly exported.
- A symbolic value now refuses to give a truth value, as CasADi's own `SX` does. A Python `if`,
  `and`, `or`, or `not` on a callback argument, the builtins `max`, `min`, and `sorted`, and the
  `in` operator all raise `TypeError` under `"auto"`, with a message naming the symbolic
  spellings. This was scheduled for 0.3.0 as a behavior change; it ships now as a fix, because
  the behavior it replaces was a silently wrong answer — `x if x > 2 else -x` transcribed to
  `+x` under `"auto"` and `-x` under central differences.
- Sixteen exported functions raised `TypeError` on a *scalar* symbolic argument (a phase's
  `final_time`, a `parameter[i]`, an `integral[i]`) while working on an array (`state[i]`):
  `abs`, `absolute`, `cbrt`, `conj`, `conjugate`, `deg2rad`, `degrees`, `exp2`, `log2`,
  `negative`, `positive`, `rad2deg`, `radians`, `reciprocal`, `square`, and `trunc`. The two
  paths dispatched differently, and only the array path had YAPSS's own implementations; the
  scalar path handed the raw symbol to numpy, which on casadi 3.7.2 has none of these. Both
  paths now resolve through one table of CasADi implementations.
- A scalar symbolic value times an array — `final_time * final_state`, `parameter[0] *
  initial_state` — no longer collapses into a single wrapper holding a CasADi matrix, which
  failed three layers later inside CasADi's derivative code with an unhelpful message when
  assigned to a discrete constraint. The result is an ordinary symbolic array, in either
  operand order, for float arrays and 2-D arrays too, and augmented assignment on a symbolic
  array (`array += w`) writes each element in place.
- Reductions of a symbolic array (`sum`, `mean`, `linalg.norm`, `dot`) return a symbolic scalar
  rather than a 0-d array, and `sum` of a symbolic scalar works.
- `rint` and `round` are supported, exactly. 0.2.2 refused `rint` on the belief that numpy's
  half-to-even tie rule could not be reproduced symbolically; it can, from `floor` and the
  fractional part (which are exact in double precision) and a conditional, without the
  `floor(x + 0.5)` addition that every simpler scheme gets wrong half an ulp below a tie. Both
  are checked bit-for-bit against numpy at every half-integer in a range, at its floating-point
  neighbors, and for `round` at the classic `decimals` cases (`round(2.675, 2)` is `2.68`, as
  numpy has it, not the `2.67` of Python's builtin, which rounds the exact decimal value; for
  that reason the builtin `round(x, n)` on a symbol is refused, while `round(x)` works). On
  casadi 3.8.0,
  `round` had previously gone through to CasADi's own rounding, which is half-away-from-zero
  and disagreed with numpy at every tie. `nextafter`, `signbit`, and `spacing` remain refused;
  those really do read the bit pattern.
- In-place arithmetic on a symbolic value under the `"auto"` derivative method — a user callback
  that accumulates a term at a time, such as `d = 0.0` followed by `d += ...` in a loop — no
  longer risks unbounded recursion. `NDArrayOperatorsMixin` spells every augmented assignment as
  `ufunc(self, other, out=(self,))`, so the internal `SXW` wrapper reached its `__array_ufunc__`
  with an `SXW` in `out=` even though nothing in the callback mentions it. Only the positional
  inputs were unwrapped, so the untouched `out=` was forwarded to the ufunc, and because numpy's
  dispatch considers `out` operands as well as inputs, that re-entered the same method with
  identical arguments. The chain terminated only while CasADi handled the inner
  call and ignored `out=`; whenever CasADi returned `NotImplemented` instead, numpy fell through
  to the same override again and recursed without bound. That was reachable on the currently
  pinned `casadi<=3.7.2` — `np.negative`, `np.absolute`, `np.square` and `np.reciprocal` with an
  explicit `out=` all raised `RecursionError` there — and, under CasADi 3.8's opt-in
  `GlobalOptions.setNumpyMode(1)`, for every ufunc including the augmented-assignment form. An
  `SXW` wraps an immutable CasADi value and can never be an output buffer, so `out=` is now
  dropped rather than forwarded, and passing it can no longer change the outcome of a call. No
  correct program was previously made wrong: the calls that recursed already raised `TypeError`
  without `out=`, and they now raise that same `TypeError`. The pre-existing rule that an explicit
  `out=` is accepted and ignored is unchanged.
- `Problem.validate()` now rejects initial-state or final-state bounds that do not overlap the
  state bounds. The NLP bound on a boundary state is the intersection of the two (the larger of
  the lower bounds and the smaller of the upper bounds), so a pair that was each consistent on
  its own, such as `state` in `[5, 10]` and `initial_state` in `[-1, 0]`, passed validation and
  reached Ipopt as a lower bound above the upper bound, with no diagnostic pointing at the
  cause. The message names the phase, the offending bound, and the indices. Bounds that touch
  at a single point (a fixed boundary state inside the state bounds) remain valid.

## [0.2.2] - 2026-08-23

### Changed

- The NLP Jacobian and Hessian are now assembled from single plans: a sequence of blocks that each
  own their index pairs together with the closure producing the matching values, rather than a
  structure builder and an evaluator kept in step by hand. The structure fold moved inside the
  plans, so nothing mutates the NLP's derivative surface after construction. There is no
  user-visible change in behavior. Hessian evaluation is about 10% faster, and the module that
  had held both halves shrank from 1335 lines to 505. The assembly is pinned by golden tests
  generated from the pre-refactor implementation, covering every arm of the assembler across all
  four derivative methods.

### Deprecated

- Setting both orders of one variable pair in a user-defined Hessian callback — for example both
  `hessian[("f", 0), ("x", 2), ("u", 0)]` and `hessian[("f", 0), ("u", 0), ("x", 2)]` — now emits
  `yapss.MirroredHessianPairWarning` and will raise `ValueError` **in 0.3.0**. The two keys name
  the same second partial derivative. They currently assemble as two entries that are summed,
  which is right for a formulation that splits one derivative across the two orders and wrong
  (doubled) for one that supplies both triangles of a symmetric Hessian; since nothing downstream
  can tell which was meant, the ambiguity is refused. The behavior is unchanged in 0.2.x so that
  the former formulations keep working through a patch release; the warning's message says which
  case the entries look like. Supply each pair once, in either order. This joins
  `Problem.ipopt_source` on the list of 0.3.0 removals. The warning is a `FutureWarning` for
  the reason given under 0.2.0: `DeprecationWarning` is suppressed by default outside `__main__`.

- `yapss.math.nextafter`, `rint`, `signbit`, and `spacing` now emit
  `yapss.math.UnsupportedMathFunctionWarning` on real arguments and will raise
  `yapss.math.UnsupportedMathFunctionError` **in 0.3.0**. These step between adjacent
  floating-point values, round half to even, or read the sign bit, and have no symbolic
  equivalent, so they already failed under the `"auto"` derivative method — with a bare numpy
  `TypeError` before 0.2.2, and now with `UnsupportedMathFunctionError`, a subclass of
  `TypeError` whose message names the function and suggests `numpy` directly. Rejecting both
  paths is the goal: a function that works under one derivative method and fails under another
  lets a formulation depend on the derivative method chosen. The real path stays open through
  0.2.x because it worked in 0.2.1. The warning is a `FutureWarning`, for the reason given
  under 0.2.0.

### Fixed

- Comparison and logical operations on symbolic values no longer silently collapse to `True` under
  the `"auto"` derivative method. numpy's object-dtype loop coerces each elementwise result to
  `bool`, and the symbolic values YAPSS passes to a callback define no `__bool__`, so Python's
  default made every comparison true. A gated
  expression such as `(abs(y) <= 1) * x` — enforcing a constraint only over a finite interval —
  therefore evaluated correctly under the finite-difference derivative methods and lost its mask
  entirely under `"auto"`, silently enforcing the constraint over the whole domain. CasADi
  represents comparisons exactly, so the two paths now agree.

  This is fixed at two levels. `yapss.math.equal`, `not_equal`, `less`, `less_equal`, `greater`,
  `greater_equal`, `logical_and`, `logical_or`, and `logical_not` are now dispatched explicitly, as
  `maximum` and `minimum` already were. The operator forms (`<=`, `&`, `~`, and so on) cannot be
  intercepted by `yapss.math` at all, since they go straight to `numpy.ndarray`; the symbolic
  state, control, and time arrays now override the comparison and mask-combination operators
  themselves.

  Formulations that unknowingly relied on the mask being dropped will now produce different
  results. Note also that agreement of *values* does not imply agreement of *derivatives*: CasADi
  differentiates a comparison node to zero almost everywhere, while central differencing straddles
  the discontinuity, so the two backends may still converge to different points on a gated problem.

- The following `yapss.math` functions raised `TypeError` under the `"auto"` derivative method and
  now work: `copysign`, `float_power`, `floor_divide`, `fmod`, `heaviside`, `logaddexp`,
  `logaddexp2`, and `logical_xor`. `logaddexp` and `logaddexp2` are computed in a form that does
  not overflow for large arguments, matching numpy.
- `yapss.math.fmax` and `fmin` returned their first argument rather than the larger or smaller of
  the two under the `"auto"` derivative method.
- `yapss.math.mod` and `remainder` took the sign of the dividend rather than the divisor under the
  `"auto"` derivative method, following casadi's truncating `fmod` instead of numpy's convention.
  `yapss.math.fmod` keeps the truncating convention, as numpy does.
- `arg.phase[p].time` in continuous callbacks is now read-only, like the other attributes the
  framework sets. It had been left assignable as a workaround for the symbolic time array being
  patched in after construction, which left a gap in the protection against attribute typos.
- `yapss.math.exp2` computed `2**log(x)` rather than `2**x` under the `"auto"` derivative method.
- `yapss.math.cbrt` raised `AttributeError` under the `"auto"` derivative method, calling the
  nonexistent `casadi.abs` instead of `casadi.fabs`.
- `yapss.math.trunc` raised `TypeError` under the `"auto"` derivative method; the symbolic values
  defined `__floor__` and `__ceil__` but not `__trunc__`.
- Constant entries in user-supplied continuous Jacobian and Hessian callbacks can be written as
  scalars, such as `hessian[("f", 0), ("x", 0), ("u", 0)] = 1.0`, for every kind of term. Previously
  a scalar Hessian entry raised an uninformative `IndexError` unless the term was an integrand or
  path pair not involving time, and the 0.2.1 → 0.2.2 assembly refactor had removed even that case.
  Broadcasting a constant over the time grid by hand is no longer necessary anywhere; the user
  derivative callbacks follow the same scalar convention as the continuous function itself.

## [0.2.1] - 2026-08-17

### Fixed

- Fixed defects in the assembled NLP derivatives. Generated central-difference callbacks no longer
  leave continuous outputs at a perturbed point. The first Hessian evaluation now uses the current
  phase times instead of zero-initialized values.
- Removed inactive LG/LGR zero-mode constraint rows that were retained for research. Although the
  rows were completely unbounded, Ipopt did not remove them, and their presence caused the
  platform-dependent convergence failure of the Delta III ascent example when the LGR spectral
  method was used.
- Improved cold-start robustness of the Delta III ascent example across supported Ipopt backends.
  The original boundary-data-only guess worked in many solver configurations but introduced a large
  position and velocity discontinuity between phases, leading to nondeterministic restoration
  failures with conda-forge Ipopt/MUMPS. The example now retains a simple boundary-data-based guess
  while interpolating position continuously in geocentric latitude, longitude, and altitude. The
  example test now also requires Ipopt to report a converged status, and the notebook documents the
  motivation for the revised guess.

## [0.2.0] - 2026-08-12

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
- `bounds.phase[p].zero_mode` is now private (`_zero_mode`). It was never meant for
  users to set — it exists for internal testing and research use only.
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

- Fixed cyipopt's Hessian callback silently swallowing Python exceptions raised by a
  user's Hessian function. Previously an exception raised inside a Hessian evaluation —
  whether from a bug, a domain error, or an intentional `SystemExit` — was discarded by
  cyipopt's C extension before reaching Python; Ipopt continued iterating on stale
  Hessian values and either failed to converge or returned a result with no indication
  anything had gone wrong. YAPSS now intercepts cyipopt's Hessian callback, latches the
  first exception and its traceback, asks Ipopt to stop through the intermediate
  callback, and re-raises the original exception once `solve()` returns. Scoped to the
  Conda/cyipopt backend's Hessian callback specifically: cyipopt's objective, gradient,
  constraint, Jacobian, and intermediate callbacks already propagated exceptions
  correctly and needed no change.
- Fixed the vendored (pip) Ipopt interface's handling of exceptions raised by any user
  callback — objective, gradient, constraints, Jacobian, Hessian, or the intermediate
  callback. Through 0.1.1, YAPSS called the upstream `mseipopt.ez.Problem` interface
  directly. Every callback exception, including a `SystemExit` raised inside a Hessian
  evaluation, was caught, its traceback printed to the console, and status `-13`
  ("Invalid number in NLP function or derivative detected") returned to Ipopt — a real
  but misleading status, since nothing was actually numerically invalid. No exception
  ever reached `solve()`'s caller; an apparently normal `Solution` was returned
  regardless, and 0.1.1 predates `IpoptConvergenceWarning`, so nothing else signaled a
  problem either — a user not watching the console for the printed traceback, or who
  didn't think to check `nlp_info.ipopt_status`, would see nothing wrong. Some
  structural bugs in the same upstream code were worse: a malformed Jacobian could
  crash the process outright, and index validation relied on Python `assert` statements
  that silently disappear under `python -O`. Every callback now retains the original
  exception and traceback across the native call and re-raises it once `solve()`
  returns, and sparse structures are validated before they reach the native layer, so a
  bug in a user-supplied function — or a malformed Jacobian/Hessian structure — now
  raises a catchable Python exception instead of crashing, printing an unreachable
  traceback, or surfacing as an unconverged solution.
- Fixed synchronization of the physical time vector passed to numeric continuous
  callbacks. Some internal derivative evaluations copied a new NLP decision vector
  without rebuilding time from the phase endpoints and mesh nodes, leaving callbacks
  with stale time values. For time-dependent dynamics and integrands, this could produce
  incorrect derivative sparsity or central-difference derivatives and, in turn, incorrect
  solutions. Numeric continuous arguments now update decision variables and time together
  for the LG, LGR, and LGL spectral methods.
- Fixed `solution.phase[p].control_multiplier`, which always returned an incorrect,
  unrelated value. This was wrong in every previous release.
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
  changed `Bool` from a four-byte `int` to a one-byte `bool` in 3.14, and the interface
  still declared it four bytes wide. Rather than detect the width at runtime, YAPSS now
  fixes it to the post-3.14 `bool` layout and enforces an Ipopt >= 3.14 floor via the ABI
  check described above, refusing to load an older, incompatible build. In practice the
  old four-byte assumption had not misfired, but a failed option-setting call in
  principle could have been read as success and silently ignored.
- Fixed `Guess.from_solution`, which assigned a solution's `control` array directly onto
  the guess's `time` grid. `control` is defined on `time_c`, which only has the same
  length as `time` for the lgl spectral method; for lgr and lg, `from_solution` either
  raised on `guess.validate()` or produced a misaligned guess. `control` is now
  interpolated (with extrapolation at the endpoints) from `time_c` onto `time` before
  being assigned, so `from_solution` works for all three spectral methods.
- Fixed the error message for an invalid `nx` keyword, which claimed `nx` must be
  positive when the actual, and intended, requirement is nonnegative — a phase with
  zero states is valid and already exercised elsewhere (a controls-only phase driven
  purely by an integral cost). Validation behavior is unchanged; only the message text
  was wrong.

## [0.1.1] - 2026-08-02

### Changed

- Updated the supported Python versions to 3.10 and newer in `pyproject.toml` and `tox.ini` to match
  the supported test matrix. This change is consistent with Python 3.9 having reached its end-of-life
  phase in October 2025.
- The `notebook` extra no longer installs `black`, `isort`, or `jupyterlab_code_formatter` by
  default. These remain available via the `dev` extra for contributors; installing `yapss[notebook]`
  no longer forces an opinionated formatting setup on end users.

### Removed

- Stopped publishing documentation to GitHub Pages. The Pages copy was unreferenced by the
  README, package metadata, and PyPI listing (all of which already pointed at
  [readthedocs.io](https://yapss.readthedocs.io/)), so it had gone stale without anyone noticing.
  The `gh-pages` branch now redirects to Read the Docs instead of serving old content.

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

## [0.1.0] - 2024-12-28

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

[0.2.2]: https://github.com/stevenrhall/yapss/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/stevenrhall/yapss/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/stevenrhall/yapss/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/stevenrhall/yapss/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/stevenrhall/yapss/releases/tag/v0.1.0
