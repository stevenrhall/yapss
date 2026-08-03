# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Unlike the rest of this package, this module is original YAPSS code and is
# not derived from mseipopt. It lives here because locating the library is part
# of the backend's job, but the mseipopt copyright does not extend to it.

"""Locate the IPOPT shared library, preferring the copy CasADi has already loaded.

Rationale
---------
CasADi wheels vendor a full IPOPT stack (IPOPT, MUMPS, OpenBLAS, libgfortran).
A separately built IPOPT -- a pip-installed ``cyipopt``, say -- puts a *second*
copy of that stack into the same address space, and the two vendored OpenMP
runtimes then collide. That is the crash the pip/conda backend split exists to
avoid, so YAPSS binds to CasADi's copy rather than loading another.

That only works if we load *the same file*, which is stricter than it sounds::

    $ ls -li casadi/libipopt.so*
    14328004 libipopt.so           SONAME libipopt.so.3
    14328005 libipopt.so.3         SONAME libipopt.so.3
    14328006 libipopt.so.3.14.11   SONAME libipopt.so.3

Three distinct inodes, identical content, identical SONAME -- pip materializes
the wheel's symlinks as real files. ``dlopen`` deduplicates on ``(device,
inode)``, *not* on SONAME, so asking for ``libipopt.so`` when CasADi has loaded
``libipopt.so.3`` maps a second copy. Verified directly: two mapped inodes, two
distinct ``CreateIpoptProblem`` addresses. Guessing the filename is unsafe.

So the primary strategy does not guess. It forces CasADi to load its IPOPT
plugin, asks the dynamic loader which file that was, and opens exactly that
path. Globbing the package directory by filename pattern remains as a fallback
for platforms where loader introspection is unavailable.

There is deliberately no override -- no environment variable, no explicit-path
argument. Given the above, a user-supplied path is a loaded gun.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DuplicateIpoptLibraryError",
    "IpoptAbiError",
    "IpoptHeaderInfo",
    "IpoptLibraryNotFoundError",
    "IpoptVerificationWarning",
    "casadi_ipopt_path",
    "duplicate_ipopt_copies",
    "glob_ipopt_in_casadi",
    "initialize_ipopt",
    "load_ipopt",
    "read_ipopt_header",
    "resolve_ipopt_library",
    "smoke_test",
    "strategy1_failure_reason",
]

PROBE_NAME = "yapss_probe"
"""Name given to the throwaway CasADi ``nlpsol`` used to force the plugin load.

Must not begin with an underscore: CasADi rejects such names outright with
"Function name is not valid". A leading underscore here silently disables the
primary strategy and degrades the module to filename guessing, which is the
behavior it exists to replace. Tests assert this name is accepted.
"""


class IpoptLibraryNotFoundError(RuntimeError):
    """Raised when no usable IPOPT shared library could be located or loaded."""


class DuplicateIpoptLibraryError(RuntimeError):
    """Raised when more than one IPOPT library is mapped into the process.

    Two IPOPT binaries mean two OpenMP runtimes, which is the configuration
    that crashes once both are actually used. Raising here converts a
    hard-to-diagnose segfault during a solve into an error at load time.
    """


class IpoptAbiError(RuntimeError):
    """Raised when the shipped IPOPT does not match the ctypes declarations.

    Specifically, when IPOPT was built with 64-bit indices or single-precision
    reals. Both change the meaning of every array crossing the boundary, so
    continuing would corrupt results rather than fail cleanly.
    """


class IpoptVerificationWarning(Warning):
    """Warns that the single-copy property could not be checked on this system.

    Not an error: IPOPT loaded and YAPSS will run normally. It means the
    platform offers no way to enumerate mapped libraries, so the check simply
    did not run. Silence with::

        warnings.filterwarnings("ignore", category=IpoptVerificationWarning)
    """


# --------------------------------------------------------------------------
# Failure bookkeeping
# --------------------------------------------------------------------------

_strategy1_failure: str | None = None
"""Why strategy 1 did not produce a path, or None if it succeeded / never ran.

Strategy 1 is allowed to be unavailable, so it must not raise. But a swallowed
failure is exactly what hid the leading-underscore probe bug, so the reason is
recorded here and quoted in any error this module does raise.
"""


def _record_failure(reason: str) -> None:
    """Stash why strategy 1 failed, for inclusion in later error messages."""
    global _strategy1_failure  # noqa: PLW0603
    _strategy1_failure = reason
    logger.debug("IPOPT strategy 1 unavailable: %s", reason)


def strategy1_failure_reason() -> str | None:
    """Return why loader introspection failed, or None if it did not."""
    return _strategy1_failure


# --------------------------------------------------------------------------
# Which shared libraries are currently mapped into this process?
# --------------------------------------------------------------------------


_MAPS_FIELDS = 6
"""Number of whitespace-separated fields in a ``/proc/self/maps`` line."""


def _mapped_paths_linux() -> list[str]:
    """Parse ``/proc/self/maps``.

    Fields are ``address perms offset dev inode pathname``; the pathname is
    last and may contain spaces, so the split must be bounded. An unbounded
    ``split()`` plus ``parts[-1]`` yields a trailing fragment for such paths,
    which then fails the absolute-path test and silently drops the entry.
    """
    paths: set[str] = set()
    try:
        with Path("/proc/self/maps").open(encoding="utf-8", errors="surrogateescape") as fp:
            for line in fp:
                parts = line.rstrip("\n").split(maxsplit=_MAPS_FIELDS - 1)
                if len(parts) < _MAPS_FIELDS:
                    continue
                path = parts[_MAPS_FIELDS - 1]
                if not path.startswith("/"):
                    continue
                # An unlinked-but-still-mapped file keeps this suffix.
                paths.add(path.removesuffix(" (deleted)"))
    except OSError as exc:
        _record_failure(f"could not read /proc/self/maps: {exc}")
        return []
    return sorted(paths)


def _mapped_paths_darwin() -> list[str]:
    """Walk the dyld image list via libSystem.

    ``_dyld_get_image_name`` returns a pointer to dyld-owned storage; it must
    not be freed. The list is not stable across concurrent ``dlopen`` calls,
    but we read it synchronously.
    """
    try:
        libc = ctypes.CDLL(None)
        image_count = libc._dyld_image_count
        image_count.restype = ctypes.c_uint32
        image_count.argtypes = []
        image_name = libc._dyld_get_image_name
        image_name.restype = ctypes.c_char_p
        image_name.argtypes = [ctypes.c_uint32]
    except (AttributeError, OSError) as exc:
        _record_failure(f"dyld image list unavailable: {exc}")
        return []

    names = (image_name(i) for i in range(image_count()))
    return [os.fsdecode(name) for name in names if name]


def _mapped_paths_windows() -> list[str]:
    """Enumerate loaded modules via ``psapi``.

    Every signature here is declared explicitly. Left to its own defaults
    ctypes treats the return of ``GetCurrentProcess`` as a 32-bit ``int``,
    which truncates the pseudo-handle on 64-bit Windows -- the resulting
    failure looks like "no modules loaded" rather than a type error.
    """
    from ctypes import wintypes  # importable on Windows only

    # Fetched by name rather than referenced directly: ctypes.WinDLL does not
    # exist off Windows, and a direct reference is both a type error there and
    # a platform assertion that would let a type checker mark the rest of this
    # function dead. Which lines look dead then depends on the platform the
    # checker runs on, so the branchless form is the portable one.
    windll = getattr(ctypes, "WinDLL", None)
    last_error = getattr(ctypes, "get_last_error", lambda: 0)
    if windll is None:
        _record_failure("ctypes.WinDLL is unavailable; not running on Windows")
        return []

    try:
        psapi = windll("psapi", use_last_error=True)
        kernel32 = windll("kernel32", use_last_error=True)
    except OSError as exc:
        _record_failure(f"could not load psapi/kernel32: {exc}")
        return []

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    psapi.EnumProcessModules.restype = wintypes.BOOL
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        wintypes.LPDWORD,
    ]
    psapi.GetModuleFileNameExW.restype = wintypes.DWORD
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]

    handle = kernel32.GetCurrentProcess()
    needed = wintypes.DWORD()
    count = 1024
    # EnumProcessModules reports the bytes it *would* have needed, so a single
    # retry at the reported size is enough however many modules are loaded.
    for _ in range(2):
        modules = (wintypes.HMODULE * count)()
        if not psapi.EnumProcessModules(
            handle, modules, ctypes.sizeof(modules), ctypes.byref(needed)
        ):
            _record_failure(f"EnumProcessModules failed: {last_error()}")
            return []
        wanted = needed.value // ctypes.sizeof(wintypes.HMODULE)
        if wanted <= count:
            count = wanted
            break
        count = wanted
    else:
        modules = (wintypes.HMODULE * count)()

    out: list[str] = []
    buffer = ctypes.create_unicode_buffer(32768)
    # An explicit loop, not a comprehension: the buffer is reused across calls
    # and must be read immediately after each one, which a lazy comprehension
    # would make subtle rather than shorter.
    for i in range(count):
        if psapi.GetModuleFileNameExW(handle, modules[i], buffer, len(buffer)):
            out.append(buffer.value)  # noqa: PERF401
    return out


def _loaded_library_paths() -> list[str]:
    """Return absolute paths of shared libraries mapped into this process.

    Empty where introspection is unavailable. Callers must read that as
    "unknown", never as "nothing is loaded".

    Dispatch is a lookup rather than an if-chain so that no branch is a
    platform assertion; see the note in _mapped_paths_windows.
    """
    mappers = {
        "linux": _mapped_paths_linux,
        "darwin": _mapped_paths_darwin,
        "win32": _mapped_paths_windows,
    }
    return mappers[_platform_key()]()


def _is_ipopt(path: str) -> bool:
    """Return True if *path* names IPOPT itself, rather than a neighbor of it."""
    stem = Path(path).name.lower()
    if "ipopt" not in stem:
        return False
    # Exclude CasADi's plugin shim (libcasadi_nlpsol_ipopt) and the sensitivity
    # library (libsipopt); both match "ipopt" but are different objects.
    return not (stem.startswith(("libsipopt", "sipopt")) or "casadi" in stem)


def duplicate_ipopt_copies() -> list[str]:
    """Distinct IPOPT files currently mapped into this process.

    More than one entry means two copies of IPOPT -- and of the OpenMP and
    BLAS runtimes beneath it -- are live at once, which is the configuration
    that crashes. An empty list means introspection was unavailable, not that
    nothing is loaded.
    """
    return sorted({p for p in _loaded_library_paths() if _is_ipopt(p)})


# --------------------------------------------------------------------------
# Strategy 1: ask the loader what CasADi loaded
# --------------------------------------------------------------------------


def _force_casadi_ipopt_load() -> bool:
    """Make CasADi load its IPOPT plugin. True on success.

    CasADi resolves solver plugins lazily, so importing it is not enough --
    IPOPT is not mapped until an ``ipopt`` nlpsol is actually instantiated.
    Costs roughly 18 ms, against roughly 45 ms for ``import casadi`` itself,
    and is paid once because the result is cached.
    """
    try:
        import casadi  # deliberately deferred; keeps import yapss cheap
    except ImportError as exc:
        _record_failure(f"casadi could not be imported: {exc}")
        return False

    try:
        x = casadi.SX.sym("x")
        casadi.nlpsol(
            PROBE_NAME,
            "ipopt",
            {"x": x, "f": x * x},
            {"ipopt.print_level": 0, "print_time": False},
        )
    except Exception as exc:  # noqa: BLE001 -- plugin missing, or a build without IPOPT
        _record_failure(f"casadi ipopt probe failed: {type(exc).__name__}: {exc}")
        return False
    return True


def casadi_ipopt_path() -> str | None:
    """Absolute path of the IPOPT library CasADi has loaded, if discoverable."""
    if not _force_casadi_ipopt_load():
        return None
    for path in _loaded_library_paths():
        if _is_ipopt(path):
            return path
    _record_failure("casadi loaded its ipopt plugin, but no IPOPT library is mapped")
    return None


# --------------------------------------------------------------------------
# Strategy 2: glob the CasADi package directory
# --------------------------------------------------------------------------

_PATTERNS: dict[str, tuple[str, ...]] = {
    # Ordered most- to least-preferred. The SONAME-versioned name comes first
    # because that is what CasADi's DT_NEEDED / LC_LOAD_DYLIB resolves to, and
    # so is the copy most likely to be the one already mapped. This ordering is
    # a heuristic; strategy 1 is what makes correctness not depend on it.
    "linux": ("libipopt.so.[0-9]", "libipopt.so.[0-9]*", "libipopt.so"),
    "darwin": ("libipopt.[0-9].dylib", "libipopt.[0-9]*.dylib", "libipopt.dylib"),
    "win32": ("ipopt-[0-9]*.dll", "libipopt-[0-9]*.dll", "ipopt.dll", "libipopt.dll"),
}

_PLATFORM_KEYS = {"win32": "win32", "cygwin": "win32", "darwin": "darwin"}
"""sys.platform values that need their own tables; everything else uses Linux
conventions, which is the right guess for other Unixes and for unknown
platforms."""


def _platform_key() -> str:
    """Return the key into the per-platform filename tables."""
    return _PLATFORM_KEYS.get(sys.platform, "linux")


def casadi_package_dir() -> Path | None:
    """Directory of the installed CasADi package, found without importing it."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("casadi")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def glob_ipopt_in_casadi() -> str | None:
    """Find a vendored IPOPT by filename pattern in the CasADi package dir."""
    package = casadi_package_dir()
    if package is None:
        return None
    for pattern in _PATTERNS[_platform_key()]:
        matches = sorted(p for p in package.glob(pattern) if _is_ipopt(str(p)))
        if matches:
            return str(matches[0])
    return None


# --------------------------------------------------------------------------
# Resolution and loading
# --------------------------------------------------------------------------

_resolved_path: str | None = None
"""Cached result of resolve_ipopt_library().

Resolution is worth caching for more than speed: the probe has process-global
side effects, and the answer cannot change within a process once a library is
mapped.
"""


def _not_found_message() -> str:
    """Explain what was searched, for the case where nothing was found."""
    package = casadi_package_dir()
    try:
        import casadi  # deliberately deferred

        version = casadi.__version__
    except Exception:  # noqa: BLE001 -- diagnostics must not raise
        version = "unknown (casadi could not be imported)"

    contents = "casadi package directory not found"
    if package is not None:
        try:
            matches = sorted(p.name for p in package.glob("*ipopt*"))
            contents = ", ".join(matches) if matches else "no files matching *ipopt*"
        except OSError as exc:
            contents = f"could not be listed: {exc}"

    reason = strategy1_failure_reason() or "no reason recorded"
    patterns = ", ".join(_PATTERNS[_platform_key()])
    return (
        "YAPSS could not find the IPOPT library that CasADi bundles.\n\n"
        f"  casadi version : {version}\n"
        f"  casadi package : {package}\n"
        f"  looked for     : {patterns}\n"
        f"  found there    : {contents}\n"
        f"  loader probe   : {reason}\n\n"
        "YAPSS deliberately does not fall back to searching the system for some "
        "other IPOPT: loading one that CasADi did not bundle is what causes the "
        "OpenMP runtime collision this design exists to avoid, and it fails as a "
        "crash rather than an error. Every CasADi wheel ships IPOPT, so reaching "
        "this point means something unexpected -- please report it to the YAPSS "
        "maintainers with the details above."
    )


def resolve_ipopt_library() -> str:
    """Return the path of the IPOPT library to load, resolving once per process.

    Tries the copy CasADi has already loaded, then a glob of the CasADi package
    directory. There is no third strategy on purpose: falling back to a bare
    library name would let the dynamic loader supply an IPOPT from anywhere on
    the system, which is precisely the hazard this module prevents.
    """
    global _resolved_path  # noqa: PLW0603
    if _resolved_path is not None:
        return _resolved_path

    path = casadi_ipopt_path()
    if path is not None:
        logger.debug("IPOPT resolved by loader introspection: %s", path)
    else:
        path = glob_ipopt_in_casadi()
        if path is None:
            raise IpoptLibraryNotFoundError(_not_found_message())
        logger.debug("IPOPT resolved by globbing the casadi package: %s", path)

    _resolved_path = path
    return path


def _describe(path: str) -> str:
    """Render a path with the file identity dlopen actually deduplicates on."""
    try:
        st = Path(path).stat()
    except OSError:
        return f"  {path}\n      (could not stat)"
    return f"  {path}\n      device={st.st_dev} inode={st.st_ino} size={st.st_size}"


def _check_single_copy(before: list[str], after: list[str]) -> None:
    """Verify that exactly one IPOPT library is mapped, or explain why not.

    The point of the whole module is this property, and it is worth checking
    directly rather than trusting that whichever strategy found the path did
    so correctly. Verify the outcome, not the mechanism.

    *before* is sampled after resolution but immediately before our own
    ``CDLL`` call, so that a duplicate can be attributed to whoever caused it.
    """
    if len(after) == 1:
        logger.debug("verified: exactly one IPOPT library mapped (%s)", after[0])
        return

    if not after:
        # We just loaded IPOPT, so it is certainly mapped; an empty list means
        # the platform gave us no way to look, not that nothing is there.
        reason = strategy1_failure_reason() or "no reason recorded"
        warnings.warn(
            "YAPSS could not verify that only one IPOPT library is loaded, because "
            f"this platform provides no way to list mapped libraries ({reason}). "
            "IPOPT loaded and YAPSS will run normally; this is a gap in checking, "
            "not a detected problem.",
            IpoptVerificationWarning,
            stacklevel=3,
        )
        return

    preexisting = sorted(set(before))
    introduced = sorted(set(after) - set(before))
    if len(preexisting) > 1:
        cause = (
            "More than one was already mapped before YAPSS loaded anything, so "
            "another package brought its own IPOPT into this process -- a "
            "pip-installed cyipopt is the usual cause. YAPSS and CasADi must "
            "share a single IPOPT; see the documentation on why cyipopt is "
            "supported only under Conda."
        )
    elif introduced:
        cause = (
            "YAPSS's own load added a second copy, meaning it selected a file "
            "other than the one CasADi had already loaded. That is a bug in "
            "this module rather than anything you did -- please report it."
        )
    else:
        cause = "The duplicate was already present and was not introduced by YAPSS."

    listing = "\n".join(_describe(p) for p in after)
    detail = strategy1_failure_reason()
    footer = f"\n\nLoader introspection note: {detail}" if detail else ""
    msg = (
        f"{len(after)} IPOPT libraries are mapped into this process, but exactly "
        f"one is required. Two IPOPT binaries carry two OpenMP runtimes, which "
        f"crashes as soon as both are used.\n\n{listing}\n\n{cause}{footer}"
    )
    raise DuplicateIpoptLibraryError(msg)


def load_ipopt() -> tuple[ctypes.CDLL, str]:
    """Load IPOPT and return ``(library, resolved_path)``.

    Uses ``RTLD_LOCAL``, which is ctypes' default and is deliberate here:
    ``RTLD_GLOBAL`` would expose IPOPT's symbols for interposition against
    CasADi's own copy, which is the class of failure this module exists to
    prevent.
    """
    path = resolve_ipopt_library()

    # Sampled after resolution, not before: resolving runs the CasADi probe,
    # which maps CasADi's own IPOPT. The boundary that lets us attribute a
    # duplicate is what was mapped immediately before *our* CDLL call.
    before = duplicate_ipopt_copies()

    # Since Python 3.8, Windows no longer searches PATH when resolving a DLL's
    # own dependencies (MUMPS, OpenBLAS, libgfortran). Without this, the load
    # fails with a bare "DLL load failed" that names only the top-level
    # library and none of the dependency that was actually missing.
    cookie = None
    if os.name == "nt" and Path(path).is_absolute():
        directory = Path(path).parent
        if directory.is_dir() and hasattr(os, "add_dll_directory"):
            cookie = os.add_dll_directory(str(directory))

    try:
        library = ctypes.CDLL(path)
    except OSError as exc:
        reason = strategy1_failure_reason()
        detail = f" (loader introspection was unavailable: {reason})" if reason else ""
        msg = (
            f"could not load the IPOPT shared library from {path!r}{detail}. "
            "YAPSS uses the IPOPT library bundled with CasADi, so check that "
            "casadi is installed and its wheel is intact."
        )
        raise IpoptLibraryNotFoundError(msg) from exc
    finally:
        if cookie is not None:
            cookie.close()

    _check_single_copy(before, duplicate_ipopt_copies())
    return library, path


# --------------------------------------------------------------------------
# ABI verification from the headers CasADi ships
# --------------------------------------------------------------------------

_SMOKE_TOLERANCE = 1e-6
"""Agreement required between the smoke solve and its analytic solution."""

_BOOL_NARROWED_IN = (3, 14)
"""IPOPT release that changed ``typedef int Bool`` to ``typedef bool Bool``."""

_DEFINE_RE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)(?:[ \t]+(.*?))?[ \t]*$", re.MULTILINE)
_UNDEF_RE = re.compile(r"^[ \t]*/\*[ \t]*#[ \t]*undef[ \t]+(\w+)[ \t]*\*/[ \t]*$", re.MULTILINE)


@dataclass(frozen=True)
class IpoptHeaderInfo:
    """What ``IpoptConfig.h`` says about how this IPOPT was built."""

    path: Path
    version: tuple[int, int, int] | None
    int64: bool
    """True if built with 64-bit indices, making our c_int declarations wrong."""
    single: bool
    """True if built with single-precision reals, making c_double wrong."""

    @property
    def bool_ctype(self) -> Any:
        """Return the ctypes type matching this IPOPT's ``Bool`` typedef."""
        if self.version is not None and self.version[:2] < _BOOL_NARROWED_IN:
            return ctypes.c_int
        return ctypes.c_bool


def read_ipopt_header() -> IpoptHeaderInfo | None:
    """Read ``IpoptConfig.h`` from the CasADi package, or None if unavailable.

    Autoconf writes an inactive macro as ``/* #undef NAME */`` rather than
    omitting it, so "defined", "explicitly undefined", and "absent" are three
    different states and only the first is a problem.

    Returns None rather than raising: CasADi trimming headers out of a wheel is
    far more plausible than CasADi switching to 64-bit indices, and the smoke
    test still provides independent (weaker) evidence.
    """
    package = casadi_package_dir()
    if package is None:
        return None
    header = package / "include" / "coin-or" / "IpoptConfig.h"
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("could not read %s: %s", header, exc)
        return None

    defined = {m.group(1): (m.group(2) or "").strip() for m in _DEFINE_RE.finditer(text)}

    version: tuple[int, int, int] | None = None
    try:
        version = (
            int(defined["IPOPT_VERSION_MAJOR"]),
            int(defined["IPOPT_VERSION_MINOR"]),
            int(defined["IPOPT_VERSION_RELEASE"]),
        )
    except (KeyError, ValueError):
        logger.debug("no usable IPOPT version macros in %s", header)

    return IpoptHeaderInfo(
        path=header,
        version=version,
        int64="IPOPT_INT64" in defined,
        single="IPOPT_SINGLE" in defined,
    )


def _verify_header_abi(info: IpoptHeaderInfo) -> None:
    """Raise if this IPOPT was built in a way our declarations cannot match."""
    problems = []
    if info.int64:
        problems.append(
            "IPOPT_INT64 is defined, so IPOPT uses 64-bit indices, but YAPSS "
            "declares them as c_int (32-bit)"
        )
    if info.single:
        problems.append(
            "IPOPT_SINGLE is defined, so IPOPT uses single-precision reals, "
            "but YAPSS declares them as c_double"
        )
    if not problems:
        return
    detail = "\n".join(f"  - {p}" for p in problems)
    msg = (
        f"The IPOPT bundled with CasADi was built with options YAPSS does not "
        f"support:\n\n{detail}\n\n"
        f"Read from {info.path}. Every array crossing the boundary would be "
        f"misinterpreted, so YAPSS stops rather than returning wrong answers. "
        f"Please report this to the YAPSS maintainers."
    )
    raise IpoptAbiError(msg)


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------


def smoke_test() -> None:
    """Solve a small constrained problem, or raise.

    Deliberately uses the declarations in ``mseipopt.bare`` rather than private
    copies: the point is to exercise what production actually uses, and a test
    with its own signatures could pass while the real ones were wrong.

    The problem has three variables, two constraints, an off-diagonal Hessian
    entry and a Jacobian with five structurally distinct nonzeros, so both the
    structure and values passes carry real index traffic. A one-variable,
    zero-constraint problem -- the obvious thing to write -- would barely
    exercise ``eval_jac_g`` and is weak evidence for exactly the index-width
    mismatch this is meant to catch.
    """
    from . import bare  # deferred: keeps this module loadable standalone

    failures: list[BaseException] = []

    def guard(fn: Any) -> Any:
        """Stop a Python exception from unwinding into C; re-raise it after."""

        def wrapper(*args: Any) -> bool:
            try:
                fn(*args)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)
                return False  # tells IPOPT to abort cleanly
            return True

        return wrapper

    # Unused parameters keep an underscore prefix but their IPOPT names, so the
    # signatures still line up with IpStdCInterface.h.
    # minimize (x0-1)^2 + (x1-2)^2 + (x2-3)^2  s.t.  x0+x1+x2 <= 4, x0*x1 free
    @guard
    def eval_f(_n: int, x: Any, _new_x: Any, obj: Any, _data: Any) -> None:
        obj[0] = (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + (x[2] - 3.0) ** 2

    @guard
    def eval_grad_f(_n: int, x: Any, _new_x: Any, grad: Any, _data: Any) -> None:
        for i, target in enumerate((1.0, 2.0, 3.0)):
            grad[i] = 2.0 * (x[i] - target)

    @guard
    def eval_g(_n: int, x: Any, _new_x: Any, _m: int, g: Any, _data: Any) -> None:
        g[0] = x[0] + x[1] + x[2]
        g[1] = x[0] * x[1]

    jac_rows, jac_cols = (0, 0, 0, 1, 1), (0, 1, 2, 0, 1)

    @guard
    def eval_jac_g(
        _n: int,
        x: Any,
        _new_x: Any,
        _m: int,
        _nnz: int,
        i_row: Any,
        j_col: Any,
        values: Any,
        _data: Any,
    ) -> None:
        if not values:
            for k, (r, c) in enumerate(zip(jac_rows, jac_cols)):
                i_row[k], j_col[k] = r, c
        else:
            values[0] = values[1] = values[2] = 1.0
            values[3], values[4] = x[1], x[0]

    hess_rows, hess_cols = (0, 1, 1, 2), (0, 0, 1, 2)

    @guard
    def eval_h(
        _n: int,
        _x: Any,
        _new_x: Any,
        obj_factor: float,
        _m: int,
        lam: Any,
        _new_lam: Any,
        _nnz: int,
        i_row: Any,
        j_col: Any,
        values: Any,
        _data: Any,
    ) -> None:
        if not values:
            for k, (r, c) in enumerate(zip(hess_rows, hess_cols)):
                i_row[k], j_col[k] = r, c
        else:
            values[0] = 2.0 * obj_factor
            values[1] = lam[1]  # d2/dx0dx1 of the bilinear constraint
            values[2] = 2.0 * obj_factor
            values[3] = 2.0 * obj_factor

    # These must outlive the solve; IPOPT holds raw pointers to them and would
    # jump into freed memory otherwise.
    callbacks = (
        bare.Eval_F_CB(eval_f),
        bare.Eval_G_CB(eval_g),
        bare.Eval_Grad_F_CB(eval_grad_f),
        bare.Eval_Jac_G_CB(eval_jac_g),
        bare.Eval_H_CB(eval_h),
    )

    inf = 2.0e19  # beyond IPOPT's default nlp_(lower|upper)_bound_inf
    x_l = (ctypes.c_double * 3)(-inf, -inf, -inf)
    x_u = (ctypes.c_double * 3)(inf, inf, inf)
    g_l = (ctypes.c_double * 2)(-inf, -inf)
    g_u = (ctypes.c_double * 2)(4.0, inf)

    problem = bare.CreateIpoptProblem(3, x_l, x_u, 2, g_l, g_u, 5, 4, 0, *callbacks)
    if not problem:
        msg = "CreateIpoptProblem returned NULL during the IPOPT smoke test"
        raise IpoptAbiError(msg)

    try:
        bare.AddIpoptIntOption(problem, b"print_level", 0)
        bare.AddIpoptStrOption(problem, b"sb", b"yes")
        x = (ctypes.c_double * 3)(0.0, 0.0, 0.0)
        objective = (ctypes.c_double * 1)(0.0)
        status = bare.IpoptSolve(problem, x, None, objective, None, None, None, None)
    finally:
        bare.FreeIpoptProblem(problem)

    if failures:
        raise failures[0]
    if status != 0:
        msg = f"the IPOPT smoke test returned status {status}, expected 0 (solved)"
        raise IpoptAbiError(msg)

    expected = (1.0 / 3.0, 4.0 / 3.0, 7.0 / 3.0)
    if any(abs(x[i] - expected[i]) > _SMOKE_TOLERANCE for i in range(3)):
        got = ", ".join(f"{x[i]:.6f}" for i in range(3))
        want = ", ".join(f"{v:.6f}" for v in expected)
        msg = (
            f"the IPOPT smoke test converged to ({got}) but the analytic "
            f"solution is ({want}); the ctypes configuration is wrong in a way "
            f"that corrupts values rather than crashing"
        )
        raise IpoptAbiError(msg)


# --------------------------------------------------------------------------
# One-call initialization
# --------------------------------------------------------------------------

_initialized = False


def initialize_ipopt() -> str:
    """Resolve, load, verify, and configure IPOPT. Return the path loaded.

    Deliberately a single call rather than a sequence the caller assembles:
    every step here exists to make a later step safe, and one that is easy to
    omit is one that will eventually be omitted.

    Idempotent, and cheap after the first call.
    """
    global _initialized  # noqa: PLW0603
    from . import bare

    if _initialized:
        return resolve_ipopt_library()

    library, path = load_ipopt()

    header = read_ipopt_header()
    if header is None:
        logger.debug("IpoptConfig.h unavailable; relying on the smoke test alone")
    else:
        _verify_header_abi(header)
        logger.debug("IPOPT %s verified from %s", header.version, header.path)
        bare.set_bool_type(header.bool_ctype)

    bare.use_library(library)
    smoke_test()
    _initialized = True
    return path
