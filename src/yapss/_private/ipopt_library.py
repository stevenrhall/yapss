# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

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
import sys
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DuplicateIpoptLibraryError",
    "IpoptLibraryNotFoundError",
    "IpoptVerificationWarning",
    "casadi_ipopt_path",
    "duplicate_ipopt_copies",
    "glob_ipopt_in_casadi",
    "load_ipopt",
    "resolve_ipopt_library",
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
