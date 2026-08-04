"""Worker for the IPOPT load-order tests. Not collected by pytest.

Each check must run in a fresh interpreter: ``dlopen`` maps a library
permanently, so once a test has loaded IPOPT there is no way back and later
checks in the same process would be measuring the first one's leftovers.

    python _ipopt_worker.py <check>

Exit status: 0 pass, 1 fail, 2 skip (a precondition is unavailable here).
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

PASS, FAIL, SKIP = 0, 1, 2


def loadable_ipopt_files(library):
    """Every IPOPT file in the CasADi package that could actually be loaded.

    Excludes import libraries and libtool archives (.la, .dll.a, .lib), which
    match the name pattern but are not loadable objects.
    """
    package = library.casadi_package_dir()
    if package is None:
        return []
    if sys.platform == "darwin":
        suffixes = (".dylib",)
    elif sys.platform.startswith("win") or sys.platform == "win32":
        suffixes = (".dll",)
    else:
        suffixes = (".so",)
    out = []
    for path in sorted(package.glob("*ipopt*")):
        name = path.name.lower()
        loadable = name.endswith(suffixes) or (".so." in name and suffixes == (".so",))
        if loadable and path.is_file() and library._is_ipopt(str(path)):
            out.append(str(path))
    return out


def force_casadi():
    """Make CasADi load its IPOPT, independently of the code under test.

    Deliberately does NOT call library._force_casadi_ipopt_load(). Ground
    truth has to be established by other means, because if that function is
    broken -- which is exactly what these tests exist to detect -- reusing it
    would turn a failure into a skip, and the test would report "precondition
    unavailable" for the one condition it is supposed to catch.
    """
    try:
        import casadi
    except ImportError:
        return False
    try:
        x = casadi.SX.sym("x")
        casadi.nlpsol(
            "truth_probe",  # hardcoded valid name; no leading underscore
            "ipopt",
            {"x": x, "f": x * x},
            {"ipopt.print_level": 0, "print_time": False},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    casadi probe failed: {type(exc).__name__}: {exc}")
        return False
    return True


def check_cold_start():
    """Resolution must work before anything has touched CasADi."""
    from yapss._private.mseipopt import library

    if library.duplicate_ipopt_copies():
        print("    IPOPT already mapped at startup")
        return SKIP
    path = library.resolve_ipopt_library()
    print(f"    cold resolve -> {path}")
    if not Path(path).is_absolute() or not Path(path).exists():
        print("    FAIL: did not resolve to a real absolute path")
        return FAIL
    copies = library.duplicate_ipopt_copies()
    if len(copies) > 1:
        print(f"    FAIL: cold start mapped {len(copies)} copies")
        return FAIL
    return PASS


def check_idempotent():
    """Repeated resolve and load must stay at exactly one copy."""
    from yapss._private.mseipopt import library

    paths = set()
    for _ in range(5):
        path = library.resolve_ipopt_library()
        ctypes.CDLL(path)
        paths.add(path)
    copies = library.duplicate_ipopt_copies()
    print(f"    distinct resolutions: {paths}")
    print(f"    copies mapped       : {len(copies)}")
    if len(paths) != 1:
        return FAIL
    return PASS if len(copies) <= 1 else FAIL


def check_initialize():
    """A full initialization must leave exactly one IPOPT mapped."""
    from yapss._private.mseipopt import bare, library

    path = library.initialize_ipopt()
    copies = library.duplicate_ipopt_copies()
    print(f"    initialized : {path}")
    print(f"    bare.Bool   : {bare.Bool.__name__}")
    print(f"    copies      : {len(copies)}")
    if copies and len(copies) != 1:
        return FAIL
    return PASS


def _sabotage(strategy1_alive: bool):
    """Point the glob at the wrong file and see whether we still stay at one.

    This is the only check that proves strategy 1 does any work. On Linux and
    macOS the glob happens to pick the same file CasADi loaded, so every other
    test passes even with loader introspection completely dead -- which is
    exactly how the original leading-underscore probe bug went unnoticed.
    """
    from yapss._private.mseipopt import library

    if not force_casadi():
        print("    casadi could not load its ipopt plugin")
        return SKIP
    truth = library.duplicate_ipopt_copies()
    if len(truth) != 1:
        print(f"    expected exactly one IPOPT mapped, got {truth}")
        return SKIP
    truth_path = truth[0]

    alternates = [p for p in loadable_ipopt_files(library) if p != truth_path]
    if not alternates:
        # Windows ships a single IPOPT DLL, so there is nothing to confuse.
        print("    only one loadable IPOPT file present")
        return SKIP

    key = library._platform_key()
    library._PATTERNS[key] = (Path(alternates[0]).name, *library._PATTERNS[key])
    library._resolved_path = None
    if not strategy1_alive:
        library._force_casadi_ipopt_load = lambda: False

    path = library.resolve_ipopt_library()
    ctypes.CDLL(path)
    copies = library.duplicate_ipopt_copies()
    print(f"    truth    : {Path(truth_path).name}")
    print(f"    resolved : {Path(path).name}")
    print(f"    copies   : {len(copies)}")

    if strategy1_alive:
        # Strategy 1 should ignore the sabotaged glob entirely.
        return PASS if path == truth_path and len(copies) == 1 else FAIL
    # Control: without strategy 1 the glob really does map a second copy.
    # If this stops happening the hazard is gone and the design can be revisited.
    return PASS if len(copies) == 2 else FAIL


def check_sabotage_strategy1_alive():
    """With introspection working, a wrong glob must not matter."""
    return _sabotage(strategy1_alive=True)


def check_sabotage_strategy1_dead():
    """Control: without introspection, the wrong glob maps a second copy."""
    return _sabotage(strategy1_alive=False)


CHECKS = {
    "cold_start": check_cold_start,
    "idempotent": check_idempotent,
    "initialize": check_initialize,
    "sabotage_strategy1_alive": check_sabotage_strategy1_alive,
    "sabotage_strategy1_dead": check_sabotage_strategy1_dead,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(f"usage: {Path(__file__).name} <{'|'.join(CHECKS)}>")
        return FAIL
    try:
        return CHECKS[sys.argv[1]]()
    except ImportError as exc:
        print(f"    unavailable: {exc}")
        return SKIP


if __name__ == "__main__":
    sys.exit(main())
