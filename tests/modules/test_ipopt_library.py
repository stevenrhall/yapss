"""

Test yapss._private.mseipopt.library, the IPOPT resolver and ABI checks.

These run in the ordinary pytest process. Anything that depends on load
*order* lives in test_ipopt_isolation.py instead, because a shared library
cannot be unmapped once loaded, so those cases need a fresh interpreter.

"""

import ctypes

import pytest

from yapss._private.config import get_conda_prefix
from yapss._private.mseipopt import bare, library

# On conda YAPSS uses cyipopt and never reaches this module, so tests that
# actually load IPOPT would be exercising a code path that platform does not
# use. Parsing and message-formatting tests still run everywhere.
requires_mseipopt = pytest.mark.skipif(
    bool(get_conda_prefix()),
    reason="conda uses the cyipopt backend; the resolver is not used there",
)


def write_header(tmp_path, **macros):
    """Write a synthetic IpoptConfig.h and return the package root.

    Autoconf emits an inactive macro as ``/* #undef NAME */`` rather than
    omitting it, so a None value here means "explicitly undefined".
    """
    directory = tmp_path / "include" / "coin-or"
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        f"#define {key} {value}" if value is not None else f"/* #undef {key} */"
        for key, value in macros.items()
    ]
    (directory / "IpoptConfig.h").write_text("\n".join(lines) + "\n")
    return tmp_path


# --------------------------------------------------------------- helpers ---


def test_is_ipopt_excludes_neighbours():
    """IPOPT's neighbours in the CasADi wheel must not be mistaken for it."""
    assert library._is_ipopt("/x/libipopt.so.3")
    assert library._is_ipopt("/x/libipopt-3.dll")
    assert library._is_ipopt("/x/libipopt.3.dylib")
    # The sensitivity library and CasADi's plugin shim both contain "ipopt".
    assert not library._is_ipopt("/x/libsipopt.so.3")
    assert not library._is_ipopt("/x/libcasadi_nlpsol_ipopt.so")
    assert not library._is_ipopt("/x/libcasadi.so")


def test_platform_key_is_known():
    """Every platform must map to a table we actually have."""
    assert library._platform_key() in {"linux", "darwin", "win32"}
    assert library._platform_key() in library._PATTERNS


# ------------------------------------------------------- strategy 1 alive ---


@requires_mseipopt
def test_probe_name_is_accepted_by_casadi():
    """The probe name must not start with an underscore. Strict on purpose.

    CasADi rejects leading-underscore function names, so a name like
    "_yapss_probe" silently disables loader introspection and degrades the
    resolver to the filename guessing it exists to replace. Production
    tolerates strategy 1 being unavailable; CI must not.
    """
    pytest.importorskip("casadi")
    assert not library.PROBE_NAME.startswith("_")
    assert library._force_casadi_ipopt_load() is True


@requires_mseipopt
def test_resolver_finds_the_library_casadi_loaded():
    """Resolution must agree with what the loader reports as mapped."""
    pytest.importorskip("casadi")
    assert library._force_casadi_ipopt_load() is True
    mapped = library.duplicate_ipopt_copies()
    if not mapped:
        pytest.skip("loader introspection unavailable on this platform")
    assert library.resolve_ipopt_library() == mapped[0]


# ------------------------------------------------------------ the guard ---


def test_guard_accepts_exactly_one_copy(recwarn):
    """One mapped copy is the good case: no error, no warning."""
    library._check_single_copy(["/x/libipopt.so.3"], ["/x/libipopt.so.3"])
    assert [w for w in recwarn if w.category is library.IpoptVerificationWarning] == []


def test_guard_blames_another_package_when_duplicate_preexisted():
    """Two copies before we loaded means something else brought its own."""
    both = ["/x/libipopt.so", "/x/libipopt.so.3"]
    with pytest.raises(library.DuplicateIpoptLibraryError) as excinfo:
        library._check_single_copy(both, both)
    message = str(excinfo.value)
    assert "already mapped before YAPSS loaded anything" in message
    assert "cyipopt" in message
    # The paths must be listed, since that is what makes it diagnosable.
    assert "/x/libipopt.so.3" in message


def test_guard_blames_itself_when_it_introduced_the_duplicate():
    """A copy we added means the resolver picked the wrong file: our bug."""
    with pytest.raises(library.DuplicateIpoptLibraryError) as excinfo:
        library._check_single_copy(
            ["/x/libipopt.so.3"],
            ["/x/libipopt.so", "/x/libipopt.so.3"],
        )
    message = str(excinfo.value)
    assert "YAPSS's own load added a second copy" in message
    assert "report it" in message


def test_guard_warns_when_it_cannot_verify():
    """No introspection means unverified, which must be said, not implied."""
    with pytest.warns(library.IpoptVerificationWarning, match="could not verify"):
        library._check_single_copy([], [])


# ---------------------------------------------------- the not-found error ---


def test_not_found_error_explains_what_was_searched(monkeypatch):
    """The error has to carry enough detail to act on without a round trip."""
    monkeypatch.setattr(library, "_resolved_path", None)
    monkeypatch.setattr(library, "_force_casadi_ipopt_load", lambda: False)
    key = library._platform_key()
    monkeypatch.setitem(library._PATTERNS, key, ("definitely-not-a-real-lib",))

    with pytest.raises(library.IpoptLibraryNotFoundError) as excinfo:
        library.resolve_ipopt_library()

    message = str(excinfo.value)
    assert "casadi version" in message
    assert "looked for" in message
    assert "definitely-not-a-real-lib" in message
    # It must refuse rather than offer to search the system for another IPOPT.
    assert "does not fall back" in message


# ---------------------------------------------------------- header / ABI ---


@requires_mseipopt
def test_real_header_is_readable_and_sane():
    """The IPOPT CasADi actually ships must match our declarations."""
    info = library.read_ipopt_header()
    if info is None:
        pytest.skip("IpoptConfig.h not shipped in this CasADi build")
    assert info.int64 is False
    assert info.single is False
    assert info.version is not None
    library._verify_header_abi(info)  # must not raise


def test_header_absent_returns_none_without_raising(monkeypatch, tmp_path):
    """A missing header degrades to the smoke test; it is not an error."""
    monkeypatch.setattr(library, "casadi_package_dir", lambda: tmp_path)
    assert library.read_ipopt_header() is None


def test_header_reports_undefined_flags(monkeypatch, tmp_path):
    """`/* #undef X */` means undefined, not absent, and not defined."""
    root = write_header(
        tmp_path,
        IPOPT_VERSION_MAJOR=3,
        IPOPT_VERSION_MINOR=14,
        IPOPT_VERSION_RELEASE=11,
        IPOPT_INT64=None,
        IPOPT_SINGLE=None,
    )
    monkeypatch.setattr(library, "casadi_package_dir", lambda: root)
    info = library.read_ipopt_header()
    assert info.version == (3, 14, 11)
    assert info.int64 is False
    assert info.single is False
    assert info.bool_ctype is ctypes.c_bool


@pytest.mark.parametrize(
    ("macro", "expected"),
    [("IPOPT_INT64", "64-bit indices"), ("IPOPT_SINGLE", "single-precision")],
)
def test_incompatible_build_raises(monkeypatch, tmp_path, macro, expected):
    """64-bit indices or single precision would corrupt every array."""
    macros = {
        "IPOPT_VERSION_MAJOR": 3,
        "IPOPT_VERSION_MINOR": 14,
        "IPOPT_VERSION_RELEASE": 11,
        "IPOPT_INT64": None,
        "IPOPT_SINGLE": None,
        macro: 1,
    }
    root = write_header(tmp_path, **macros)
    monkeypatch.setattr(library, "casadi_package_dir", lambda: root)

    info = library.read_ipopt_header()
    with pytest.raises(library.IpoptAbiError, match=expected):
        library._verify_header_abi(info)


def test_pre_314_header_is_rejected(monkeypatch, tmp_path):
    """The hardened interface has one fixed, modern Bool ABI."""
    root = write_header(
        tmp_path,
        IPOPT_VERSION_MAJOR=3,
        IPOPT_VERSION_MINOR=12,
        IPOPT_VERSION_RELEASE=13,
        IPOPT_INT64=None,
        IPOPT_SINGLE=None,
    )
    monkeypatch.setattr(library, "casadi_package_dir", lambda: root)
    info = library.read_ipopt_header()
    with pytest.raises(library.IpoptAbiError, match="older than the required"):
        library._verify_header_abi(info)


def test_header_without_version_macros_is_rejected(monkeypatch, tmp_path):
    """A smoke solve cannot prove the callback Bool width."""
    root = write_header(tmp_path, IPOPT_INT64=None, IPOPT_SINGLE=None)
    monkeypatch.setattr(library, "casadi_package_dir", lambda: root)
    info = library.read_ipopt_header()
    assert info.version is None
    with pytest.raises(library.IpoptAbiError, match="version macros"):
        library._verify_header_abi(info)


def test_callback_bool_type_is_fixed():
    """All supported callbacks use C bool in arguments and returns."""
    assert bare.Bool is ctypes.c_bool
    assert bare.Eval_F_CB._restype_ is ctypes.c_bool


class FakeFunction:
    """ctypes function stand-in that accepts declaration attributes."""


class FakeIpoptLibrary:
    """Library stand-in exposing Ipopt 3.14.11's eleven C functions."""

    def __init__(self, handle=1):
        self._handle = handle
        for name in (
            "CreateIpoptProblem",
            "FreeIpoptProblem",
            "AddIpoptStrOption",
            "AddIpoptNumOption",
            "AddIpoptIntOption",
            "OpenIpoptOutputFile",
            "SetIpoptProblemScaling",
            "SetIntermediateCallback",
            "IpoptSolve",
            "GetIpoptCurrentIterate",
            "GetIpoptCurrentViolations",
        ):
            setattr(self, name, FakeFunction())


def test_bare_declares_all_ipopt_314_functions(monkeypatch):
    """The raw layer mirrors the full 3.14.11 function inventory."""
    fake = FakeIpoptLibrary()
    monkeypatch.setattr(bare, "_ipopt_lib", None)
    bare.use_library(fake)

    assert fake.GetIpoptCurrentIterate.restype is ctypes.c_bool
    assert fake.GetIpoptCurrentIterate.argtypes == [
        bare.IpoptProblem,
        ctypes.c_bool,
        ctypes.c_int,
        bare.c_double_p,
        bare.c_double_p,
        bare.c_double_p,
        ctypes.c_int,
        bare.c_double_p,
        bare.c_double_p,
    ]
    assert fake.GetIpoptCurrentViolations.restype is ctypes.c_bool
    assert len(fake.GetIpoptCurrentViolations.argtypes) == 11


def test_bare_library_binding_is_immutable(monkeypatch):
    """A native problem can never be dispatched through a replacement library."""
    first = FakeIpoptLibrary(handle=1)
    same_native_handle = FakeIpoptLibrary(handle=1)
    replacement = FakeIpoptLibrary(handle=2)
    monkeypatch.setattr(bare, "_ipopt_lib", None)

    bare.use_library(first)
    bare.use_library(same_native_handle)
    assert bare._ipopt_lib is first
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        bare.use_library(replacement)


def reset_initialization(monkeypatch):
    """Reset initializer globals for isolated state-machine tests."""
    monkeypatch.setattr(library, "_initialization_state", "uninitialized")
    monkeypatch.setattr(library, "_initialization_path", None)
    monkeypatch.setattr(library, "_initialization_error", None)


def compatible_header(tmp_path):
    """Return parsed metadata for the supported ABI."""
    path = tmp_path / "IpoptConfig.h"
    return library.IpoptHeaderInfo(path, (3, 14, 11), int64=False, single=False)


def test_initialize_is_statefully_idempotent(monkeypatch, tmp_path):
    """Ready initialization performs native setup and smoke testing only once."""
    reset_initialization(monkeypatch)
    calls = []
    native = object()
    monkeypatch.setattr(library, "read_ipopt_header", lambda: compatible_header(tmp_path))
    monkeypatch.setattr(library, "load_ipopt", lambda: (native, "/casadi/libipopt"))
    monkeypatch.setattr(bare, "use_library", lambda value: calls.append(("use", value)))
    monkeypatch.setattr(library, "smoke_test", lambda: calls.append(("smoke", None)))

    assert library.initialize_ipopt() == "/casadi/libipopt"
    assert library.initialize_ipopt() == "/casadi/libipopt"
    assert calls == [("use", native), ("smoke", None)]


def test_initialize_latches_failure(monkeypatch):
    """A failed native initialization is never retried in the same process."""
    reset_initialization(monkeypatch)
    failure = library.IpoptAbiError("bad ABI")
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(library, "read_ipopt_header", fail)
    for _ in range(2):
        with pytest.raises(library.IpoptAbiError) as excinfo:
            library.initialize_ipopt()
        assert excinfo.value is failure
    assert calls == 1


def test_initialize_requires_matching_header(monkeypatch):
    """Verified initialization must fail closed before loading without a header."""
    reset_initialization(monkeypatch)
    monkeypatch.setattr(library, "read_ipopt_header", lambda: None)

    def unexpected_load():
        pytest.fail("native library was loaded before its ABI could be verified")

    monkeypatch.setattr(library, "load_ipopt", unexpected_load)
    with pytest.raises(library.IpoptAbiError, match="IpoptConfig.h is required"):
        library.initialize_ipopt()


def test_initialize_rejects_recursion(monkeypatch):
    """Recursive entry cannot start a second initialization sequence."""
    reset_initialization(monkeypatch)
    monkeypatch.setattr(library, "_initialization_state", "initializing")
    with pytest.raises(RuntimeError, match="recursive"):
        library.initialize_ipopt()


# ------------------------------------------------------------ smoke test ---


@requires_mseipopt
def test_smoke_test_passes_against_the_real_library():
    """The end-to-end ctypes configuration must actually work."""
    library.initialize_ipopt()
    library.smoke_test()


@requires_mseipopt
def test_initialize_is_idempotent():
    """Repeated initialization must not reload or remap anything."""
    first = library.initialize_ipopt()
    assert library.initialize_ipopt() == first
    copies = library.duplicate_ipopt_copies()
    if copies:
        assert len(copies) == 1
