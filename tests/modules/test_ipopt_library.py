"""

Test yapss._private.mseipopt.library, the IPOPT resolver and ABI checks.

These run in the ordinary pytest process. Anything that depends on load
*order* lives in test_ipopt_isolation.py instead, because a shared library
cannot be unmapped once loaded, so those cases need a fresh interpreter.

"""

import builtins
import ctypes
import io
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace

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


def test_linux_mapper_parses_paths_spaces_and_deleted_suffix(monkeypatch):
    """The procfs parser keeps complete absolute paths and deduplicates them."""
    maps = io.StringIO(
        "1000-2000 r-xp 0 00:00 1 /opt/a path/libipopt.so.3 (deleted)\n"
        "malformed line\n"
        "2000-3000 r--p 0 00:00 2 [heap]\n"
        "3000-4000 r-xp 0 00:00 3 /opt/libother.so\n"
    )
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: maps)
    assert library._mapped_paths_linux() == [
        "/opt/a path/libipopt.so.3",
        "/opt/libother.so",
    ]


def test_linux_mapper_records_read_failure(monkeypatch):
    """Unavailable procfs produces an empty unknown result and a diagnostic."""
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("denied")),
    )
    assert library._mapped_paths_linux() == []
    assert "denied" in library.strategy1_failure_reason()


def test_darwin_mapper_handles_unavailable_and_valid_dyld(monkeypatch):
    """dyld introspection handles both missing symbols and returned image names."""
    monkeypatch.setattr(library.ctypes, "CDLL", lambda value: object())
    assert library._mapped_paths_darwin() == []
    assert "dyld image list unavailable" in library.strategy1_failure_reason()

    class Dyld:
        _dyld_image_count = FakeFunction(3)
        _dyld_get_image_name = FakeFunction()

    dyld = Dyld()

    # Special-method lookup is class-based, so use a small callable holder.
    class ImageName(FakeFunction):
        def __call__(self, index):
            return (b"/a/libipopt.dylib", None, b"/b/libother.dylib")[index]

    dyld._dyld_get_image_name = ImageName()
    monkeypatch.setattr(library.ctypes, "CDLL", lambda value: dyld)
    assert library._mapped_paths_darwin() == [
        "/a/libipopt.dylib",
        "/b/libother.dylib",
    ]


def test_windows_mapper_handles_unavailable_api(monkeypatch):
    """Calling the Windows mapper elsewhere records that WinDLL is unavailable."""
    monkeypatch.delattr(library.ctypes, "WinDLL", raising=False)
    assert library._mapped_paths_windows() == []
    assert "WinDLL is unavailable" in library.strategy1_failure_reason()


def test_windows_mapper_enumerates_module_paths(monkeypatch):
    """The psapi adapter declares signatures, sizes the array, and reads each path."""

    class Function:
        def __init__(self, operation):
            self.operation = operation

        def __call__(self, *args):
            return self.operation(*args)

    class Kernel32:
        GetCurrentProcess = Function(lambda: 123)

    paths = iter(("C:/libs/ipopt.dll", "C:/libs/other.dll"))

    def enum_modules(handle, modules, byte_count, needed):
        needed._obj.value = 2 * ctypes.sizeof(wintypes.HMODULE)
        return 1

    def module_name(handle, module, buffer, length):
        buffer.value = next(paths)
        return len(buffer.value)

    class Psapi:
        EnumProcessModules = Function(enum_modules)
        GetModuleFileNameExW = Function(module_name)

    monkeypatch.setattr(
        library.ctypes,
        "WinDLL",
        lambda name, use_last_error: Psapi() if name == "psapi" else Kernel32(),
        raising=False,
    )
    monkeypatch.setattr(library.ctypes, "get_last_error", lambda: 0, raising=False)
    assert library._mapped_paths_windows() == [
        "C:/libs/ipopt.dll",
        "C:/libs/other.dll",
    ]


def test_windows_mapper_reports_dll_and_enumeration_failures(monkeypatch):
    """Windows API setup and enumeration failures remain diagnosable."""
    monkeypatch.setattr(
        library.ctypes,
        "WinDLL",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing psapi")),
        raising=False,
    )
    assert library._mapped_paths_windows() == []
    assert "missing psapi" in library.strategy1_failure_reason()

    class Function:
        def __init__(self, result):
            self.result = result

        def __call__(self, *args):
            return self.result

    kernel = SimpleNamespace(GetCurrentProcess=Function(123))
    psapi = SimpleNamespace(
        EnumProcessModules=Function(0),
        GetModuleFileNameExW=Function(0),
    )
    monkeypatch.setattr(
        library.ctypes,
        "WinDLL",
        lambda name, use_last_error: psapi if name == "psapi" else kernel,
        raising=False,
    )
    monkeypatch.setattr(library.ctypes, "get_last_error", lambda: 5, raising=False)
    assert library._mapped_paths_windows() == []
    assert "EnumProcessModules failed: 5" in library.strategy1_failure_reason()


def test_loaded_paths_dispatch_and_duplicate_filter(monkeypatch):
    """Platform dispatch feeds only actual Ipopt libraries to the duplicate guard."""
    monkeypatch.setattr(library, "_platform_key", lambda: "linux")
    monkeypatch.setattr(
        library,
        "_mapped_paths_linux",
        lambda: ["/x/libipopt.so", "/x/libipopt.so", "/x/libsipopt.so"],
    )
    assert library.duplicate_ipopt_copies() == ["/x/libipopt.so"]


def test_casadi_package_discovery_and_glob_fallback(monkeypatch, tmp_path):
    """Package discovery tolerates bad specs and the glob chooses its first match."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert library.casadi_package_dir() is None
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ValueError("partially imported")),
    )
    assert library.casadi_package_dir() is None

    (tmp_path / "libipopt.so.3").touch()
    (tmp_path / "libipopt.so.4").touch()
    monkeypatch.setattr(library, "casadi_package_dir", lambda: tmp_path)
    monkeypatch.setattr(library, "_platform_key", lambda: "linux")
    assert library.glob_ipopt_in_casadi() == str(tmp_path / "libipopt.so.3")
    monkeypatch.setattr(library, "casadi_package_dir", lambda: None)
    assert library.glob_ipopt_in_casadi() is None
    assert library.read_ipopt_header() is None


def test_resolver_cache_and_glob_strategy(monkeypatch):
    """A fallback resolution is cached and avoids repeating either strategy."""
    calls = []
    monkeypatch.setattr(library, "_resolved_path", None)
    monkeypatch.setattr(library, "casadi_ipopt_path", lambda: calls.append("loader") or None)
    monkeypatch.setattr(
        library,
        "glob_ipopt_in_casadi",
        lambda: calls.append("glob") or "/casadi/libipopt.so.3",
    )
    assert library.resolve_ipopt_library() == "/casadi/libipopt.so.3"
    assert library.resolve_ipopt_library() == "/casadi/libipopt.so.3"
    assert calls == ["loader", "glob"]


def test_path_description_handles_stat_success_and_failure(monkeypatch, tmp_path):
    """Duplicate diagnostics include identity when available and degrade cleanly."""
    path = tmp_path / "libipopt.so"
    path.write_bytes(b"native")
    description = library._describe(str(path))
    assert "device=" in description
    assert "size=6" in description
    assert "could not stat" in library._describe(str(tmp_path / "missing.so"))


# ------------------------------------------------------- strategy 1 alive ---


def test_casadi_probe_records_import_and_plugin_failures(monkeypatch):
    """Both unavailable-CasADi cases fail softly while retaining their reason."""
    original_import = builtins.__import__

    def missing_casadi(name, *args, **kwargs):
        if name == "casadi":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_casadi)
    assert library._force_casadi_ipopt_load() is False
    assert "could not be imported" in library.strategy1_failure_reason()

    monkeypatch.setattr(builtins, "__import__", original_import)

    class SX:
        @staticmethod
        def sym(name):
            return 2

    fake_casadi = SimpleNamespace(
        SX=SX,
        nlpsol=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plugin absent")),
    )
    monkeypatch.setitem(__import__("sys").modules, "casadi", fake_casadi)
    assert library._force_casadi_ipopt_load() is False
    assert "plugin absent" in library.strategy1_failure_reason()


def test_casadi_ipopt_path_handles_probe_and_mapping_misses(monkeypatch):
    """Strategy one distinguishes probe failure from an unobservable loaded image."""
    monkeypatch.setattr(library, "_force_casadi_ipopt_load", lambda: False)
    assert library.casadi_ipopt_path() is None

    monkeypatch.setattr(library, "_force_casadi_ipopt_load", lambda: True)
    monkeypatch.setattr(library, "_loaded_library_paths", lambda: ["/x/libcasadi.so"])
    assert library.casadi_ipopt_path() is None
    assert "no IPOPT library is mapped" in library.strategy1_failure_reason()


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


def test_guard_handles_duplicate_already_present_without_new_path():
    """The unattributed duplicate branch still explains the unsafe state."""
    duplicate_listing = ["/x/libipopt.so", "/x/libipopt.so"]
    with pytest.raises(library.DuplicateIpoptLibraryError, match="not introduced by YAPSS"):
        library._check_single_copy(duplicate_listing, duplicate_listing)


def test_load_ipopt_checks_mapping_boundary(monkeypatch):
    """Loading samples mappings immediately around CDLL and verifies the result."""
    native = object()
    mappings = iter([["/casadi/libipopt.so.3"], ["/casadi/libipopt.so.3"]])
    checked = []
    monkeypatch.setattr(library, "resolve_ipopt_library", lambda: "/casadi/libipopt.so.3")
    monkeypatch.setattr(library, "duplicate_ipopt_copies", lambda: next(mappings))
    monkeypatch.setattr(library.ctypes, "CDLL", lambda path: native)
    monkeypatch.setattr(
        library, "_check_single_copy", lambda before, after: checked.append((before, after))
    )

    assert library.load_ipopt() == (native, "/casadi/libipopt.so.3")
    assert checked == [(["/casadi/libipopt.so.3"], ["/casadi/libipopt.so.3"])]


def test_load_ipopt_wraps_loader_failure_with_context(monkeypatch):
    """A dependency-load failure names the selected path and probe context."""
    monkeypatch.setattr(library, "resolve_ipopt_library", lambda: "/broken/libipopt.so")
    monkeypatch.setattr(library, "duplicate_ipopt_copies", lambda: [])
    monkeypatch.setattr(library, "strategy1_failure_reason", lambda: "dyld unavailable")
    monkeypatch.setattr(
        library.ctypes,
        "CDLL",
        lambda path: (_ for _ in ()).throw(OSError("missing dependency")),
    )
    with pytest.raises(library.IpoptLibraryNotFoundError) as excinfo:
        library.load_ipopt()
    assert "/broken/libipopt.so" in str(excinfo.value)
    assert "dyld unavailable" in str(excinfo.value)


def test_windows_load_closes_temporary_dll_directory(monkeypatch, tmp_path):
    """The Windows dependency-search cookie is closed after a successful load."""
    path = tmp_path / "ipopt.dll"
    path.touch()
    closed = []

    class Cookie:
        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        library,
        "os",
        SimpleNamespace(name="nt", add_dll_directory=lambda directory: Cookie()),
    )
    monkeypatch.setattr(library, "resolve_ipopt_library", lambda: str(path))
    monkeypatch.setattr(library, "duplicate_ipopt_copies", lambda: [])
    monkeypatch.setattr(library.ctypes, "CDLL", lambda selected: object())
    monkeypatch.setattr(library, "_check_single_copy", lambda before, after: None)
    library.load_ipopt()
    assert closed == [True]


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

    def __init__(self, result=1):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


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


def test_bare_load_library_delegates_to_ctypes_loader(monkeypatch):
    """The explicit expert API loads exactly the requested library."""
    native = object()
    adopted = []
    monkeypatch.setattr(bare, "_ipopt_lib", None)
    monkeypatch.setattr(bare.ctypes.cdll, "LoadLibrary", lambda name: native)
    monkeypatch.setattr(bare, "use_library", adopted.append)

    bare.load_library("/chosen/libipopt.so")
    assert adopted == [native]


def test_bare_load_library_rejects_reconfiguration(monkeypatch):
    """The path-loading API honors the same immutable binding rule."""
    monkeypatch.setattr(bare, "_ipopt_lib", object())
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        bare.load_library("/other/libipopt.so")


def test_bare_unconfigured_creation_has_actionable_error(monkeypatch):
    """Raw creation fails explicitly instead of guessing a shared library."""
    monkeypatch.setattr(bare, "_ipopt_lib", None)
    with pytest.raises(RuntimeError, match="will not choose one"):
        bare.CreateIpoptProblem(1, None, None, 0, None, None, 0, 0, 0, *([None] * 5))


def test_bare_wrappers_encode_and_forward_arguments(monkeypatch):
    """Every raw convenience wrapper forwards to the configured function table."""
    native = FakeIpoptLibrary()
    native.IpoptSolve.result = -13
    monkeypatch.setattr(bare, "_ipopt_lib", native)
    problem = object()

    bare.FreeIpoptProblem(problem)
    assert native.FreeIpoptProblem.calls == [(problem,)]
    assert bare.AddIpoptStrOption(problem, "sb", "yes") == 1
    assert native.AddIpoptStrOption.calls[-1] == (problem, b"sb", b"yes")
    assert bare.AddIpoptStrOption(problem, b"mu_strategy", b"adaptive") == 1
    assert native.AddIpoptStrOption.calls[-1] == (problem, b"mu_strategy", b"adaptive")
    assert bare.AddIpoptNumOption(problem, "tol", 1e-8) == 1
    assert native.AddIpoptNumOption.calls[-1] == (problem, b"tol", 1e-8)
    assert bare.AddIpoptNumOption(problem, b"acceptable_tol", 1e-6) == 1
    assert bare.AddIpoptIntOption(problem, "max_iter", 7) == 1
    assert native.AddIpoptIntOption.calls[-1] == (problem, b"max_iter", 7)
    assert bare.AddIpoptIntOption(problem, b"print_level", 0) == 1
    assert bare.OpenIpoptOutputFile(problem, "ipopt.log", 5) == 1
    assert native.OpenIpoptOutputFile.calls[-1] == (problem, b"ipopt.log", 5)
    assert bare.OpenIpoptOutputFile(problem, b"second.log", 2) == 1
    assert bare.SetIpoptProblemScaling(problem, 2.0, "x", "g") == 1
    assert native.SetIpoptProblemScaling.calls[-1] == (problem, 2.0, "x", "g")
    assert bare.SetIntermediateCallback(problem, "callback") == 1
    assert native.SetIntermediateCallback.calls[-1] == (problem, "callback")
    assert bare.IpoptSolve(problem, "x", "g", "f", "mg", "ml", "mu", "data") == -13
    assert native.IpoptSolve.calls[-1] == (
        problem,
        "x",
        "g",
        "f",
        "mg",
        "ml",
        "mu",
        "data",
    )
    assert bare.GetIpoptCurrentIterate(problem, False, 1, "x", "zl", "zu", 2, "g", "mg") == 1
    assert (
        bare.GetIpoptCurrentViolations(
            problem,
            True,
            1,
            "xlv",
            "xuv",
            "cl",
            "cu",
            "grad",
            2,
            "gv",
            "cg",
        )
        == 1
    )


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


def test_require_initialized_distinguishes_ready_failed_and_uninitialized(monkeypatch):
    """The NumPy-layer gate preserves a failed initializer as the exception cause."""
    monkeypatch.setattr(library, "_initialization_state", "ready")
    library.require_initialized()

    failure = library.IpoptAbiError("bad ABI")
    monkeypatch.setattr(library, "_initialization_state", "failed")
    monkeypatch.setattr(library, "_initialization_error", failure)
    with pytest.raises(RuntimeError, match="not initialized") as excinfo:
        library.require_initialized()
    assert excinfo.value.__cause__ is failure

    monkeypatch.setattr(library, "_initialization_state", "uninitialized")
    with pytest.raises(RuntimeError, match="initialize_ipopt"):
        library.require_initialized()


# ------------------------------------------------------------ smoke test ---


def install_smoke_stubs(monkeypatch, *, problem=object(), status=0, corrupt_callback=False):
    """Replace native smoke entry points while retaining its real ctypes callbacks."""
    freed = []

    def create(*args):
        if corrupt_callback:
            x = (ctypes.c_double * 3)()
            assert args[9](3, x, True, bare.c_double_p(), None) is False
        return problem

    monkeypatch.setattr(bare, "CreateIpoptProblem", create)
    monkeypatch.setattr(bare, "AddIpoptIntOption", lambda *args: 1)
    monkeypatch.setattr(bare, "AddIpoptStrOption", lambda *args: 1)
    monkeypatch.setattr(bare, "IpoptSolve", lambda *args: status)
    monkeypatch.setattr(bare, "FreeIpoptProblem", freed.append)
    return freed


def test_smoke_test_rejects_null_problem(monkeypatch):
    """A null native constructor result becomes an ABI error and is not freed."""
    freed = install_smoke_stubs(monkeypatch, problem=None)
    with pytest.raises(library.IpoptAbiError, match="returned NULL"):
        library.smoke_test()
    assert freed == []


def test_smoke_test_rejects_status_and_wrong_solution_and_always_frees(monkeypatch):
    """Both semantic failure modes release the native problem exactly once."""
    freed = install_smoke_stubs(monkeypatch, status=-13)
    with pytest.raises(library.IpoptAbiError, match="status -13"):
        library.smoke_test()
    assert len(freed) == 1

    freed = install_smoke_stubs(monkeypatch, status=0)
    with pytest.raises(library.IpoptAbiError, match="analytic solution"):
        library.smoke_test()
    assert len(freed) == 1


def test_smoke_test_reraises_guarded_callback_failure(monkeypatch):
    """A callback error crosses no C frame and wins over the solve result."""
    freed = install_smoke_stubs(monkeypatch, status=0, corrupt_callback=True)
    with pytest.raises(ValueError, match="NULL pointer"):
        library.smoke_test()
    assert len(freed) == 1


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
