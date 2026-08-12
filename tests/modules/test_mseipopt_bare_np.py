"""Contract tests for the hardened NumPy-facing mseipopt constructor."""

import ctypes
import gc
import weakref
from typing import Any, get_type_hints

import numpy as np
import pytest

from yapss._private.config import get_conda_prefix
from yapss._private.mseipopt import bare, bare_np, library

requires_mseipopt = pytest.mark.skipif(
    bool(get_conda_prefix()),
    reason="conda uses cyipopt; the vendored CasADi interface is not the active backend",
)


@pytest.fixture
def native(monkeypatch):
    """Replace native entry points while retaining the real ctypes callbacks."""
    calls = {"create": [], "free": [], "options": []}
    problem_pointer = object()
    monkeypatch.setattr(library, "require_initialized", lambda: None)

    def create(*args):
        calls["create"].append(args)
        return problem_pointer

    monkeypatch.setattr(bare, "CreateIpoptProblem", create)
    monkeypatch.setattr(bare, "FreeIpoptProblem", calls["free"].append)

    def add_option(problem, keyword, value):
        calls["options"].append((problem, keyword, value))
        return 1

    monkeypatch.setattr(bare, "AddIpoptStrOption", add_option)
    monkeypatch.setattr(bare, "SetIntermediateCallback", lambda *args: 1)
    return calls, problem_pointer


def callbacks(jacobian=None, hessian=None):
    """Return the minimal successful callback set."""
    if jacobian is None:
        jacobian = lambda x, new_x, out: True
    result = {
        "eval_f": lambda x, new_x, out: True,
        "eval_g": lambda x, new_x, out: True,
        "eval_grad_f": lambda x, new_x, out: True,
        "eval_jac_g": jacobian,
    }
    if hessian is not None:
        result["eval_h"] = hessian
    return result


def make_problem(**overrides):
    """Construct a small valid problem, applying keyword overrides."""
    arguments = {
        "x_l": [-1.0, -2.0],
        "x_u": [1.0, 2.0],
        "g_l": [0.0],
        "g_u": [0.0],
        "jacobian_structure": ([0, 0], [1, 0]),
        **callbacks(),
    }
    arguments.update(overrides)
    return bare_np.Problem(**arguments)


def test_public_constructor_parameters_have_specific_types():
    """Every value accepted by the public constructor has a concrete annotation."""
    hints = get_type_hints(bare_np.Problem.__init__)
    public_parameters = {
        "x_l",
        "x_u",
        "g_l",
        "g_u",
        "eval_f",
        "eval_g",
        "eval_grad_f",
        "jacobian_structure",
        "eval_jac_g",
        "hessian_structure",
        "eval_h",
    }
    assert public_parameters <= hints.keys()
    assert all(hints[name] is not Any for name in public_parameters)


def test_problem_requires_verified_initialization(monkeypatch):
    """Raw bare configuration alone must not authorize the NumPy layer."""
    failure = RuntimeError("not verified")
    monkeypatch.setattr(library, "require_initialized", lambda: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(
        bare,
        "CreateIpoptProblem",
        lambda *args: pytest.fail("entered native creation before initialization"),
    )
    with pytest.raises(RuntimeError) as excinfo:
        make_problem()
    assert excinfo.value is failure


def test_private_unverified_opt_out_is_solver_only_escape_hatch(monkeypatch, native):
    """The deprecated explicit-path route can construct after raw bare setup."""
    monkeypatch.setattr(
        library,
        "require_initialized",
        lambda: (_ for _ in ()).throw(RuntimeError("not verified")),
    )
    problem = bare_np.Problem(
        [-1.0],
        [1.0],
        [],
        [],
        eval_f=lambda *args: True,
        eval_g=lambda *args: True,
        eval_grad_f=lambda *args: True,
        jacobian_structure=([], []),
        eval_jac_g=lambda *args: True,
        _unsafe_allow_unverified_library=True,
    )
    assert problem.n == 1


@pytest.mark.parametrize("name", ["eval_f", "eval_g", "eval_grad_f", "eval_jac_g"])
def test_constructor_rejects_noncallable_required_callbacks(native, name):
    """Every required callback is checked before native problem creation."""
    calls, _ = native
    with pytest.raises(TypeError, match=f"{name} must be callable"):
        make_problem(**{name: None})
    assert calls["create"] == []


def test_constructor_rejects_noncallable_exact_hessian(native):
    """Exact-Hessian structure cannot be paired with a non-callable value."""
    calls, _ = native
    with pytest.raises(TypeError, match="eval_h must be callable"):
        make_problem(hessian_structure=([0], [0]), eval_h=3)
    assert calls["create"] == []


def test_dimension_overflow_is_rejected():
    """Dimensions outside Ipopt's configured C-int ABI fail before narrowing."""
    with pytest.raises(OverflowError, match="does not fit"):
        bare_np._validate_dimension(np.iinfo(np.intc).max + 1, "n")


def test_internal_null_pointer_and_limited_memory_helpers():
    """Empty arrays map to null pointers and the dummy Hessian always rejects use."""
    assert bare_np.data_ptr(None) is None
    assert bare_np.data_ptr(np.empty(0)) is None
    assert bare_np._limited_memory_hessian() is False


def test_data_ptr_rejects_unsafe_native_buffers():
    """The final pointer boundary enforces its invariants without assertions."""
    with pytest.raises(TypeError, match="numpy ndarray"):
        bare_np.data_ptr([1.0])
    with pytest.raises(TypeError, match="float64"):
        bare_np.data_ptr(np.ones(1, dtype=np.float32))

    storage = bytearray(np.dtype(np.float64).itemsize + 1)
    unaligned = np.ndarray((1,), dtype=np.float64, buffer=storage, offset=1)
    with pytest.raises(ValueError, match="aligned"):
        bare_np.data_ptr(unaligned)

    with pytest.raises(ValueError, match="C-contiguous"):
        bare_np.data_ptr(np.arange(4.0)[::2])


def test_constructor_infers_counts_and_owns_structures(native):
    """Structure order and duplicates survive in independent intc storage."""
    calls, _ = native
    rows = np.array([0, 0, 0], dtype=np.int64)
    columns = np.array([1, 0, 1], dtype=np.int64)
    problem = make_problem(jacobian_structure=(rows, columns))

    create = calls["create"][0]
    assert create[0] == 2
    assert create[3] == 1
    assert create[6:9] == (3, 0, 0)
    owned_rows, owned_columns = problem._jacobian_structure
    assert owned_rows.dtype == np.intc
    assert owned_rows.flags.owndata and owned_rows.flags.c_contiguous and owned_rows.flags.aligned
    rows[:] = 99
    columns[:] = 99
    np.testing.assert_array_equal(owned_rows, [0, 0, 0])
    np.testing.assert_array_equal(owned_columns, [1, 0, 1])


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"x_l": [], "x_u": []}, "at least one"),
        ({"x_l": [[0.0]], "x_u": [[1.0]]}, "one-dimensional"),
        ({"x_l": [0.0], "x_u": [1.0, 2.0]}, "identical shapes"),
        ({"x_l": [np.nan, 0.0]}, "must not contain NaN"),
        ({"x_l": [2.0, 0.0], "x_u": [1.0, 2.0]}, "must not exceed"),
    ],
)
def test_invalid_bounds_fail_before_native(native, overrides, match):
    calls, _ = native
    with pytest.raises(ValueError, match=match):
        make_problem(**overrides)
    assert calls["create"] == []


@pytest.mark.parametrize(
    ("structure", "error", "match"),
    [
        (None, TypeError, "pair of index arrays"),
        (([0], [0], [0]), TypeError, "pair of index arrays"),
        (([0], [0, 1]), ValueError, "equal length"),
        (([[0]], [[0]]), ValueError, "one-dimensional"),
        (([False], [0]), TypeError, "must be integers"),
        (([0.0], [0]), TypeError, "must be integers"),
        (([-1], [0]), ValueError, "out of range"),
        (([1], [0]), ValueError, "out of range"),
        (([0], [2]), ValueError, "out of range"),
        (([0], [np.iinfo(np.int64).max]), OverflowError, "does not fit"),
    ],
)
def test_invalid_jacobian_structure_fails_before_native(native, structure, error, match):
    calls, _ = native
    with pytest.raises(error, match=match):
        make_problem(jacobian_structure=structure)
    assert calls["create"] == []


def test_hessian_pair_and_triangle_are_validated_before_native(native):
    calls, _ = native
    with pytest.raises(TypeError, match="supplied together"):
        make_problem(hessian_structure=([0], [0]))
    with pytest.raises(TypeError, match="supplied together"):
        make_problem(eval_h=lambda *args: True)
    with pytest.raises(ValueError, match="lower-triangular"):
        make_problem(
            hessian_structure=([0], [1]),
            eval_h=lambda x, new_x, factor, mult, new_mult, out: True,
        )
    assert calls["create"] == []


def test_limited_memory_setup_failure_frees_owned_problem_once(native, monkeypatch):
    calls, problem_pointer = native
    monkeypatch.setattr(bare, "AddIpoptStrOption", lambda *args: 0)
    with pytest.raises(ValueError, match="invalid option"):
        make_problem()
    assert calls["free"] == [problem_pointer]


def test_null_creation_does_not_free(native, monkeypatch):
    calls, _ = native
    monkeypatch.setattr(bare, "CreateIpoptProblem", lambda *args: None)
    with pytest.raises(RuntimeError, match="creating IPOPT"):
        make_problem()
    assert calls["free"] == []


def test_close_is_idempotent_and_closed_operations_fail(native):
    calls, problem_pointer = native
    problem = make_problem()
    problem.close()
    problem.close()
    assert calls["free"] == [problem_pointer]
    with pytest.raises(RuntimeError, match="closed"):
        problem.add_num_option("tol", 1e-8)


def test_free_alias_releases_problem_once(native):
    """The compatibility alias delegates to idempotent close semantics."""
    calls, problem_pointer = native
    problem = make_problem()
    problem.free()
    problem.free()
    assert calls["free"] == [problem_pointer]


def test_option_and_output_methods_translate_native_results(native, monkeypatch):
    """Integer/numeric options and output files expose native rejection cleanly."""
    problem = make_problem()
    int_calls = []
    num_calls = []
    output_calls = []
    monkeypatch.setattr(bare, "AddIpoptIntOption", lambda *args: int_calls.append(args) or 1)
    monkeypatch.setattr(bare, "AddIpoptNumOption", lambda *args: num_calls.append(args) or 1)
    monkeypatch.setattr(bare, "OpenIpoptOutputFile", lambda *args: output_calls.append(args) or 1)
    problem.add_int_option("max_iter", 10)
    problem.add_num_option("tol", 1e-8)
    problem.open_output_file("ipopt.log", 4)
    assert int_calls[-1][1:] == ("max_iter", 10)
    assert num_calls[-1][1:] == ("tol", 1e-8)
    assert output_calls[-1][1:] == ("ipopt.log", 4)

    monkeypatch.setattr(bare, "AddIpoptIntOption", lambda *args: 0)
    monkeypatch.setattr(bare, "AddIpoptNumOption", lambda *args: 0)
    monkeypatch.setattr(bare, "OpenIpoptOutputFile", lambda *args: 0)
    with pytest.raises(ValueError, match="invalid option"):
        problem.add_int_option("bad", 1)
    with pytest.raises(ValueError, match="invalid option"):
        problem.add_num_option("bad", 1.0)
    with pytest.raises(RuntimeError, match="opening output"):
        problem.open_output_file("bad.log", 1)


def test_scaling_validates_shapes_and_native_result(native, monkeypatch):
    """Scaling checks dimensions, forwards float64 data, and enables the option."""
    calls, problem_pointer = native
    problem = make_problem()
    with pytest.raises(ValueError, match="x scaling"):
        problem.set_scaling(1.0, [1.0], [1.0])
    with pytest.raises(ValueError, match="g scaling"):
        problem.set_scaling(1.0, [1.0, 1.0], [])

    scaling_calls = []
    monkeypatch.setattr(
        bare,
        "SetIpoptProblemScaling",
        lambda *args: scaling_calls.append(args) or 1,
    )
    problem.set_scaling(2, [3, 4], [5])
    assert scaling_calls[0][0:2] == (problem_pointer, 2.0)
    assert calls["options"][-1][1:] == ("nlp_scaling_method", "user-scaling")

    monkeypatch.setattr(bare, "SetIpoptProblemScaling", lambda *args: 0)
    with pytest.raises(RuntimeError, match="setting problem scaling"):
        problem.set_scaling(1.0, [1.0, 1.0], [1.0])


def test_scaling_copies_strided_inputs_to_contiguous_native_buffers(native, monkeypatch):
    """Strided scaling views are copied before their pointers cross into C."""
    problem = make_problem(
        g_l=[0.0, 0.0],
        g_u=[0.0, 0.0],
        jacobian_structure=([0, 1], [1, 0]),
    )
    received = []

    def set_scaling(problem_pointer, objective, x_pointer, g_pointer):
        received.append(
            (
                np.ctypeslib.as_array(x_pointer, shape=(2,)).copy(),
                np.ctypeslib.as_array(g_pointer, shape=(2,)).copy(),
            )
        )
        return 1

    monkeypatch.setattr(bare, "SetIpoptProblemScaling", set_scaling)
    x_scaling = np.array([3.0, -1.0, 4.0, -1.0])[::2]
    g_scaling = np.array([5.0, -1.0, 6.0, -1.0])[::2]
    assert not x_scaling.flags.c_contiguous
    assert not g_scaling.flags.c_contiguous

    problem.set_scaling(2.0, x_scaling, g_scaling)

    np.testing.assert_array_equal(received[0][0], [3.0, 4.0])
    np.testing.assert_array_equal(received[0][1], [5.0, 6.0])


def test_close_and_recursive_solve_are_rejected_while_solving(native):
    problem = make_problem()
    problem._solving = True
    with pytest.raises(RuntimeError, match="while solve"):
        problem.close()
    with pytest.raises(RuntimeError, match="already active"):
        problem.solve(np.zeros(2))
    assert native[0]["free"] == []


def test_intermediate_callback_replacement_and_disable_update_lifetime(native, monkeypatch):
    installed = []
    monkeypatch.setattr(
        bare,
        "SetIntermediateCallback",
        lambda problem, callback: installed.append(callback) or 1,
    )
    problem = make_problem()

    class Callback:
        def __call__(self, *args):
            return True

    first = Callback()
    first_ref = weakref.ref(first)
    problem.set_intermediate_callback(first)
    del first
    gc.collect()
    assert first_ref() is not None

    second = Callback()
    second_ref = weakref.ref(second)
    problem.set_intermediate_callback(second)
    del second
    # Do not let this test's native-call spy itself pin obsolete callbacks.
    installed[:-1] = []
    gc.collect()
    assert first_ref() is None
    assert second_ref() is not None

    problem.set_intermediate_callback(None)
    installed.clear()
    gc.collect()
    assert second_ref() is None
    assert "intermediate_cb" not in problem._callbacks


def test_failed_intermediate_replacement_retains_previous_callback(native, monkeypatch):
    problem = make_problem()
    previous = lambda *args: True
    problem.set_intermediate_callback(previous)
    installed = problem._callbacks["intermediate_cb"]
    monkeypatch.setattr(bare, "SetIntermediateCallback", lambda *args: 0)

    with pytest.raises(RuntimeError, match="setting problem intermediate"):
        problem.set_intermediate_callback(lambda *args: False)
    assert problem._callbacks["intermediate_cb"] is installed

    with pytest.raises(RuntimeError, match="disabling problem intermediate"):
        problem.set_intermediate_callback(None)
    assert problem._callbacks["intermediate_cb"] is installed


def test_intermediate_callback_rejects_noncallable(native):
    """Only a callable or None may cross the intermediate callback boundary."""
    problem = make_problem()
    with pytest.raises(TypeError, match="must be callable"):
        problem.set_intermediate_callback(3)


@pytest.mark.parametrize("x_present", [False, True])
def test_structure_callback_copies_owned_data_without_calling_user(native, x_present):
    called = []

    def jacobian(x, new_x, values):
        called.append((x, new_x, values))
        return True

    problem = make_problem(eval_jac_g=jacobian)
    callback = problem._callbacks["eval_jac_g"]
    x_buffer = (ctypes.c_double * 2)()
    x_pointer = ctypes.cast(x_buffer, bare.c_double_p) if x_present else bare.c_double_p()
    row_buffer = (ctypes.c_int * 2)()
    column_buffer = (ctypes.c_int * 2)()
    result = callback(
        2,
        x_pointer,
        False,
        1,
        2,
        row_buffer,
        column_buffer,
        bare.c_double_p(),
        None,
    )
    assert result is True
    assert list(row_buffer) == [0, 0]
    assert list(column_buffer) == [1, 0]
    assert called == []


def test_hessian_structure_callback_copies_owned_data_without_user_call(native):
    """Exact Hessian structure comes solely from validated constructor metadata."""
    called = []

    def hessian(*args):
        called.append(args)
        return True

    problem = make_problem(
        hessian_structure=([0, 1], [0, 0]),
        eval_h=hessian,
    )
    callback = problem._callbacks["eval_h"]
    rows = (ctypes.c_int * 2)()
    columns = (ctypes.c_int * 2)()
    assert callback(
        2,
        bare.c_double_p(),
        False,
        1.0,
        1,
        bare.c_double_p(),
        False,
        2,
        rows,
        columns,
        bare.c_double_p(),
        None,
    )
    assert list(rows) == [0, 1]
    assert list(columns) == [0, 0]
    assert called == []


def test_exception_metadata_failure_does_not_hide_original(native):
    """An exception type that forbids attributes is still retained authoritatively."""

    class LockedError(RuntimeError):
        def __setattr__(self, name, value):
            if name.startswith("mseipopt_"):
                raise AttributeError(name)
            super().__setattr__(name, value)

    problem = make_problem()
    error = LockedError("locked")
    problem._latch_exception(error, "eval_f", "values")
    assert problem._callback_exception == (error, error.__traceback__, "eval_f", "values")


class CallbackError(RuntimeError):
    """Distinct error used to prove callback exception identity and traceback."""


def invoke_native_callback(problem, name):
    """Invoke one ctypes callback with valid native-shaped buffers."""
    callback = problem._callbacks[name]
    x = (ctypes.c_double * 2)()
    output_2 = (ctypes.c_double * 2)()
    output_1 = (ctypes.c_double * 1)()
    null_int = bare.c_int_p()
    if name == "eval_f":
        return callback(2, x, True, output_1, None)
    if name == "eval_g":
        return callback(2, x, True, 1, output_1, None)
    if name == "eval_grad_f":
        return callback(2, x, True, output_2, None)
    if name == "eval_jac_g":
        return callback(2, x, True, 1, 2, null_int, null_int, output_2, None)
    if name == "eval_h":
        multipliers = (ctypes.c_double * 1)()
        return callback(
            2,
            x,
            True,
            1.0,
            1,
            multipliers,
            True,
            2,
            null_int,
            null_int,
            output_2,
            None,
        )
    if name == "intermediate_cb":
        return callback(0, 1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 1.0, 0, None)
    raise AssertionError(f"unknown callback {name}")


@pytest.mark.parametrize(
    "callback_name",
    ["eval_f", "eval_g", "eval_grad_f", "eval_jac_g", "eval_h", "intermediate_cb"],
)
def test_callback_exception_is_reraised_after_native_unwind(native, monkeypatch, callback_name):
    error = CallbackError(callback_name)

    def fail(*args):
        raise error

    overrides = {}
    constructor_name = callback_name
    if callback_name == "eval_h":
        overrides.update(hessian_structure=([0, 1], [0, 1]), eval_h=fail)
    elif callback_name == "intermediate_cb":
        constructor_name = None
    else:
        overrides[callback_name] = fail
    problem = make_problem(**overrides)
    if callback_name == "intermediate_cb":
        problem.set_intermediate_callback(fail)

    def solve(*args):
        assert invoke_native_callback(problem, callback_name) is False
        expected_name = "intermediate" if callback_name == "intermediate_cb" else callback_name
        assert problem._callback_exception[2:] == (
            expected_name,
            "iteration" if callback_name == "intermediate_cb" else "values",
        )
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    with pytest.raises(CallbackError) as excinfo:
        problem.solve(np.zeros(2))
    assert excinfo.value is error
    assert "fail" in [entry.name for entry in excinfo.traceback]
    assert error.mseipopt_callback == (
        "intermediate" if callback_name == "intermediate_cb" else callback_name
    )
    assert error.mseipopt_phase == ("iteration" if callback_name == "intermediate_cb" else "values")
    assert problem._callback_exception is None
    assert constructor_name is None or callback_name in str(error)


@pytest.mark.parametrize("bad_result", [None, 0, 1, "yes"])
def test_callback_result_must_be_an_actual_boolean(native, monkeypatch, bad_result):
    problem = make_problem(eval_f=lambda *args: bad_result)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    with pytest.raises(TypeError, match="eval_f must return bool or numpy.bool_"):
        problem.solve(np.zeros(2))


@pytest.mark.parametrize("result", [False, np.bool_(False)])
def test_explicit_false_is_not_a_python_exception(native, monkeypatch, result):
    problem = make_problem(eval_f=lambda *args: result)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    assert problem.solve(np.zeros(2)).status == -13


def test_invalid_point_is_an_intentional_evaluation_failure(native, monkeypatch):
    def invalid(*args):
        raise bare_np.InvalidPoint("outside domain")

    problem = make_problem(eval_f=invalid)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    assert problem.solve(np.zeros(2)).status == -13


def test_invalid_point_in_intermediate_is_reraised(native, monkeypatch):
    error = bare_np.InvalidPoint("not an evaluation callback")

    def invalid(*args):
        raise error

    problem = make_problem()
    problem.set_intermediate_callback(invalid)

    def solve(*args):
        assert invoke_native_callback(problem, "intermediate_cb") is False
        return 5

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    with pytest.raises(bare_np.InvalidPoint) as excinfo:
        problem.solve(np.zeros(2))
    assert excinfo.value is error


def test_system_exit_is_preserved_as_an_unexpected_base_exception(native, monkeypatch):
    error = SystemExit("stop Python after native unwind")

    def exit_callback(*args):
        raise error

    problem = make_problem(eval_f=exit_callback)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    with pytest.raises(SystemExit) as excinfo:
        problem.solve(np.zeros(2))
    assert excinfo.value is error


def test_keyboard_interrupt_requests_graceful_intermediate_stop(native, monkeypatch):
    user_intermediate_calls = []

    def interrupt(*args):
        raise KeyboardInterrupt

    problem = make_problem(eval_f=interrupt)
    problem.set_intermediate_callback(lambda *args: user_intermediate_calls.append(args) or True)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        assert invoke_native_callback(problem, "intermediate_cb") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    assert problem.solve(np.zeros(2)).status == 5
    assert user_intermediate_calls == []


def test_first_callback_failure_wins_and_suppresses_later_user_code(native, monkeypatch):
    first = CallbackError("first")
    later_calls = []

    def fail(*args):
        raise first

    problem = make_problem(eval_f=fail, eval_g=lambda *args: later_calls.append(args) or True)

    def solve(*args):
        assert invoke_native_callback(problem, "eval_f") is False
        assert invoke_native_callback(problem, "eval_g") is False
        return -13

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    with pytest.raises(CallbackError) as excinfo:
        problem.solve(np.zeros(2))
    assert excinfo.value is first
    assert later_calls == []


@requires_mseipopt
def test_native_exception_and_interrupt_unwind_contract():
    """Exercise the latch through the real CasADi-bundled Ipopt call stack."""
    library.initialize_ipopt()

    def eval_g(x, new_x, out):
        return True

    def eval_grad_f(x, new_x, out):
        out[:] = 2 * (x - 1)
        return True

    def eval_jac_g(x, new_x, out):
        return True

    def construct(eval_f):
        problem = bare_np.Problem(
            [-10.0],
            [10.0],
            [],
            [],
            eval_f=eval_f,
            eval_g=eval_g,
            eval_grad_f=eval_grad_f,
            jacobian_structure=([], []),
            eval_jac_g=eval_jac_g,
        )
        problem.add_int_option("print_level", 0)
        problem.add_str_option("sb", "yes")
        problem.set_intermediate_callback(lambda *args: True)
        return problem

    error = CallbackError("through native Ipopt")

    def fail(x, new_x, out):
        raise error

    with construct(fail) as problem:
        with pytest.raises(CallbackError) as excinfo:
            problem.solve(np.zeros(1))
        assert excinfo.value is error

    def interrupt(x, new_x, out):
        raise KeyboardInterrupt

    with construct(interrupt) as problem:
        assert problem.solve(np.zeros(1)).status == 5


def test_solve_result_uses_exact_native_buffers(native, monkeypatch):
    problem = make_problem()
    x = np.array([0.25, 0.5])
    g = np.array([9.0])
    obj_val = np.array(8.0)
    mult_g = np.array([7.0])
    mult_x_l = np.array([6.0, 5.0])
    mult_x_u = np.array([4.0, 3.0])

    def solve(problem_pointer, x_ptr, g_ptr, obj_ptr, mg_ptr, ml_ptr, mu_ptr, user_data):
        assert mg_ptr[0] == 7.0
        assert list(ml_ptr[:2]) == [6.0, 5.0]
        assert list(mu_ptr[:2]) == [4.0, 3.0]
        x_ptr[0], x_ptr[1] = 1.0, 2.0
        g_ptr[0] = 3.0
        obj_ptr[0] = 4.0
        mg_ptr[0] = 5.0
        ml_ptr[0], ml_ptr[1] = 6.0, 7.0
        mu_ptr[0], mu_ptr[1] = 8.0, 9.0
        return 0

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    result = problem.solve(x, g, obj_val, mult_g, mult_x_l, mult_x_u)
    assert result.status == 0
    assert result.x is x
    assert result.g is g
    assert result.obj_val is obj_val
    assert result.mult_g is mult_g
    assert result.mult_x_L is mult_x_l
    assert result.mult_x_U is mult_x_u
    np.testing.assert_array_equal(result.x, [1.0, 2.0])
    np.testing.assert_array_equal(result.mult_x_U, [8.0, 9.0])


def test_solve_allocates_every_omitted_output(native, monkeypatch):
    problem = make_problem()
    monkeypatch.setattr(bare, "IpoptSolve", lambda *args: 5)
    x = np.zeros(2)
    result = problem.solve(x)
    assert result.x is x
    for array, shape in (
        (result.g, (1,)),
        (result.obj_val, ()),
        (result.mult_g, (1,)),
        (result.mult_x_L, (2,)),
        (result.mult_x_U, (2,)),
    ):
        assert array.shape == shape
        assert array.dtype == np.float64
        assert array.flags.aligned and array.flags.c_contiguous and array.flags.writeable


def test_solve_results_are_distinct_but_reused_buffers_alias(native, monkeypatch):
    problem = make_problem()
    calls = 0

    def solve(problem_pointer, x_ptr, *args):
        nonlocal calls
        calls += 1
        x_ptr[0] = calls
        return calls

    monkeypatch.setattr(bare, "IpoptSolve", solve)
    x = np.zeros(2)
    first = problem.solve(x)
    second = problem.solve(x)
    assert first is not second
    assert first.x is second.x is x
    assert first.x[0] == 2.0
    assert (first.status, second.status) == (1, 2)


def test_solve_result_copy_is_a_deep_snapshot(native, monkeypatch):
    problem = make_problem()
    monkeypatch.setattr(bare, "IpoptSolve", lambda *args: 0)
    result = problem.solve(np.array([1.0, 2.0]))
    snapshot = result.copy()
    assert snapshot.status == result.status
    for original, copied in (
        (result.x, snapshot.x),
        (result.g, snapshot.g),
        (result.obj_val, snapshot.obj_val),
        (result.mult_g, snapshot.mult_g),
        (result.mult_x_L, snapshot.mult_x_L),
        (result.mult_x_U, snapshot.mult_x_U),
    ):
        assert copied is not original
        np.testing.assert_array_equal(copied, original)
    result.x[:] = 99.0
    assert snapshot.x.tolist() == [1.0, 2.0]


def test_solve_rejects_noncontiguous_and_unsafe_buffers_before_native(native, monkeypatch):
    problem = make_problem()
    monkeypatch.setattr(
        bare, "IpoptSolve", lambda *args: pytest.fail("unsafe buffer reached native solve")
    )
    with pytest.raises(TypeError, match="numpy ndarray"):
        problem.solve([0.0, 0.0])
    with pytest.raises(TypeError, match="float64"):
        problem.solve(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="invalid shape"):
        problem.solve(np.zeros((2, 1)))

    readonly = np.zeros(2)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        problem.solve(readonly)

    strided = np.zeros(4)[::2]
    assert strided.shape == (2,) and not strided.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        problem.solve(strided)

    storage = bytearray(2 * np.dtype(np.float64).itemsize + 1)
    unaligned = np.ndarray((2,), dtype=np.float64, buffer=storage, offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        problem.solve(unaligned)

    valid_x = np.zeros(2)
    bad_multiplier = np.zeros(4)[::2]
    with pytest.raises(ValueError, match="mult_x_L must be C-contiguous"):
        problem.solve(valid_x, mult_x_L=bad_multiplier)
