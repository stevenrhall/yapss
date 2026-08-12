# Copyright (c) 2018 Centro de Estudos Aeronáuticos da UFMG
# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Derived from mseipopt (https://github.com/cea-ufmg/mseipopt), modified by
# the YAPSS authors. Original and modified portions are both under the MIT
# license; see the LICENSE file and the LICENSES directory at the repository
# root.

"""NumPy-facing Ipopt wrapper with memory, lifetime, and exception safety.

User callbacks receive NumPy views of Ipopt buffers and return actual Boolean
results. Python exceptions are retained while C owns the stack and re-raised
from ``solve()`` after Ipopt returns; they never unwind through a ctypes
callback frame.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, TypeAlias

import numpy as np
from numpy.ctypeslib import as_array
from numpy.typing import ArrayLike, NDArray

from . import bare, library

_C_INT_INFO = np.iinfo(np.intc)
_USER_REQUESTED_STOP = 5

FloatArray: TypeAlias = NDArray[np.float64]
IndexArrayLike: TypeAlias = Sequence[int] | NDArray[np.integer[Any]]
SparsityStructure: TypeAlias = tuple[IndexArrayLike, IndexArrayLike]
CallbackResult: TypeAlias = bool | np.bool_
EvaluationCallback: TypeAlias = Callable[[FloatArray, bool, FloatArray], CallbackResult]
HessianCallback: TypeAlias = Callable[
    [FloatArray, bool, float, FloatArray, bool, FloatArray], CallbackResult
]


class InvalidPoint(RuntimeError):
    """Signal an intentional, callback-specific evaluation failure to Ipopt."""


def _limited_memory_hessian(*args: Any) -> bool:
    """Satisfy the C interface's non-null callback requirement; Ipopt must not call it."""
    return False


@dataclass(frozen=True)
class SolveResult:
    """Ipopt status and the exact in/out arrays used by one native solve."""

    status: int
    x: NDArray[np.float64]
    g: NDArray[np.float64]
    obj_val: NDArray[np.float64]
    mult_g: NDArray[np.float64]
    mult_x_L: NDArray[np.float64]
    mult_x_U: NDArray[np.float64]

    def copy(self) -> SolveResult:
        """Return a deep snapshot that does not alias any native-call buffer."""
        return SolveResult(
            status=self.status,
            x=self.x.copy(),
            g=self.g.copy(),
            obj_val=self.obj_val.copy(),
            mult_g=self.mult_g.copy(),
            mult_x_L=self.mult_x_L.copy(),
            mult_x_U=self.mult_x_U.copy(),
        )


def _validate_dimension(value: int, name: str) -> None:
    if value > _C_INT_INFO.max:
        raise OverflowError(f"{name} does not fit in the supported C int index type")


def _validate_bounds(
    lower: Any, upper: Any, name: str, *, allow_empty: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lower_array = np.require(lower, dtype=np.float64, requirements=["A", "C", "O"])
    upper_array = np.require(upper, dtype=np.float64, requirements=["A", "C", "O"])
    if lower_array.ndim != 1 or upper_array.ndim != 1:
        raise ValueError(f"{name} bounds must be one-dimensional")
    if lower_array.shape != upper_array.shape:
        raise ValueError(f"{name} lower and upper bounds must have identical shapes")
    if not allow_empty and lower_array.size == 0:
        raise ValueError("at least one decision variable is required")
    if np.isnan(lower_array).any() or np.isnan(upper_array).any():
        raise ValueError(f"{name} bounds must not contain NaN")
    if np.any(lower_array > upper_array):
        raise ValueError(f"{name} lower bounds must not exceed upper bounds")
    return lower_array, upper_array


def _validate_structure(
    structure: Any,
    *,
    row_limit: int,
    column_limit: int,
    name: str,
    lower_triangular: bool = False,
) -> tuple[NDArray[np.intc], NDArray[np.intc]]:
    try:
        rows_input, columns_input = structure
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}_structure must be a pair of index arrays") from exc
    rows = np.asarray(rows_input)
    columns = np.asarray(columns_input)
    if rows.ndim != 1 or columns.ndim != 1:
        raise ValueError(f"{name} structure arrays must be one-dimensional")
    if rows.shape != columns.shape:
        raise ValueError(f"{name} row and column arrays must have equal length")
    invalid_rows = rows.dtype.kind not in "iu" and (rows.size or isinstance(rows_input, np.ndarray))
    invalid_columns = columns.dtype.kind not in "iu" and (
        columns.size or isinstance(columns_input, np.ndarray)
    )
    if invalid_rows or invalid_columns:
        raise TypeError(f"{name} structure indices must be integers, not Boolean or floating point")
    if rows.size > _C_INT_INFO.max:
        raise OverflowError(f"{name} nonzero count does not fit in C int")
    for values, axis, limit in (
        (rows, "row", row_limit),
        (columns, "column", column_limit),
    ):
        if values.size and (np.any(values < _C_INT_INFO.min) or np.any(values > _C_INT_INFO.max)):
            raise OverflowError(f"{name} {axis} index does not fit in C int")
        if values.size and (np.any(values < 0) or np.any(values >= limit)):
            raise ValueError(f"{name} {axis} index is out of range")
    if lower_triangular and rows.size and np.any(rows < columns):
        raise ValueError("hessian structure must contain only lower-triangular entries")
    return (
        np.require(rows, dtype=np.intc, requirements=["A", "C", "O"]),
        np.require(columns, dtype=np.intc, requirements=["A", "C", "O"]),
    )


class Problem:
    """Own one native Ipopt problem and its NumPy callback boundary.

    Bounds and sparse structures are copied into owned, aligned, contiguous
    arrays during construction. Sparse indices use zero-based ``numpy.intc``
    storage and retain the caller's order and duplicates. Ordinary construction
    requires :func:`library.initialize_ipopt` to have completed successfully.

    Evaluation callbacks receive views of Ipopt-owned buffers and must return
    ``bool`` or ``numpy.bool_``. Unexpected exceptions are retained and re-raised
    from :meth:`solve` only after Ipopt returns. :class:`InvalidPoint` reports an
    intentional evaluation failure, while a first callback
    :class:`KeyboardInterrupt` requests a graceful status-5 stop.

    Omitting both ``hessian_structure`` and ``eval_h`` selects Ipopt's
    limited-memory Hessian approximation. Supplying one requires supplying the
    other, and Hessian structure entries must be lower triangular.
    """

    def __init__(
        self,
        x_l: ArrayLike,
        x_u: ArrayLike,
        g_l: ArrayLike,
        g_u: ArrayLike,
        *,
        eval_f: EvaluationCallback,
        eval_g: EvaluationCallback,
        eval_grad_f: EvaluationCallback,
        jacobian_structure: SparsityStructure,
        eval_jac_g: EvaluationCallback,
        hessian_structure: SparsityStructure | None = None,
        eval_h: HessianCallback | None = None,
        _unsafe_allow_unverified_library: bool = False,
    ) -> None:
        if not _unsafe_allow_unverified_library:
            library.require_initialized()
        callbacks = {
            "eval_f": eval_f,
            "eval_g": eval_g,
            "eval_grad_f": eval_grad_f,
            "eval_jac_g": eval_jac_g,
        }
        for name, callback in callbacks.items():
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        if (hessian_structure is None) != (eval_h is None):
            raise TypeError("hessian_structure and eval_h must be supplied together")
        if eval_h is not None and not callable(eval_h):
            raise TypeError("eval_h must be callable")

        x_l_array, x_u_array = _validate_bounds(x_l, x_u, "x", allow_empty=False)
        g_l_array, g_u_array = _validate_bounds(g_l, g_u, "g", allow_empty=True)
        self.n = n = x_l_array.size
        self.m = m = g_l_array.size
        _validate_dimension(n, "n")
        _validate_dimension(m, "m")

        jac_rows, jac_cols = _validate_structure(
            jacobian_structure, row_limit=m, column_limit=n, name="jacobian"
        )
        hess_rows: NDArray[np.intc]
        hess_cols: NDArray[np.intc]
        if hessian_structure is None:
            hess_rows = np.empty(0, dtype=np.intc)
            hess_cols = np.empty(0, dtype=np.intc)
        else:
            hess_rows, hess_cols = _validate_structure(
                hessian_structure,
                row_limit=n,
                column_limit=n,
                name="hessian",
                lower_triangular=True,
            )

        self._jacobian_structure = (jac_rows, jac_cols)
        self._hessian_structure = (hess_rows, hess_cols)
        self._problem = None
        self._solving = False
        self._callback_exception: tuple[BaseException, TracebackType | None, str, str] | None = None
        self._cancel_requested = False
        self._cancellation_context: tuple[str, str] | None = None
        c_callbacks = {
            "eval_f": wrap_f(eval_f, self._invoke_callback),
            "eval_g": wrap_g(eval_g, self._invoke_callback),
            "eval_grad_f": wrap_grad_f(eval_grad_f, self._invoke_callback),
            "eval_jac_g": wrap_jac_g(eval_jac_g, jac_rows, jac_cols, self._invoke_callback),
            "eval_h": (
                wrap_h(eval_h, hess_rows, hess_cols, self._invoke_callback)
                if eval_h is not None
                else wrap_h(
                    _limited_memory_hessian,
                    hess_rows,
                    hess_cols,
                    self._invoke_callback,
                )
            ),
        }
        self._callbacks = c_callbacks
        problem = bare.CreateIpoptProblem(
            n,
            data_ptr(x_l_array),
            data_ptr(x_u_array),
            m,
            data_ptr(g_l_array),
            data_ptr(g_u_array),
            len(jac_rows),
            len(hess_rows),
            0,
            c_callbacks["eval_f"],
            c_callbacks["eval_g"],
            c_callbacks["eval_grad_f"],
            c_callbacks["eval_jac_g"],
            c_callbacks["eval_h"],
        )
        if not problem:
            raise RuntimeError("Error creating IPOPT problem")
        self._problem = problem
        try:
            if eval_h is None:
                self.add_str_option("hessian_approximation", "limited-memory")
        except BaseException:
            bare.FreeIpoptProblem(problem)
            self._problem = None
            raise

    def _termination_pending(self) -> bool:
        return self._callback_exception is not None or self._cancel_requested

    def _latch_exception(self, error: BaseException, callback_name: str, phase: str) -> None:
        if not self._termination_pending():
            try:
                error.mseipopt_callback = callback_name  # type: ignore[attr-defined]
                error.mseipopt_phase = phase  # type: ignore[attr-defined]
            except BaseException:
                # Exception subclasses can theoretically forbid attributes;
                # traceback preservation and re-raise remain authoritative.
                pass
            self._callback_exception = (error, error.__traceback__, callback_name, phase)

    def _invoke_callback(
        self,
        callback_name: str,
        phase: str,
        operation: Callable[[], Any],
        *,
        allow_invalid_point: bool = True,
    ) -> bool:
        if self._termination_pending():
            return False
        try:
            result = operation()
            if not isinstance(result, (bool, np.bool_)):
                raise TypeError(
                    f"{callback_name} must return bool or numpy.bool_, got "
                    f"{type(result).__name__}"
                )
            return bool(result)
        except InvalidPoint as error:
            if allow_invalid_point:
                return False
            self._latch_exception(error, callback_name, phase)
            return False
        except KeyboardInterrupt:
            if not self._termination_pending():
                self._cancel_requested = True
                self._cancellation_context = (callback_name, phase)
            return False
        except BaseException as error:
            self._latch_exception(error, callback_name, phase)
            return False

    def _clear_termination(self) -> None:
        self._callback_exception = None
        self._cancel_requested = False
        self._cancellation_context = None

    def _require_open(self) -> None:
        if self._problem is None:
            raise RuntimeError("Problem is closed")

    def close(self) -> None:
        """Free the owned native problem once.

        Repeated calls do nothing. Closing during an active solve is rejected,
        because Ipopt may still use the problem and its pinned callbacks.
        """
        if self._problem is None:
            return
        if self._solving:
            raise RuntimeError("cannot close a problem while solve() is active")
        problem = self._problem
        self._problem = None
        bare.FreeIpoptProblem(problem)
        self._callbacks.clear()

    def free(self) -> None:
        """Alias for :meth:`close` retained for low-level compatibility."""
        Problem.close(self)

    def add_str_option(self, keyword: str, val: Any) -> None:
        """Set one Ipopt string option, raising if Ipopt rejects it."""
        self._require_open()
        if not bare.AddIpoptStrOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def add_int_option(self, keyword: str, val: Any) -> None:
        """Set one Ipopt integer option, raising if Ipopt rejects it."""
        self._require_open()
        if not bare.AddIpoptIntOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def add_num_option(self, keyword: str, val: Any) -> None:
        """Set one Ipopt floating-point option, raising if Ipopt rejects it."""
        self._require_open()
        if not bare.AddIpoptNumOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def open_output_file(self, file_name: str, print_level: int) -> None:
        """Direct Ipopt diagnostic output to a file at the requested level."""
        self._require_open()
        if not bare.OpenIpoptOutputFile(self._problem, file_name, print_level):
            raise RuntimeError("error opening output file")

    def set_scaling(self, obj_scaling: Any, x_scaling: Any, g_scaling: Any) -> None:
        """Install objective, variable, and constraint user scaling.

        Variable and constraint arrays must have lengths ``n`` and ``m``;
        accepted inputs are converted to aligned ``float64`` arrays for this
        native call. The Ipopt scaling method is set to ``user-scaling``.
        """
        self._require_open()
        x_scaling = np.require(x_scaling, np.double, ["A", "C"])
        g_scaling = np.require(g_scaling, np.double, ["A", "C"])

        if x_scaling.shape != (self.n,):
            raise ValueError("invalid shape for the x scaling")
        if g_scaling.shape != (self.m,):
            raise ValueError("invalid shape for the g scaling")

        obj_s = float(obj_scaling)
        x_s = data_ptr(x_scaling)
        g_s = data_ptr(g_scaling)
        if not bare.SetIpoptProblemScaling(self._problem, obj_s, x_s, g_s):
            raise RuntimeError("error setting problem scaling")
        self.add_str_option("nlp_scaling_method", "user-scaling")

    def set_intermediate_callback(self, cb: Any | None) -> None:
        """Install, replace, or disable the iteration callback.

        The callable is strongly retained only after Ipopt accepts it. Passing
        ``None`` disables the native callback and releases the previous strong
        reference only after native success.
        """
        self._require_open()
        if cb is None:
            if not bare.SetIntermediateCallback(self._problem, None):
                raise RuntimeError("error disabling problem intermediate callback")
            self._callbacks.pop("intermediate_cb", None)
            return
        if not callable(cb):
            raise TypeError("intermediate callback must be callable")
        intermediate_cb = wrap_intermediate_cb(cb, self._invoke_callback)
        if not bare.SetIntermediateCallback(self._problem, intermediate_cb):
            raise RuntimeError("error setting problem intermediate callback")
        self._callbacks["intermediate_cb"] = intermediate_cb

    def solve(
        self,
        x: Any,
        g: Any = None,
        obj_val: Any = None,
        mult_g: Any = None,
        mult_x_L: Any = None,
        mult_x_U: Any = None,
    ) -> SolveResult:
        """Solve in place using validated native-call buffers.

        ``x`` and every supplied output must be a writable, aligned,
        C-contiguous ``float64`` NumPy array of the exact required shape.
        Omitted outputs are allocated. The returned :class:`SolveResult`
        references these exact arrays; call :meth:`SolveResult.copy` for a
        non-aliasing snapshot.

        A solve cannot recurse or overlap another solve on this problem.
        Unexpected callback exceptions are re-raised with their traceback after
        native unwinding. Graceful callback cancellation returns status 5 and
        the current unconverged buffers.
        """
        self._require_open()
        if self._solving:
            raise RuntimeError("solve() is already active for this problem")
        _validate_io_array(x, (self.n,), "x")
        g = np.empty(self.m, dtype=np.float64) if g is None else g
        obj_val = np.empty((), dtype=np.float64) if obj_val is None else obj_val
        mult_g = np.zeros(self.m, dtype=np.float64) if mult_g is None else mult_g
        mult_x_L = np.zeros(self.n, dtype=np.float64) if mult_x_L is None else mult_x_L
        mult_x_U = np.zeros(self.n, dtype=np.float64) if mult_x_U is None else mult_x_U
        _validate_io_array(g, (self.m,), "g")
        _validate_io_array(obj_val, (), "obj_val")
        _validate_io_array(mult_g, (self.m,), "mult_g")
        _validate_io_array(mult_x_L, (self.n,), "mult_x_L")
        _validate_io_array(mult_x_U, (self.n,), "mult_x_U")
        self._clear_termination()
        self._solving = True
        try:
            status = bare.IpoptSolve(
                self._problem,
                data_ptr(x),
                data_ptr(g),
                data_ptr(obj_val),
                data_ptr(mult_g),
                data_ptr(mult_x_L),
                data_ptr(mult_x_U),
                None,
            )
        finally:
            self._solving = False
        failure = self._callback_exception
        cancelled = self._cancel_requested
        self._clear_termination()
        if failure is not None:
            error, traceback, _, _ = failure
            raise error.with_traceback(traceback)
        if cancelled:
            status = _USER_REQUESTED_STOP
        return SolveResult(status, x, g, obj_val, mult_g, mult_x_L, mult_x_U)

    def __enter__(self) -> Problem:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def wrap_f(f: Any, invoke: Callable[..., bool]) -> Any:
    """Wrap a NumPy objective callback in Ipopt's objective callback type."""

    # Wrapped in a plain call rather than `@bare.Eval_F_CB` as a decorator --
    # CFUNCTYPE's dynamically-generated type isn't precisely known to mypy,
    # so using it as a decorator makes the decorated function untyped
    # ([misc] "Untyped decorator makes function untyped"). Calling it
    # explicitly on an already-fully-typed `wrapper` avoids that, and is
    # behaviorally identical -- ctypes doesn't care whether the conversion
    # happens via decorator syntax or a plain call.
    @functools.wraps(f)
    def wrapper(n: Any, x: Any, new_x: Any, obj_value: Any, user_data: Any) -> Any:
        def operation() -> Any:
            x_array = as_array(x, (n,))
            obj_value_array = as_array(obj_value, ())
            return f(x_array, bool(new_x), obj_value_array)

        return invoke("eval_f", "values", operation)

    return bare.Eval_F_CB(wrapper)


def wrap_grad_f(grad_f: Any, invoke: Callable[..., bool]) -> Any:
    """Wrap a NumPy objective-gradient callback in Ipopt's callback type."""

    @functools.wraps(grad_f)
    def wrapper(n: Any, x: Any, new_x: Any, grad_ptr: Any, user_data: Any) -> Any:
        def operation() -> Any:
            x_array = as_array(x, (n,))
            grad_f_array = as_array(grad_ptr, (n,))
            return grad_f(x_array, bool(new_x), grad_f_array)

        return invoke("eval_grad_f", "values", operation)

    return bare.Eval_Grad_F_CB(wrapper)


def wrap_g(g: Any, invoke: Callable[..., bool]) -> Any:
    """Wrap a NumPy constraint callback in Ipopt's constraint callback type."""

    @functools.wraps(g)
    def wrapper(n: Any, x: Any, new_x: Any, m: Any, g_ptr: Any, user_data: Any) -> Any:
        def operation() -> Any:
            x_array = as_array(x, (n,))
            g_array = as_array(g_ptr, (m,)) if m else np.empty(0, dtype=np.float64)
            return g(x_array, bool(new_x), g_array)

        return invoke("eval_g", "values", operation)

    return bare.Eval_G_CB(wrapper)


def wrap_jac_g(
    jac_g: Any,
    rows: NDArray[np.intc],
    columns: NDArray[np.intc],
    invoke: Callable[..., bool],
) -> Any:
    """Wrap Jacobian values and copy owned indices for structure requests."""

    @functools.wraps(jac_g)
    def wrapper(
        n: Any,
        x: Any,
        new_x: Any,
        m: Any,
        nele_jac: Any,
        iRow: Any,
        jCol: Any,
        values: Any,
        user_data: Any,
    ) -> Any:
        # A null ctypes pointer is a falsy pointer object, not necessarily None.
        x_present = bool(x)
        rows_present = bool(iRow)
        columns_present = bool(jCol)
        values_present = bool(values)
        structure_request = not values_present and (
            (rows_present and columns_present) or (nele_jac == 0 and not x_present)
        )
        values_request = (
            x_present
            and not rows_present
            and not columns_present
            and (values_present or nele_jac == 0)
        )

        def operation() -> Any:
            if nele_jac != len(rows):
                raise RuntimeError("native Jacobian nonzero count does not match the problem")
            if structure_request:
                if nele_jac:
                    as_array(iRow, (nele_jac,))[...] = rows
                    as_array(jCol, (nele_jac,))[...] = columns
                return True
            if values_request:
                values_array = (
                    as_array(values, (nele_jac,)) if nele_jac else np.empty(0, dtype=np.float64)
                )
                return jac_g(as_array(x, (n,)), bool(new_x), values_array)
            raise RuntimeError(
                "invalid native Jacobian callback pointer combination: "
                f"x={x_present}, iRow={rows_present}, jCol={columns_present}, "
                f"values={values_present}"
            )

        structure_phase = not values_present and (rows_present or columns_present or not x_present)
        phase = "structure" if structure_phase else "values"
        return invoke(
            "eval_jac_g",
            phase,
            operation,
            allow_invalid_point=phase == "values",
        )

    return bare.Eval_Jac_G_CB(wrapper)


def wrap_h(
    h: Any,
    rows: NDArray[np.intc],
    columns: NDArray[np.intc],
    invoke: Callable[..., bool],
) -> Any:
    """Wrap Hessian values and copy owned indices for structure requests."""

    @functools.wraps(h)
    def wrapper(
        n: Any,
        x: Any,
        new_x: Any,
        obj_factor: Any,
        m: Any,
        mult: Any,
        new_mult: Any,
        nele_hess: Any,
        iRow: Any,
        jCol: Any,
        values: Any,
        user_data: Any,
    ) -> Any:
        # A null ctypes pointer is a falsy pointer object, not necessarily None.
        x_present = bool(x)
        multipliers_present = bool(mult)
        rows_present = bool(iRow)
        columns_present = bool(jCol)
        values_present = bool(values)
        structure_request = not values_present and (
            (rows_present and columns_present) or (nele_hess == 0 and not x_present)
        )
        values_request = (
            x_present
            and (multipliers_present or m == 0)
            and not rows_present
            and not columns_present
            and (values_present or nele_hess == 0)
        )

        def operation() -> Any:
            if nele_hess != len(rows):
                raise RuntimeError("native Hessian nonzero count does not match the problem")
            if structure_request:
                if nele_hess:
                    as_array(iRow, (nele_hess,))[...] = rows
                    as_array(jCol, (nele_hess,))[...] = columns
                return True
            if values_request:
                values_array = (
                    as_array(values, (nele_hess,)) if nele_hess else np.empty(0, dtype=np.float64)
                )
                mult_array = as_array(mult, (m,)) if m else np.empty(0, dtype=np.float64)
                return h(
                    as_array(x, (n,)),
                    bool(new_x),
                    obj_factor,
                    mult_array,
                    bool(new_mult),
                    values_array,
                )
            raise RuntimeError(
                "invalid native Hessian callback pointer combination: "
                f"x={x_present}, lambda={multipliers_present}, iRow={rows_present}, "
                f"jCol={columns_present}, values={values_present}"
            )

        structure_phase = not values_present and (rows_present or columns_present or not x_present)
        phase = "structure" if structure_phase else "values"
        return invoke("eval_h", phase, operation, allow_invalid_point=phase == "values")

    return bare.Eval_H_CB(wrapper)


def wrap_intermediate_cb(cb: Any, invoke: Callable[..., bool]) -> Any:
    """Wrap the Python iteration callback in Ipopt's intermediate callback type."""

    @functools.wraps(cb)
    def wrapper(
        alg_mod: Any,
        iter_count: Any,
        obj_value: Any,
        inf_pr: Any,
        inf_du: Any,
        mu: Any,
        d_norm: Any,
        regularization_size: Any,
        alpha_du: Any,
        alpha_pr: Any,
        ls_trials: Any,
        user_data: Any,
    ) -> Any:
        def operation() -> Any:
            return cb(
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            )

        return invoke(
            "intermediate",
            "iteration",
            operation,
            allow_invalid_point=False,
        )

    return bare.Intermediate_CB(wrapper)


def _validate_io_array(a: Any, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray instance")
    if a.dtype != np.double:
        raise TypeError(f"{name} must be an array of float64 values")
    if a.shape != shape:
        raise ValueError(f"invalid shape for {name}: expected {shape}, got {a.shape}")
    if not a.flags.aligned:
        raise ValueError(f"{name} must be aligned")
    if not a.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not a.flags.writeable:
        raise ValueError(f"{name} must be writeable")


def data_ptr(arr: NDArray[np.float64] | None) -> Any:
    """Return a safe ``double *``, or null for ``None`` and empty arrays."""
    if arr is None:
        return arr
    if not isinstance(arr, np.ndarray):
        raise TypeError("native double buffer must be a numpy ndarray instance")
    if arr.dtype != np.double:
        raise TypeError("native double buffer must contain float64 values")
    if not arr.flags.aligned:
        raise ValueError("native double buffer must be aligned")
    if not arr.flags.c_contiguous:
        raise ValueError("native double buffer must be C-contiguous")
    return arr.ctypes.data_as(bare.c_double_p) if arr.size else None
