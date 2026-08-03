"""Bare wrapper around IPOPT using numpy arrays and exception handling.

This module has an intermediate level of abstraction in the wrapping. Exception
handling is done in the callbacks and the error is signalled back to IPOPT. The
parameters in the callbacks are converted to ndarrays and the function inputs
are validated more thoroughly. However, the functions and callbacks map almost
directly to the IPOPT c interface.

"""

from __future__ import annotations

import functools
from types import TracebackType
from typing import Any, Callable

import numpy as np
from numpy.ctypeslib import as_array
from numpy.typing import NDArray

from . import bare


def default_handler(e: BaseException) -> None:
    """Exception handler for IPOPT ctypes callbacks, prints the traceback."""
    import traceback

    traceback.print_exc()


class Problem:
    def __init__(
        self,
        x_bounds: Any,
        g_bounds: Any,
        nele_jac: Any,
        nele_hess: Any,
        index_style: Any,
        f: Any,
        g: Any,
        grad_f: Any,
        jac_g: Any,
        h: Any = None,
        *,
        handler: Callable[[BaseException], Any] = default_handler,
    ) -> None:
        # Unpack and validate decision variable bounds
        x_L, x_U = x_bounds
        x_L = np.require(x_L, np.double, ["A", "C"])
        x_U = np.require(x_U, np.double, ["A", "C"])
        n = x_L.size
        if x_U.size != n:
            raise ValueError("Inconsistent sizes of 'x' lower and upper bounds")

        # Unpack and validate constraint bounds
        g_L, g_U = g_bounds
        g_L = np.require(g_L, np.double, ["A", "C"])
        g_U = np.require(g_U, np.double, ["A", "C"])
        m = g_L.size
        if g_U.size != m:
            raise ValueError("Inconsistent sizes of 'g' lower and upper bounds")

        # Wrap the callbacks
        eval_f = wrap_f(f, handler)
        eval_g = wrap_g(g, handler)
        eval_grad_f = wrap_grad_f(grad_f, handler)
        eval_jac_g = wrap_jac_g(jac_g, handler)
        eval_h = wrap_h(h, handler) if h is not None else bare.Eval_H_CB()
        problem = bare.CreateIpoptProblem(
            n,
            data_ptr(x_L),
            data_ptr(x_U),
            m,
            data_ptr(g_L),
            data_ptr(g_U),
            nele_jac,
            nele_hess,
            index_style,
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_h,
        )
        if not problem:
            raise RuntimeError("Error creating IPOPT problem")

        # Save object data
        self._problem = problem
        """Pointer to the underlying `IpoptProblemInfo` structure."""

        self.n = n
        """Number of decision variables (length of `x`)."""

        self.m = m
        """Number of constraints (length of `g`)."""

        self._callbacks = {
            "eval_f": eval_f,
            "eval_g": eval_g,
            "eval_grad_f": eval_grad_f,
            "eval_jac_g": eval_jac_g,
            "eval_h": eval_h,
        }
        """Reference to callbacks to ensure they aren't garbage collected."""

        # Set options
        if h is None:
            self.add_str_option("hessian_approximation", "limited-memory")

    def free(self) -> None:
        if not self._problem:
            raise RuntimeError("Problem invalid or already freed")
        bare.FreeIpoptProblem(self._problem)
        del self._callbacks
        self._problem = None

    def add_str_option(self, keyword: str, val: Any) -> None:
        if not bare.AddIpoptStrOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def add_int_option(self, keyword: str, val: Any) -> None:
        if not bare.AddIpoptIntOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def add_num_option(self, keyword: str, val: Any) -> None:
        if not bare.AddIpoptNumOption(self._problem, keyword, val):
            raise ValueError("invalid option or value")

    def open_output_file(self, file_name: str, print_level: int) -> None:
        if not bare.OpenIpoptOutputFile(self._problem, file_name, print_level):
            raise RuntimeError("error opening output file")

    def set_scaling(self, obj_scaling: Any, x_scaling: Any, g_scaling: Any) -> None:
        x_scaling = np.require(x_scaling, np.double, "A")
        g_scaling = np.require(g_scaling, np.double, "A")

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

    def set_intermediate_callback(self, cb: Any) -> None:
        intermediate_cb = wrap_intermediate_cb(cb)
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
    ) -> int:
        exc = (
            validate_io_array(x, (self.n,), "x", none_ok=False)
            or validate_io_array(g, (self.m,), "g")
            or validate_io_array(obj_val, (), "obj_val")
            or validate_io_array(mult_g, (self.m,), "mult_g")
            or validate_io_array(mult_x_L, (self.n,), "mult_x_L")
            or validate_io_array(mult_x_U, (self.n,), "mult_x_U")
        )
        if exc:
            raise exc
        return bare.IpoptSolve(
            self._problem,
            data_ptr(x),
            data_ptr(g),
            data_ptr(obj_val),
            data_ptr(mult_g),
            data_ptr(mult_x_L),
            data_ptr(mult_x_U),
            None,
        )

    def __enter__(self) -> Problem:
        if not self._problem:
            raise RuntimeError("Invalid context or reentering context.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.free()


def wrap_f(f: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
    # Wrapped in a plain call rather than `@bare.Eval_F_CB` as a decorator --
    # CFUNCTYPE's dynamically-generated type isn't precisely known to mypy,
    # so using it as a decorator makes the decorated function untyped
    # ([misc] "Untyped decorator makes function untyped"). Calling it
    # explicitly on an already-fully-typed `wrapper` avoids that, and is
    # behaviorally identical -- ctypes doesn't care whether the conversion
    # happens via decorator syntax or a plain call.
    @functools.wraps(f)
    def wrapper(n: Any, x: Any, new_x: Any, obj_value: Any, user_data: Any) -> Any:
        try:
            x_array = as_array(x, (n,))
            obj_value_array = as_array(obj_value, ())
            return f(x_array, new_x, obj_value_array)
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Eval_F_CB(wrapper)


def wrap_grad_f(grad_f: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
    @functools.wraps(grad_f)
    def wrapper(n: Any, x: Any, new_x: Any, grad_ptr: Any, user_data: Any) -> Any:
        try:
            x_array = as_array(x, (n,))
            grad_f_array = as_array(grad_ptr, (n,))
            return grad_f(x_array, new_x, grad_f_array)
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Eval_Grad_F_CB(wrapper)


def wrap_g(g: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
    @functools.wraps(g)
    def wrapper(n: Any, x: Any, new_x: Any, m: Any, g_ptr: Any, user_data: Any) -> Any:
        try:
            x_array = as_array(x, (n,))
            g_array = as_array(g_ptr, (m,))
            return g(x_array, new_x, g_array)
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Eval_G_CB(wrapper)


def wrap_jac_g(jac_g: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
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
        try:
            x_array = as_array(x, (n,)) if x else None
            i_array = as_array(iRow, (nele_jac,)) if iRow else None
            j_array = as_array(jCol, (nele_jac,)) if jCol else None
            values_array = as_array(values, (nele_jac,)) if values else None
            return jac_g(x_array, new_x, i_array, j_array, values_array)
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Eval_Jac_G_CB(wrapper)


def wrap_h(h: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
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
        try:
            x_array = as_array(x, (n,)) if x else None
            mult_array = as_array(mult, (m,)) if mult else None
            i_array = as_array(iRow, (nele_hess,)) if iRow else None
            j_array = as_array(jCol, (nele_hess,)) if jCol else None
            values_array = as_array(values, (nele_hess,)) if values else None
            return h(
                x_array, new_x, obj_factor, mult_array, new_mult, i_array, j_array, values_array
            )
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Eval_H_CB(wrapper)


def wrap_intermediate_cb(cb: Any, handler: Callable[[BaseException], Any] = default_handler) -> Any:
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
        try:
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
        except BaseException as e:
            if callable(handler):
                handler(e)
            return 0

    return bare.Intermediate_CB(wrapper)


def validate_io_array(
    a: Any, shape: tuple[int, ...], name: str, none_ok: bool = True
) -> TypeError | ValueError | None:
    if none_ok and a is None:
        return None
    if none_ok and not isinstance(a, np.ndarray):
        return TypeError(f"{name} must be a numpy ndarray instance or None")
    if not none_ok and not isinstance(a, np.ndarray):
        return TypeError(f"{name} must be a numpy ndarray instance")
    if a.dtype != np.double:
        return TypeError(f"{name} must be an array of doubles")
    if not (a.flags["A"] and a.flags["W"]):
        return ValueError(f"{name} must be an aligned writeable array")
    if a.shape != shape:
        return ValueError(f"invalid shape for {name}")
    return None


def data_ptr(arr: NDArray[np.float64] | None) -> Any:
    if arr is None:
        return arr
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.double
    return arr.ctypes.data_as(bare.c_double_p) if arr.size else None
