"""High-Level, pythonic, safe and easy IPOPT interface."""

import functools
import inspect
from typing import Any

import numpy as np

from . import bare_np


class Problem(bare_np.Problem):
    def __init__(
        self,
        x_bounds: Any,
        g_bounds: Any,
        f: Any,
        g: Any,
        grad: Any,
        jac: Any,
        nele_jac: Any,
        hess: Any = None,
        nele_hess: Any = None,
    ):
        if hess is not None and nele_hess is None:
            raise TypeError("'nele_hess' must be given if 'hess' is supplied")
        if hess is None:
            nele_hess = 0

        # Create callbacks
        f_cb = f_callback(f)
        grad_cb = grad_callback(grad)
        g_cb = g_callback(g)
        jac_cb = jac_callback(jac, self)
        hess_cb = hess_callback(hess, self)

        handler = self._callback_exception_handler
        super().__init__(
            x_bounds,
            g_bounds,
            nele_jac,
            nele_hess,
            0,
            f_cb,
            g_cb,
            grad_cb,
            jac_cb,
            hess_cb,
            handler=handler,
        )
        self.set_intermediate_callback(self._intermediate_callback)

    def _intermediate_callback(self, *args):
        return 0 if getattr(self, "_abort", False) else 1

    def _callback_exception_handler(self, e):
        if isinstance(e, InvalidPoint):
            return
        if isinstance(e, KeyboardInterrupt):
            self._abort = True
            return
        bare_np.default_handler(e)

    def solve(
        self,
        x: Any,
        mult_g: Any = None,
        mult_x_L: Any = None,
        mult_x_U: Any = None,
        copy: bool = True,
    ):
        if any(m is not None for m in (mult_g, mult_x_L, mult_x_U)):
            self.add_str_option("warm_start_init_point", "yes")
        else:
            self.add_str_option("warm_start_init_point", "no")

        mult_g = np.zeros(self.m) if mult_g is None else mult_g
        mult_x_L = np.zeros(self.n) if mult_x_L is None else mult_x_L
        mult_x_U = np.zeros(self.n) if mult_x_U is None else mult_x_U

        g = np.empty(self.m)
        obj_val = np.empty(())
        if copy:
            x = np.array(x, np.double, copy=True, order="C")

        status = super().solve(x, g, obj_val, mult_g, mult_x_L, mult_x_U)
        info = dict(
            g=g, obj_val=obj_val, mult_g=mult_g, mult_x_L=mult_x_L, mult_x_U=mult_x_U, status=status
        )
        return x, info


class InvalidPoint(RuntimeError):
    """Exception raised in callback to signal an invalid decision by IPOPT."""


def f_callback(f):
    @functools.wraps(f)
    def wrapper(x, new_x, obj_value):
        obj_value[()] = f(x)
        return 1

    return wrapper


def grad_callback(grad):
    @functools.wraps(grad)
    def wrapper(x, new_x, grad_array):
        grad_array[()] = grad(x)
        return 1

    return wrapper


def g_callback(g):
    def wrapper(x, new_x, g_array):
        if g_array.size:
            g_array[()] = g(x)
        return 1

    return wrapper


def jac_callback(jac, problem):
    jac_ind, jac_val = jac

    def wrapper(x, new_x, iRow, jCol, values):
        # Fill out Jacobian values
        if values is not None:
            if values.size == 0:
                return 1
            if accepts_output(jac_val):
                jac_val(x, out=values)
            else:
                values[...] = jac_val(x)
            return 1

        # Fill out jacobian indices
        if iRow is None or jCol is None or iRow.size == 0 or jCol.size == 0:
            return 1
        i, j = jac_ind() if callable(jac_ind) else jac_ind
        iRow[...] = i
        jCol[...] = j
        # Check indices for overflow, can lead to segfault
        assert np.all(iRow < problem.m), "row index overflow"
        assert np.all(jCol < problem.n), "column index overflow"
        return 1

    return wrapper


def hess_callback(hess, problem):
    if hess is None:
        return

    hess_ind, hess_val = hess

    def wrapper(x, new_x, obj_factor, mult, new_mult, iRow, jCol, values):
        # Fill out Hessian values
        if values is not None:
            if values.size == 0:
                return 1
            if accepts_output(hess_val):
                hess_val(x, obj_factor, mult, out=values)
            else:
                values[...] = hess_val(x, obj_factor, mult)
            return 1

        # Fill out jacobian indices
        if iRow is None or jCol is None or iRow.size == 0 or jCol.size == 0:
            return 1
        i, j = hess_ind() if callable(hess_ind) else hess_ind
        iRow[...] = i
        jCol[...] = j
        # Check indices for overflow, can lead to segfault
        assert np.all(iRow < problem.n), "row index overflow"
        assert np.all(jCol < problem.n), "column index overflow"
        return 1

    return wrapper


@functools.lru_cache()
def accepts_output(f):
    params = inspect.signature(f).parameters
    out = params.get("out", None)
    if out is None:
        return False

    # First parameter cannot be the output
    if list(params).index("out") == 0:
        return False

    kinds = inspect.Parameter
    return out.kind == kinds.POSITIONAL_OR_KEYWORD or kinds.KEYWORD_ONLY
