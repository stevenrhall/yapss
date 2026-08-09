# Copyright (c) 2018 Centro de Estudos Aeronáuticos da UFMG
# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Derived from mseipopt (https://github.com/cea-ufmg/mseipopt), modified by
# the YAPSS authors. Original and modified portions are both under the MIT
# license; see the LICENSE file at the repository root.

"""High-Level, pythonic, safe and easy IPOPT interface."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any

import numpy as np

from . import bare_np

logger = logging.getLogger(__name__)


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
    ) -> None:
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

    def _intermediate_callback(self, *args: Any) -> int:
        return 0 if getattr(self, "_abort", False) else 1

    def _callback_exception_handler(self, e: BaseException) -> None:
        if isinstance(e, InvalidPoint):
            return
        if isinstance(e, KeyboardInterrupt):
            self._abort = True
            return
        bare_np.default_handler(e)

    # noinspection PyMethodOverriding
    def solve(  # type: ignore[override]
        self,
        x: Any,
        mult_g: Any = None,
        mult_x_L: Any = None,
        mult_x_U: Any = None,
        copy: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        # Deliberately different signature/return type than
        # bare_np.Problem.solve() -- this is the friendly, high-level API
        # that wraps the raw one, not a Liskov-substitutable override.
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


def f_callback(f: Any) -> Any:
    @functools.wraps(f)
    def wrapper(x: Any, new_x: Any, obj_value: Any) -> int:
        obj_value[()] = f(x)
        return 1

    return wrapper


def grad_callback(grad: Any) -> Any:
    @functools.wraps(grad)
    def wrapper(x: Any, new_x: Any, grad_array: Any) -> int:
        grad_array[()] = grad(x)
        return 1

    return wrapper


def g_callback(g: Any) -> Any:
    def wrapper(x: Any, new_x: Any, g_array: Any) -> int:
        if g_array.size:
            g_array[()] = g(x)
        return 1

    return wrapper


def jac_callback(jac: Any, problem: Any) -> Any:
    jac_ind, jac_val = jac

    def wrapper(x: Any, new_x: Any, iRow: Any, jCol: Any, values: Any) -> int:
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

        # Check indices at BOTH ends. Ipopt hands these straight to the linear solver,
        # which indexes off them without validation, so an out-of-range entry is a
        # SIGSEGV in native code rather than an exception -- and a negative index is
        # just as fatal as one that is too large.
        i_arr = np.asarray(iRow)
        j_arr = np.asarray(jCol)
        logger.debug(
            "jac structure: %d entries, rows [%d, %d] of m=%d, cols [%d, %d] of n=%d",
            i_arr.size,
            i_arr.min(),
            i_arr.max(),
            problem.m,
            j_arr.min(),
            j_arr.max(),
            problem.n,
        )
        assert i_arr.min() >= 0, "negative row index"
        assert i_arr.max() < problem.m, "row index overflow"
        assert j_arr.min() >= 0, "negative column index"
        assert j_arr.max() < problem.n, "column index overflow"
        return 1

    return wrapper


def hess_callback(hess: Any, problem: Any) -> Any:
    if hess is None:
        return None

    hess_ind, hess_val = hess

    def wrapper(
        x: Any,
        new_x: Any,
        obj_factor: Any,
        mult: Any,
        new_mult: Any,
        iRow: Any,
        jCol: Any,
        values: Any,
    ) -> int:
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

        # As in `jac_callback`: check both ends, because the linear solver indexes off
        # these without validation. Additionally the Hessian must be lower triangular
        # (row >= col) -- Ipopt documents that requirement and does not enforce it.
        i_arr = np.asarray(iRow)
        j_arr = np.asarray(jCol)
        logger.debug(
            "hess structure: %d entries, rows [%d, %d], cols [%d, %d], n=%d, "
            "upper-triangular entries: %d",
            i_arr.size,
            i_arr.min(),
            i_arr.max(),
            j_arr.min(),
            j_arr.max(),
            problem.n,
            int(np.count_nonzero(i_arr < j_arr)),
        )
        assert i_arr.min() >= 0, "negative row index"
        assert i_arr.max() < problem.n, "row index overflow"
        assert j_arr.min() >= 0, "negative column index"
        assert j_arr.max() < problem.n, "column index overflow"
        assert np.all(i_arr >= j_arr), "Hessian structure is not lower triangular"
        return 1

    return wrapper


@functools.lru_cache()
def accepts_output(f: Any) -> bool:
    params = inspect.signature(f).parameters
    out = params.get("out", None)
    if out is None:
        return False

    # First parameter cannot be the output
    if list(params).index("out") == 0:
        return False

    kinds = inspect.Parameter
    return out.kind in (kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)
