# Copyright (c) 2018 Centro de Estudos Aeronáuticos da UFMG
# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Derived from mseipopt (https://github.com/cea-ufmg/mseipopt), modified by
# the YAPSS authors. Original and modified portions are both under the MIT
# license; see the LICENSE file at the repository root.

"""Bare wrapper around the Ipopt 3.14 C interface using ctypes.

The type names are the same as in `IpStdCInterface.h`, the argument names are
the same (except for lambda, which is a python keyword) and even the order of
the definitions in the module is the same. Very little checking or validation
is done. This module provides direct access to the ipopt c interface functions
using `ctypes`.

`load_library` (given an explicit path) or `use_library` (given an already
loaded library) must be called before creating a problem. This module does not
locate IPOPT itself; see `yapss._private.mseipopt.library` for why the choice of
file matters and must not be guessed.

The declarations mirror ``IpStdCInterface.h`` as shipped with Ipopt 3.14.11
and assume C ``bool``, double-precision numbers, and 32-bit indices. The
verified initializer checks those assumptions against CasADi's matching
headers before configuring this module. Configuration is immutable: a process
cannot replace the native function table after it has been selected.

After a problem is no longer needed, `FreeIpoptProblem` should be called, or
memory will leak. If `FreeIpoptProblem` is called more than once on the same
problem, the program will likely crash.
"""

from __future__ import annotations

import ctypes
from ctypes import CFUNCTYPE, POINTER, c_bool, c_char_p, c_double, c_int, c_void_p
from typing import Any

_ipopt_lib: ctypes.CDLL | None = None
"""library ctypes.CDLL object or None (if not loaded)."""


c_double_p = POINTER(c_double)
"""Pointer to double."""


c_int_p = POINTER(c_int)
"""Pointer to int."""


Bool = c_bool
"""The C type of IPOPT's ``Bool`` for the supported Ipopt 3.14+ ABI."""


Eval_F_CB = CFUNCTYPE(c_bool, c_int, c_double_p, c_bool, c_double_p, c_void_p)
"""Type of the callback for evaluating the objective function."""

Eval_Grad_F_CB = CFUNCTYPE(c_bool, c_int, c_double_p, c_bool, c_double_p, c_void_p)
"""Type of the callback for evaluating the gradient of the objective."""

Eval_G_CB = CFUNCTYPE(c_bool, c_int, c_double_p, c_bool, c_int, c_double_p, c_void_p)
"""Type of the callback for evaluating the constraint function."""

Eval_Jac_G_CB = CFUNCTYPE(
    c_bool,
    c_int,
    c_double_p,
    c_bool,
    c_int,
    c_int,
    c_int_p,
    c_int_p,
    c_double_p,
    c_void_p,
)
"""Type of the callback for evaluating the Jacobian of the constraint."""

Eval_H_CB = CFUNCTYPE(
    c_bool,
    c_int,
    c_double_p,
    c_bool,
    c_double,
    c_int,
    c_double_p,
    c_bool,
    c_int,
    c_int_p,
    c_int_p,
    c_double_p,
    c_void_p,
)
"""Type of the callback for evaluating the Hessian of the Lagrangian."""

Intermediate_CB = CFUNCTYPE(
    c_bool,
    c_int,
    c_int,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_double,
    c_int,
    c_void_p,
)
"""Type of the callback to give intermediate execution control to the user."""


class IpoptProblemInfo(ctypes.Structure):
    """Structure collecting all information about the problem."""


IpoptProblem = POINTER(IpoptProblemInfo)
"""Pointer to a IPOPT problem."""


def load_library(name: str) -> None:
    """Load the IPOPT shared library at *name* and configure its signatures.

    *name* is required. This function used to accept None and fall back to a
    platform-default name such as "libipopt.so", letting the dynamic loader
    supply whichever IPOPT it found first. That is unsafe when CasADi is also
    loaded -- see yapss._private.mseipopt.library -- so callers must now say
    exactly which file they mean.
    """
    if _ipopt_lib is not None:
        raise RuntimeError("IPOPT is already configured and cannot be replaced")
    use_library(ctypes.cdll.LoadLibrary(name))


def use_library(lib: ctypes.CDLL) -> None:
    """Adopt an already-loaded IPOPT library and configure its signatures.

    Lets the caller do the locating and loading -- and any verification it
    wants to perform on the result -- while this module remains responsible
    only for the ctypes declarations.
    """
    global _ipopt_lib
    if _ipopt_lib is not None:
        current_handle = getattr(_ipopt_lib, "_handle", id(_ipopt_lib))
        new_handle = getattr(lib, "_handle", id(lib))
        if current_handle == new_handle:
            return
        raise RuntimeError("IPOPT is already configured and cannot be replaced")
    _ipopt_lib = lib
    _setup_library()


def _setup_library() -> None:
    assert _ipopt_lib is not None, "cannot setup before loading"

    _ipopt_lib.CreateIpoptProblem.restype = IpoptProblem
    _ipopt_lib.CreateIpoptProblem.argtypes = [
        c_int,
        c_double_p,
        c_double_p,
        c_int,
        c_double_p,
        c_double_p,
        c_int,
        c_int,
        c_int,
        Eval_F_CB,
        Eval_G_CB,
        Eval_Grad_F_CB,
        Eval_Jac_G_CB,
        Eval_H_CB,
    ]

    _ipopt_lib.FreeIpoptProblem.restype = None
    _ipopt_lib.FreeIpoptProblem.argtypes = [IpoptProblem]

    # These six return IPOPT's `Bool`. Declaring them c_int when the library
    # returns a 1-byte _Bool is the one place the width mismatch could bite:
    # only `al` is architecturally meaningful for such a return, the rest of
    # `eax` is unspecified, and callers test the result for truth. Compilers
    # in practice zero-extend, so this never misfired -- but if one did not, a
    # *failed* option would read as success and be silently swallowed.
    _ipopt_lib.AddIpoptStrOption.restype = Bool
    _ipopt_lib.AddIpoptStrOption.argtypes = [IpoptProblem, c_char_p, c_char_p]

    _ipopt_lib.AddIpoptIntOption.restype = Bool
    _ipopt_lib.AddIpoptIntOption.argtypes = [IpoptProblem, c_char_p, c_int]

    _ipopt_lib.AddIpoptNumOption.restype = Bool
    _ipopt_lib.AddIpoptNumOption.argtypes = [IpoptProblem, c_char_p, c_double]

    # NOTE: these four were `.restypes` (typo, plural) in the older
    # yapss source -- ctypes silently ignores unknown attribute names
    # rather than erroring, so this was harmless in practice only because
    # ctypes' own default restype (unset) is also `c_int`, which happens to
    # match what was intended here.
    _ipopt_lib.OpenIpoptOutputFile.restype = Bool
    _ipopt_lib.OpenIpoptOutputFile.argtypes = [IpoptProblem, c_char_p, c_int]

    _ipopt_lib.SetIpoptProblemScaling.restype = Bool
    _ipopt_lib.SetIpoptProblemScaling.argtypes = [IpoptProblem, c_double, c_double_p, c_double_p]

    _ipopt_lib.SetIntermediateCallback.restype = Bool
    _ipopt_lib.SetIntermediateCallback.argtypes = [IpoptProblem, Intermediate_CB]

    # IpoptSolve returns `enum ApplicationReturnStatus`, which is int-sized.
    _ipopt_lib.IpoptSolve.restype = c_int
    _ipopt_lib.IpoptSolve.argtypes = [
        IpoptProblem,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_void_p,
    ]

    _ipopt_lib.GetIpoptCurrentIterate.restype = c_bool
    _ipopt_lib.GetIpoptCurrentIterate.argtypes = [
        IpoptProblem,
        c_bool,
        c_int,
        c_double_p,
        c_double_p,
        c_double_p,
        c_int,
        c_double_p,
        c_double_p,
    ]

    _ipopt_lib.GetIpoptCurrentViolations.restype = c_bool
    _ipopt_lib.GetIpoptCurrentViolations.argtypes = [
        IpoptProblem,
        c_bool,
        c_int,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        c_int,
        c_double_p,
        c_double_p,
    ]


def default_setup() -> None:
    """Confirm a library has been loaded, rather than guessing one.

    This used to load a platform-default library name when none had been set
    up. Guessing is exactly what causes a second IPOPT to be mapped alongside
    CasADi's, so an unconfigured module is now an error the caller must fix.
    """
    if _ipopt_lib is None:
        msg = (
            "No IPOPT library has been loaded. Call load_library() or "
            "use_library() first; this module will not choose one for you, "
            "because loading an IPOPT other than the one CasADi already has "
            "open causes an OpenMP runtime collision."
        )
        raise RuntimeError(msg)


def CreateIpoptProblem(
    n: int,
    x_L: Any,
    x_U: Any,
    m: int,
    g_L: Any,
    g_U: Any,
    nele_jac: int,
    nele_hess: int,
    index_style: int,
    eval_f: Any,
    eval_g: Any,
    eval_grad_f: Any,
    eval_jac_g: Any,
    eval_h: Any,
) -> Any:
    """Create and return a caller-owned native Ipopt problem.

    The four bound pointers address arrays of lengths ``n``, ``n``, ``m``, and
    ``m`` and are copied by Ipopt before this call returns. Callback pointers
    are not copied and must remain alive until :func:`FreeIpoptProblem`.
    ``index_style`` is zero for C indexing or one for Fortran indexing. A null
    result means construction failed and must not be freed.
    """
    default_setup()
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return _ipopt_lib.CreateIpoptProblem(
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        nele_jac,
        nele_hess,
        index_style,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )


def FreeIpoptProblem(ipopt_problem: Any) -> None:
    """Free one non-null problem returned by :func:`CreateIpoptProblem`."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    _ipopt_lib.FreeIpoptProblem(ipopt_problem)


def AddIpoptStrOption(problem: Any, keyword: str | bytes, val: str | bytes) -> int:
    """Set a string option and return Ipopt's success Boolean as an integer."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    if isinstance(val, str):
        val = val.encode("ascii")
    return int(_ipopt_lib.AddIpoptStrOption(problem, keyword, val))


def AddIpoptNumOption(problem: Any, keyword: str | bytes, val: float) -> int:
    """Set a numeric option and return Ipopt's success Boolean as an integer."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    return int(_ipopt_lib.AddIpoptNumOption(problem, keyword, val))


def AddIpoptIntOption(problem: Any, keyword: str | bytes, val: int) -> int:
    """Set an integer option and return Ipopt's success Boolean as an integer."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    return int(_ipopt_lib.AddIpoptIntOption(problem, keyword, val))


def OpenIpoptOutputFile(ipopt_problem: Any, file_name: str | bytes, print_level: int) -> int:
    """Open an Ipopt output file and return the native success Boolean."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(file_name, str):
        file_name = file_name.encode("ascii")
    return int(_ipopt_lib.OpenIpoptOutputFile(ipopt_problem, file_name, print_level))


def SetIpoptProblemScaling(
    ipopt_problem: Any,
    obj_scaling: float,
    x_scaling: Any,
    g_scaling: Any,
) -> int:
    """Set objective and optional length-``n``/``m`` scaling vectors."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return int(_ipopt_lib.SetIpoptProblemScaling(ipopt_problem, obj_scaling, x_scaling, g_scaling))


def SetIntermediateCallback(ipopt_problem: Any, intermediate_cb: Any) -> int:
    """Install a retained iteration callback, or disable it with a null pointer."""
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return int(_ipopt_lib.SetIntermediateCallback(ipopt_problem, intermediate_cb))


def IpoptSolve(
    ipopt_problem: Any,
    x: Any,
    g: Any,
    obj_val: Any,
    mult_g: Any,
    mult_x_L: Any,
    mult_x_U: Any,
    user_data: Any,
) -> int:
    """Solve one problem using caller-owned in/out buffers.

    ``x[n]`` is required. ``g[m]``, scalar ``obj_val``, ``mult_g[m]``, and the
    two bound-multiplier arrays of length ``n`` are independently nullable.
    Non-null multipliers are both warm-start inputs and final outputs. The
    return value is Ipopt's application status.
    """
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return int(
        _ipopt_lib.IpoptSolve(ipopt_problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U, user_data)
    )


def GetIpoptCurrentIterate(
    ipopt_problem: Any,
    scaled: bool,
    n: int,
    x: Any,
    z_L: Any,
    z_U: Any,
    m: int,
    g: Any,
    lambda_: Any,
) -> int:
    """Copy selected current primal/dual vectors during an intermediate callback."""
    assert _ipopt_lib is not None, "library must be loaded"
    return int(
        _ipopt_lib.GetIpoptCurrentIterate(ipopt_problem, scaled, n, x, z_L, z_U, m, g, lambda_)
    )


def GetIpoptCurrentViolations(
    ipopt_problem: Any,
    scaled: bool,
    n: int,
    x_L_violation: Any,
    x_U_violation: Any,
    compl_x_L: Any,
    compl_x_U: Any,
    grad_lag_x: Any,
    m: int,
    nlp_constraint_violation: Any,
    compl_g: Any,
) -> int:
    """Copy selected current violation vectors during an intermediate callback."""
    assert _ipopt_lib is not None, "library must be loaded"
    return int(
        _ipopt_lib.GetIpoptCurrentViolations(
            ipopt_problem,
            scaled,
            n,
            x_L_violation,
            x_U_violation,
            compl_x_L,
            compl_x_U,
            grad_lag_x,
            m,
            nlp_constraint_violation,
            compl_g,
        )
    )
