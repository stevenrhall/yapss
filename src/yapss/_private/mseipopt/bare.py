"""Bare wrapper around the IPOPT c interface using ctypes.

The type names are the same as in `IpStdCInterface.h`, the argument names are
the same (except for lambda, which is a python keyword) and even the order of
the definitions in the module is the same. Very little checking or validation
is done. This module provides direct access to the ipopt c interface functions
using `ctypes`.

If the library shared object or DLL has a nonstandard name, the `load_library`
function must be called before creating a problem. When `CreateIpoptProblem`
is called, an attempt is made to find load the library.

After a problem is no longer needed, `FreeIpoptProblem` should be called, or
memory will leak. If `FreeIpoptProblem` is called more than once on the same
problem, the program will likely crash.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_double, c_int, c_void_p
from typing import Any

_ipopt_lib: ctypes.CDLL | None = None
"""library ctypes.CDLL object or None (if not loaded)."""


c_double_p = POINTER(c_double)
"""Pointer to double."""


c_int_p = POINTER(c_int)
"""Pointer to int."""


Eval_F_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, c_double_p, c_void_p)
"""Type of the callback for evaluating the objective function."""


Eval_Grad_F_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, c_double_p, c_void_p)
"""Type of the callback for evaluating the gradient of the objective."""


Eval_G_CB = CFUNCTYPE(c_int, c_int, c_double_p, c_int, c_int, c_double_p, c_void_p)
"""Type of the callback for evaluating the constraint function."""


Eval_Jac_G_CB = CFUNCTYPE(
    c_int,
    c_int,
    c_double_p,
    c_int,
    c_int,
    c_int,
    c_int_p,
    c_int_p,
    c_double_p,
    c_void_p,
)
"""Type of the callback for evaluating the Jacobian of the constraint."""


Eval_H_CB = CFUNCTYPE(
    c_int,
    c_int,
    c_double_p,
    c_int,
    c_double,
    c_int,
    c_double_p,
    c_int,
    c_int,
    c_int_p,
    c_int_p,
    c_double_p,
    c_void_p,
)
"""Type of the callback for evaluating the Hessian of the Lagrangian."""


Intermediate_CB = CFUNCTYPE(
    c_int,
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


def default_ipopt_library_name() -> str:
    """Return the platform-default IPOPT shared library name."""
    if os.name == "nt":
        return "ipopt"
    else:
        return "libipopt.so"


def load_library(name: str | None = None) -> None:
    """Load the IPOPT shared library and configure its ctypes signatures."""
    global _ipopt_lib
    name = name or default_ipopt_library_name()
    _ipopt_lib = ctypes.cdll.LoadLibrary(name)
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

    _ipopt_lib.AddIpoptStrOption.restype = c_int
    _ipopt_lib.AddIpoptStrOption.argtypes = [IpoptProblem, c_char_p, c_char_p]

    _ipopt_lib.AddIpoptIntOption.restype = c_int
    _ipopt_lib.AddIpoptIntOption.argtypes = [IpoptProblem, c_char_p, c_int]

    _ipopt_lib.AddIpoptNumOption.restype = c_int
    _ipopt_lib.AddIpoptNumOption.argtypes = [IpoptProblem, c_char_p, c_double]

    # NOTE: these four were `.restypes` (typo, plural) in the older
    # yapss source -- ctypes silently ignores unknown attribute names
    # rather than erroring, so this was harmless in practice only because
    # ctypes' own default restype (unset) is also `c_int`, which happens to
    # match what was intended here.
    _ipopt_lib.OpenIpoptOutputFile.restype = c_int
    _ipopt_lib.OpenIpoptOutputFile.argtypes = [IpoptProblem, c_char_p, c_int]

    _ipopt_lib.SetIpoptProblemScaling.restype = c_int
    _ipopt_lib.SetIpoptProblemScaling.argtypes = [IpoptProblem, c_double, c_double_p, c_double_p]

    _ipopt_lib.SetIntermediateCallback.restype = c_int
    _ipopt_lib.SetIntermediateCallback.argtypes = [IpoptProblem, Intermediate_CB]

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


def default_setup() -> None:
    """Load the default IPOPT library if none has been loaded yet."""
    if _ipopt_lib is None:
        load_library()


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
    """Create a new IPOPT Problem object."""
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
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    _ipopt_lib.FreeIpoptProblem(ipopt_problem)


def AddIpoptStrOption(problem: Any, keyword: str | bytes, val: str | bytes) -> int:
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    if isinstance(val, str):
        val = val.encode("ascii")
    return int(_ipopt_lib.AddIpoptStrOption(problem, keyword, val))


def AddIpoptNumOption(problem: Any, keyword: str | bytes, val: float) -> int:
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    return int(_ipopt_lib.AddIpoptNumOption(problem, keyword, val))


def AddIpoptIntOption(problem: Any, keyword: str | bytes, val: int) -> int:
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    if isinstance(keyword, str):
        keyword = keyword.encode("ascii")
    return int(_ipopt_lib.AddIpoptIntOption(problem, keyword, val))


def OpenIpoptOutputFile(ipopt_problem: Any, file_name: str | bytes, print_level: int) -> int:
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
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return int(_ipopt_lib.SetIpoptProblemScaling(ipopt_problem, obj_scaling, x_scaling, g_scaling))


def SetIntermediateCallback(ipopt_problem: Any, intermediate_cb: Any) -> int:
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
    assert _ipopt_lib is not None, "library must be loaded to create problem"
    return int(
        _ipopt_lib.IpoptSolve(ipopt_problem, x, g, obj_val, mult_g, mult_x_L, mult_x_U, user_data)
    )
