# Copyright (c) 2018 Centro de Estudos Aeronáuticos da UFMG
# Copyright (c) 2021-2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
#
# Derived from mseipopt (https://github.com/cea-ufmg/mseipopt), modified by
# the YAPSS authors. Original and modified portions are both under the MIT
# license; see the LICENSE file at the repository root.

"""Bare wrapper around the IPOPT c interface using ctypes.

The type names are the same as in `IpStdCInterface.h`, the argument names are
the same (except for lambda, which is a python keyword) and even the order of
the definitions in the module is the same. Very little checking or validation
is done. This module provides direct access to the ipopt c interface functions
using `ctypes`.

`load_library` (given an explicit path) or `use_library` (given an already
loaded library) must be called before creating a problem. This module does not
locate IPOPT itself; see `yapss._private.mseipopt.library` for why the choice of
file matters and must not be guessed.

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


Bool: Any = c_bool
"""The C type of IPOPT's ``Bool``, which is version dependent.

IPOPT 3.14 changed ``typedef int Bool`` to ``typedef bool Bool``, narrowing it
from 4 bytes to 1. The default here matches 3.14 and later, which is what every
CasADi wheel currently bundles; `set_bool_type` overrides it when the shipped
``IpStdCInterface.h`` says otherwise.
"""


# Declared here so the names exist at import time; `set_bool_type` rebuilds
# them, and every consumer looks them up as attributes of this module rather
# than importing them, so rebinding takes effect everywhere.
Eval_F_CB: Any = None
"""Type of the callback for evaluating the objective function."""

Eval_Grad_F_CB: Any = None
"""Type of the callback for evaluating the gradient of the objective."""

Eval_G_CB: Any = None
"""Type of the callback for evaluating the constraint function."""

Eval_Jac_G_CB: Any = None
"""Type of the callback for evaluating the Jacobian of the constraint."""

Eval_H_CB: Any = None
"""Type of the callback for evaluating the Hessian of the Lagrangian."""

Intermediate_CB: Any = None
"""Type of the callback to give intermediate execution control to the user."""


def set_bool_type(bool_type: Any = c_bool) -> None:
    """Rebuild the callback types for an IPOPT whose ``Bool`` is *bool_type*.

    Must be called before any callback instance is created; a callback built
    from the old type would then be passed to a library expecting the new one.
    In practice this is called once, during library initialization, before any
    problem exists.

    The argument positions that take `Bool` are the ``new_x`` and
    ``new_lambda`` flags, and every callback returns it.
    """
    global Bool, Eval_F_CB, Eval_Grad_F_CB, Eval_G_CB  # noqa: PLW0603
    global Eval_Jac_G_CB, Eval_H_CB, Intermediate_CB  # noqa: PLW0603

    Bool = bool_type
    #                     ret        n      x           new_x      obj_value   user_data
    Eval_F_CB = CFUNCTYPE(bool_type, c_int, c_double_p, bool_type, c_double_p, c_void_p)
    Eval_Grad_F_CB = CFUNCTYPE(bool_type, c_int, c_double_p, bool_type, c_double_p, c_void_p)
    Eval_G_CB = CFUNCTYPE(bool_type, c_int, c_double_p, bool_type, c_int, c_double_p, c_void_p)
    Eval_Jac_G_CB = CFUNCTYPE(
        bool_type,  # ret
        c_int,  # n
        c_double_p,  # x
        bool_type,  # new_x
        c_int,  # m
        c_int,  # nele_jac
        c_int_p,  # iRow
        c_int_p,  # jCol
        c_double_p,  # values
        c_void_p,  # user_data
    )
    Eval_H_CB = CFUNCTYPE(
        bool_type,  # ret
        c_int,  # n
        c_double_p,  # x
        bool_type,  # new_x
        c_double,  # obj_factor
        c_int,  # m
        c_double_p,  # lambda
        bool_type,  # new_lambda
        c_int,  # nele_hess
        c_int_p,  # iRow
        c_int_p,  # jCol
        c_double_p,  # values
        c_void_p,  # user_data
    )
    Intermediate_CB = CFUNCTYPE(
        bool_type,  # ret
        c_int,  # alg_mod
        c_int,  # iter_count
        c_double,  # obj_value
        c_double,  # inf_pr
        c_double,  # inf_du
        c_double,  # mu
        c_double,  # d_norm
        c_double,  # regularization_size
        c_double,  # alpha_du
        c_double,  # alpha_pr
        c_int,  # ls_trials
        c_void_p,  # user_data
    )


set_bool_type()


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
    global _ipopt_lib
    _ipopt_lib = ctypes.cdll.LoadLibrary(name)
    _setup_library()


def use_library(lib: ctypes.CDLL) -> None:
    """Adopt an already-loaded IPOPT library and configure its signatures.

    Lets the caller do the locating and loading -- and any verification it
    wants to perform on the result -- while this module remains responsible
    only for the ctypes declarations.
    """
    global _ipopt_lib
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
