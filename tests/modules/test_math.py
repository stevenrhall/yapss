import casadi as ca
import numpy as np
import pytest

from yapss import math
from yapss.math.wrapper import SXW

x_sym = ca.SX.sym("x2", 100)
x2 = np.array([SXW(x_sym[i]) for i in range(100)], dtype=object)

trig_functions = (
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "deg2rad",
    "rad2deg",
    "radians",
    "degrees",
)

hyperbolic_trig_functions = ("sinh", "cosh", "tanh", "arcsinh", "arctanh")
miscellaneous_functions = ("abs", "sign", "floor", "ceil")
log_functions = ("log", "log10", "log2")


@pytest.mark.parametrize("function_name", trig_functions)
def test_trig_functions(function_name):
    check_function(function_name, -1.0, 1.0)


@pytest.mark.parametrize("function_name", hyperbolic_trig_functions)
def test_hyperbolic_trig_functions(function_name):
    check_function(function_name, -0.9, 0.9)


@pytest.mark.parametrize("function_name", miscellaneous_functions)
def test_miscellaneous_functions(function_name):
    check_function(function_name, -10.0, 10.0)


@pytest.mark.parametrize("function_name", log_functions)
def test_log_functions(function_name):
    check_function(function_name, 0.1, 100.0)


# hyperbolic trig functions
# "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
#  # exponential and logarithmic functions
# "exp", "log", "log10", "log1p", "expm1",)
# )


def check_function(function_name, x_lower, x_upper):
    x1 = np.linspace(x_lower, x_upper, 100)
    f1 = getattr(np, function_name)
    f2 = getattr(math, function_name)
    y1 = f1(x1)
    y2 = f2(x2)
    y_sym = ca.vertcat(*[SXW(item)._value for item in y2])
    f = ca.Function("f", [x_sym], [y_sym])
    error = np.max(np.abs(f(x1).toarray().flatten() - y1))
    print(error)
    assert error <= 1e-14
