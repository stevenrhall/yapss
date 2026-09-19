"""
The root package's public names: every argument type supports isinstance, and every
warning category is importable from the root.

Through 0.2.2 ContinuousArg, DiscreteArg, and ObjectiveArg were re-exported as
subscripted generics, which isinstance refuses, and IpoptOptionSettingWarning lived only
in a private module.
"""

import warnings

import pytest

from yapss import _legacy as yapss
from yapss._private import solver

ARG_TYPES = [
    "ContinuousArg",
    "ContinuousJacobianArg",
    "ContinuousHessianArg",
    "DiscreteArg",
    "DiscreteJacobianArg",
    "DiscreteHessianArg",
    "ObjectiveArg",
    "ObjectiveGradientArg",
    "ObjectiveHessianArg",
]

WARNINGS = [
    "IpoptConvergenceWarning",
    "IpoptOptionSettingWarning",
]


@pytest.mark.parametrize("name", ARG_TYPES)
def test_argument_types_are_classes(name):
    cls = getattr(yapss, name)
    assert isinstance(cls, type), f"yapss.{name} is {cls!r}, not a class"
    assert name in yapss.__all__
    assert not isinstance(object(), cls)  # isinstance is usable, and false for a stranger


@pytest.mark.parametrize("method", ["auto", "central-difference"])
def test_callback_arguments_satisfy_isinstance(method):
    """The instances a callback receives, symbolic or real, are instances of the export."""
    seen = {}

    problem = yapss.Problem(name="isinstance", nx=[1], nu=[1], nd=1)

    def objective(arg):
        seen["objective"] = isinstance(arg, yapss.ObjectiveArg)
        arg.objective = arg.phase[0].final_state[0]

    def discrete(arg):
        seen["discrete"] = isinstance(arg, yapss.DiscreteArg)
        arg.discrete[:] = (arg.phase[0].final_time,)

    def continuous(arg):
        seen["continuous"] = isinstance(arg, yapss.ContinuousArg)
        for p in arg.phase_list:
            arg.phase[p].dynamics[:] = (arg.phase[p].control[0],)

    problem.functions.objective = objective
    problem.functions.discrete = discrete
    problem.functions.continuous = continuous
    problem.derivatives.method = method
    bounds = problem.bounds.phase[0]
    bounds.initial_time.lower = bounds.initial_time.upper = 0.0
    bounds.final_time.lower = bounds.final_time.upper = 1.0
    bounds.initial_state.lower = bounds.initial_state.upper = [0.0]
    bounds.control.lower, bounds.control.upper = [-1.0], [1.0]
    problem.bounds.discrete.lower, problem.bounds.discrete.upper = [0.0], [2.0]
    problem.guess.phase[0].time = [0.0, 1.0]
    problem.ipopt_options.print_level = 0
    problem.solve()

    assert seen == {"objective": True, "discrete": True, "continuous": True}


@pytest.mark.parametrize("name", WARNINGS)
def test_warning_categories_are_exported(name):
    category = getattr(yapss, name)
    assert issubclass(category, Warning)
    assert name in yapss.__all__


def test_option_warning_is_the_solver_class():
    assert yapss.IpoptOptionSettingWarning is solver.IpoptOptionSettingWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", yapss.IpoptOptionSettingWarning)
        problem = yapss.Problem(name="filter", nx=[], ns=1)
        problem.functions.objective = lambda arg: setattr(arg, "objective", arg.parameter[0] ** 2)
        problem.ipopt_options.print_level = 0
        problem.ipopt_options.__dict__["not_a_real_option"] = 1  # the setter refuses it now
        with pytest.raises(yapss.IpoptOptionSettingWarning):
            problem.solve()
