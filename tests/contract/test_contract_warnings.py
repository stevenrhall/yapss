"""Contract: the warning and error hierarchy.

What a user may do
    - Filter or escalate every YAPSS warning in one line, with `yapss.YapssWarning`, or one
      category at a time.
    - Catch a YAPSS error either as `yapss.YapssError` or as the built-in exception it also
      inherits.
    - Rely on every unconverged solve warning, even several from one line.

What a user may get wrong
    - Filtering by ``module="yapss"``: every YAPSS warning points at the user's own code
      through `stacklevel`, so such a filter matches none of them.

Drift
    - Every warning and error class YAPSS defines outside the vendored `mseipopt` package is
      part of the hierarchy, and every public one is exported from `yapss`.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings

import pytest

import yapss._backend
from yapss._backend.exceptions import YapssDeprecationWarning, YapssError, YapssWarning

from ._contract import callback_problem


def unconverged():
    """The contract problem, stopped after one iteration."""
    ocp = callback_problem()
    ocp.ipopt_options.max_iter = 1
    return ocp


CATEGORIES = [
    yapss.IpoptConvergenceWarning,
    yapss.LargeSegmentWarning,
    yapss.IpoptOptionSettingWarning,
]


def yapss_classes(kind: type) -> list[type]:
    """Every subclass of `kind` defined in yapss, except in the vendored mseipopt package."""
    modules = [yapss, yapss._backend]
    for package in (yapss, yapss.math, yapss._backend):
        for info in pkgutil.walk_packages(package.__path__, f"{package.__name__}."):
            if "mseipopt" in info.name:
                continue
            modules.append(importlib.import_module(info.name))
    found = {
        value
        for module in modules
        for _, value in inspect.getmembers(module, inspect.isclass)
        if issubclass(value, kind) and value.__module__.startswith("yapss.")
    }
    return sorted(found, key=lambda cls: cls.__name__)


def test_every_warning_class_is_in_the_hierarchy():
    """A new warning category cannot drift out of the one-line filter."""
    classes = yapss_classes(Warning)
    assert set(CATEGORIES) <= set(classes)
    for cls in classes:
        assert issubclass(cls, YapssWarning), cls
    assert issubclass(YapssWarning, UserWarning)


def test_every_warning_category_is_exported():
    """A category users cannot name is a category they cannot filter.

    `CATEGORIES` above is hand-written, so on its own it would not notice a new class; this
    walks the package instead, and holds every category except the two bases to being
    importable from `yapss`.
    """
    bases = {YapssWarning, YapssDeprecationWarning}
    for cls in yapss_classes(Warning):
        if cls in bases:
            continue
        assert cls.__name__ in yapss.__all__, f"{cls.__name__} is not exported from yapss"
        assert getattr(yapss, cls.__name__) is cls
        assert cls in CATEGORIES, f"{cls.__name__} is missing from CATEGORIES in this file"


def test_every_error_class_is_in_the_hierarchy():
    classes = [cls for cls in yapss_classes(Exception) if not issubclass(cls, Warning)]
    assert yapss.UnsupportedMathFunctionError in classes
    for cls in classes:
        assert issubclass(cls, YapssError), cls


@pytest.mark.parametrize("category", CATEGORIES, ids=lambda cls: cls.__name__)
def test_every_category_is_public_and_documented_by_its_name(category):
    assert category.__name__ in yapss.__all__
    assert getattr(yapss, category.__name__) is category
    assert category.__doc__


def test_a_deprecation_is_also_a_future_warning():
    """FutureWarning, not DeprecationWarning: scripts and notebooks hide the latter."""
    assert issubclass(YapssDeprecationWarning, YapssWarning)
    assert issubclass(YapssDeprecationWarning, FutureWarning)


def test_one_line_escalates_every_yapss_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", YapssWarning)
        with pytest.raises(yapss.IpoptConvergenceWarning):
            unconverged().solve()


def test_an_error_is_catchable_as_its_builtin_too():
    assert issubclass(yapss.UnsupportedMathFunctionError, YapssError)
    assert issubclass(yapss.UnsupportedMathFunctionError, TypeError)


def test_a_yapss_warning_points_at_user_code_not_at_yapss():
    """Which is why `module="yapss"` filters nothing and the base class is needed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = inspect.currentframe().f_lineno + 1
        unconverged().solve()
    (warning,) = [w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)]
    assert (warning.filename, warning.lineno) == (__file__, line)


def test_repeated_unconverged_solves_from_one_line_each_warn():
    """Python's "default" action would report only the first; every solve must say so."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default", yapss.IpoptConvergenceWarning)
        for _ in range(3):
            unconverged().solve()
    convergence = [w for w in caught if issubclass(w.category, yapss.IpoptConvergenceWarning)]
    assert len(convergence) == 3


def test_a_converged_solve_still_says_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", YapssWarning)
        assert callback_problem().solve().nlp_info.ipopt_status == 0
