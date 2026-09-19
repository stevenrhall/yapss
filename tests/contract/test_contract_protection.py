"""Contract: every public container refuses assignments it does not support, and says why.

Walks every `Protected` instance a user can reach -- from the problem definition and from
the argument passed to each of the nine callbacks -- and checks the same four clauses on
each: a misspelled name is refused with a suggestion, a public attribute without a setter
is read-only, a private backing field cannot be assigned, and deletion is refused. A
`Protected` class that the walk does not reach fails the first test, so a new container
cannot go unchecked.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from yapss._legacy.examples import goddard_problem_3_phase
from yapss._private.types_ import Protected

CALLBACKS = (
    "objective",
    "objective_gradient",
    "objective_hessian",
    "continuous",
    "continuous_jacobian",
    "continuous_hessian",
    "discrete",
    "discrete_jacobian",
    "discrete_hessian",
)


def _subclasses(cls: type) -> set[type]:
    out = set()
    for sub in cls.__subclasses__():
        out |= {sub} | _subclasses(sub)
    return out


FRONT_AND_BACK = ("yapss._private", "yapss._legacy")
"""Where `Protected` containers live: the shared back end, and the 0.3.0 front end.

The redesigned front end is not here. Its containers are built on `_api.containers.Container`
rather than on `Protected`, and `tests/api` covers them.
"""

PROTECTED = sorted(
    (c for c in _subclasses(Protected) if c.__module__.startswith(FRONT_AND_BACK)),
    key=lambda c: c.__name__,
)


def _public_names(obj: Any) -> list[str]:
    names = {n for n in dir(type(obj)) if not n.startswith("_")}
    names |= {n for n in vars(obj) if not n.startswith("_")}
    return sorted(names)


@pytest.fixture(scope="module")
def instances() -> dict[type, tuple[str, Any]]:
    """One reachable instance of every protected class, with the path that reaches it."""
    ocp = goddard_problem_3_phase.setup()
    ocp.derivatives.method = "user"
    ocp.ipopt_options.print_level = 0
    ocp.ipopt_options.max_iter = 1
    captured: dict[str, Any] = {}

    def capture(name: str) -> Any:
        original = getattr(ocp.functions, name)

        def wrapped(arg: Any) -> None:
            captured.setdefault(name, arg)
            original(arg)

        return wrapped

    for name in CALLBACKS:
        setattr(ocp.functions, name, capture(name))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ocp.solve()
    assert set(captured) == set(CALLBACKS)

    found: dict[type, tuple[str, Any]] = {}
    seen: set[int] = set()

    def walk(obj: Any, path: str) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, tuple):
            if obj:
                walk(obj[0], f"{path}[0]")
            return
        # Containers only: not values (arrays and symbolic wrappers make a new object on
        # each attribute read, `.T` for one) and not `auxdata`, the user's own namespace.
        private = type(obj).__module__.startswith(FRONT_AND_BACK)
        if not private or isinstance(obj, np.ndarray) or type(obj).__name__ == "Auxdata":
            return
        if isinstance(obj, Protected):
            found.setdefault(type(obj), (path, obj))
        for name in _public_names(obj):
            try:
                value = getattr(obj, name)
            except Exception:  # noqa: BLE001 -- e.g. a guess array read before its time
                continue
            if not callable(value) or isinstance(value, Protected):
                walk(value, f"{path}.{name}")

    walk(ocp, "problem")
    for name, arg in captured.items():
        walk(arg, f"{name}(arg)")
    return found


def test_every_protected_class_is_reached(instances):
    assert {c.__name__ for c in instances} == {c.__name__ for c in PROTECTED}


@pytest.mark.parametrize("cls", PROTECTED, ids=lambda c: c.__name__)
def test_a_misspelled_name_is_refused_with_a_suggestion(instances, cls):
    path, obj = instances[cls]
    if not cls._settable:
        with pytest.raises(AttributeError, match="no such attribute"):
            obj.not_an_attribute = 0
        return
    for name in sorted(cls._settable):
        typo = name[:-1] if len(name) > 3 else name + "x"
        with pytest.raises(AttributeError) as info:
            setattr(obj, typo, 0)
        message = str(info.value)
        assert "no such attribute" in message, (path, typo, message)
        assert f"did you mean '{name}'" in message, (path, typo, message)


@pytest.mark.parametrize("cls", PROTECTED, ids=lambda c: c.__name__)
def test_a_public_attribute_without_a_setter_is_read_only(instances, cls):
    path, obj = instances[cls]
    for name in _public_names(obj):
        if name in cls._settable:
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001
            continue
        if callable(value) and not isinstance(value, Protected):
            continue
        with pytest.raises(AttributeError) as info:
            setattr(obj, name, value)
        assert "read-only" in str(info.value), (path, name, str(info.value))


@pytest.mark.parametrize("cls", PROTECTED, ids=lambda c: c.__name__)
def test_a_private_backing_field_cannot_be_assigned(instances, cls):
    path, obj = instances[cls]
    for name in sorted(vars(obj)):
        if not name.startswith("_") or name == "_sealed":
            continue
        with pytest.raises(AttributeError) as info:
            setattr(obj, name, vars(obj)[name])
        assert "no such attribute" in str(info.value), (path, name, str(info.value))


@pytest.mark.parametrize("cls", PROTECTED, ids=lambda c: c.__name__)
def test_deletion_is_refused(instances, cls):
    path, obj = instances[cls]
    for name in _public_names(obj):
        with pytest.raises(AttributeError):
            delattr(obj, name)
