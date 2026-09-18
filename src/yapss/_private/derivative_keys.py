"""

Parse and validate the keys of user-supplied derivative dictionaries.

Under the ``"user"`` derivative method, the derivative structures are the key sets of the
dictionaries that the user's callbacks fill in. They are the only structures YAPSS does not
build itself, so this module is where they enter as untrusted objects and leave as typed
structure terms. Everything downstream -- the Jacobian and Hessian assembly, the
mirrored-pair check -- can rely on what the types say.

The rules (API review decision E7b):

- A key has its documented shape. A decision variable key is ``(phase, name, index)``; a
  continuous variable key is ``(name, index)``; a continuous function key is
  ``(name, index)``. Jacobian and Hessian keys lead with the function (a key for the
  continuous functions, an integer for the discrete functions; the objective has none),
  followed by one variable key for a Jacobian or two for a Hessian.
- Every name is a member of its `Literal` type (`DVName`, `CVName`, `CFName`). The
  accepted names are read from those types, so they cannot drift from what the assembly
  dispatches on.
- Every index is a Python or NumPy integer, but not a `bool`, and is stored as `int`.
- Every index is in range, ``0 <= index < count``. A negative index would name the same
  variable as a non-negative one under a different key, and the two entries would be
  summed without warning.
- A parameter is keyed with phase 0, ``(0, "s", index)``, for the same reason.

Every violation raises with a message that names the key, the callback, and (for the
continuous functions) the phase, and says what was expected. Following Python's convention,
a part of the key of the wrong type (a key that is not a tuple, a name that is not a
string, an index that is not an integer) raises `TypeError`; a part of the right type with
a wrong value (a tuple of the wrong length, an unknown name, an index out of range) raises
`ValueError`.
"""

# future imports
from __future__ import annotations

# standard imports
import difflib
from typing import TYPE_CHECKING, NamedTuple, TypeVar, assert_never, get_args

# third party imports
import numpy as np

# package imports
from .types_ import CFName, CVName, DVName

if TYPE_CHECKING:
    # standard imports
    from collections.abc import Iterable

    from .spec import ProblemSpec
    from .types_ import DHS, DJS, OGS, OHS, CFKey, CHSPhase, CJSPhase, CVKey, DVKey

    # package imports

__all__ = [
    "continuous_hessian_structure",
    "continuous_jacobian_structure",
    "discrete_hessian_structure",
    "discrete_jacobian_structure",
    "objective_gradient_structure",
    "objective_hessian_structure",
]

DV_NAMES: tuple[DVName, ...] = get_args(DVName)
CV_NAMES: tuple[CVName, ...] = get_args(CVName)
CF_NAMES: tuple[CFName, ...] = get_args(CFName)

N = TypeVar("N", bound=str)

DV_FORM = "(phase, name, index), such as (0, 'tf', 0)"
CV_FORM = "(name, index), such as ('x', 0)"
CF_FORM = "(name, index), such as ('f', 0)"


class Items(NamedTuple):
    """How to describe the items an index selects, for error messages.

    The message reads "the <label> index 5 is out of range: <owner> has 3 <many>", with
    <one> for a count of one.
    """

    label: str
    owner: str
    one: str
    many: str

    @classmethod
    def counted(cls, one: str, many: str, owner: str) -> Items:
        """Items counted by their own noun: "phase 0 has 3 states"."""
        return cls(one, owner, one, many)

    @classmethod
    def elements(cls, output: str, p: int) -> Items:
        """Elements of a continuous function output: "phase 0 dynamics has 3 elements"."""
        return cls(output, f"phase {p} {output}", "element", "elements")


class _Key:
    """One user key being parsed, and where it came from, for error messages."""

    def __init__(self, key: object, where: str) -> None:
        self.key = key
        self.where = where

    def error(
        self,
        reason: str,
        kind: type[TypeError | ValueError] = ValueError,
    ) -> TypeError | ValueError:
        """Return the error for this key, for the caller to raise."""
        return kind(f"Invalid derivative key {self.key!r} set by {self.where}: {reason}.")

    def split(
        self, value: object, n: int, form: str, what: str | None = None
    ) -> tuple[object, ...]:
        """Return the `n` elements of `value` (the key itself when `what` is None).

        `form` is the expected form for the message; `what` names a part of the key.
        """
        if what is None:
            reason = f"expected a key of the form {form}"
        else:
            reason = f"{value!r} is not a {what}; expected {form}"
        if not isinstance(value, tuple):
            raise self.error(reason, TypeError)
        if len(value) != n:
            raise self.error(reason)
        return value

    def name(self, value: object, names: tuple[N, ...], kind: str) -> N:
        """Return `value` as a member of `names`, or fail suggesting the closest one."""
        expected = ", ".join(repr(name) for name in names)
        if not isinstance(value, str):
            reason = f"the {kind} name {value!r} is not a string; expected one of {expected}"
            raise self.error(reason, TypeError)
        for name in names:
            if value == name:
                return name
        reason = f"{value!r} is not a {kind} name; expected one of {expected}"
        close: list[str] = [name for name in names if name.lower() == value.lower()]
        close = close or difflib.get_close_matches(value, names, n=1)
        if close:
            reason += f" (did you mean {close[0]!r}?)"
        raise self.error(reason)

    def integer(self, value: object, what: str) -> int:
        """Return `value` as an `int`, accepting Python and NumPy integers but not `bool`."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            reason = f"the {what} index {value!r} is not an integer"
            raise self.error(reason, TypeError)
        return int(value)

    def index(self, value: object, count: int, what: Items) -> int:
        """Return `value` as an integer index into the `count` items `what` describes."""
        i = self.integer(value, what.label)
        if not 0 <= i < count:
            if count == 0:
                bound = f"{what.owner} has no {what.many}"
            elif count == 1:
                bound = f"{what.owner} has 1 {what.one}, so the index must be 0"
            else:
                limit = f"in range({count})"
                bound = f"{what.owner} has {count} {what.many}, so the index must be {limit}"
            reason = f"the {what.label} index {i} is out of range: {bound}"
            raise self.error(reason)
        return i


DISCRETE = Items.counted("discrete function", "discrete functions", "the problem")


def _dv_key(key: _Key, value: object, problem: ProblemSpec) -> DVKey:
    p_value, name_value, i_value = key.split(value, 3, DV_FORM, "decision variable key")
    name = key.name(name_value, DV_NAMES, "decision variable")
    if name == "s":
        p = key.integer(p_value, "phase")
        i = key.index(i_value, problem.ns, Items.counted("parameter", "parameters", "the problem"))
        if p != 0:
            reason = f"a parameter is keyed with phase 0, as {(0, 's', i)!r}"
            raise key.error(reason)
        return 0, name, i
    p = key.index(p_value, problem.np, Items.counted("phase", "phases", "the problem"))
    phase = f"phase {p}"
    match name:
        case "x0" | "xf":
            count, what = problem.nx[p], Items.counted("state", "states", phase)
        case "t0" | "tf":
            count, what = 1, Items.counted("time", "times", phase)
        case "q":
            count, what = problem.nq[p], Items.counted("integral", "integrals", phase)
        case _:
            assert_never(name)
    return p, name, key.index(i_value, count, what)


def _cv_key(key: _Key, value: object, problem: ProblemSpec, p: int) -> CVKey:
    name_value, i_value = key.split(value, 2, CV_FORM, "continuous variable key")
    name = key.name(name_value, CV_NAMES, "continuous variable")
    phase = f"phase {p}"
    match name:
        case "x":
            count, what = problem.nx[p], Items.counted("state", "states", phase)
        case "u":
            count, what = problem.nu[p], Items.counted("control", "controls", phase)
        case "s":
            count, what = problem.ns, Items.counted("parameter", "parameters", "the problem")
        case "t":
            count, what = 1, Items.counted("time", "times", phase)
        case _:
            assert_never(name)
    return name, key.index(i_value, count, what)


def _cf_key(key: _Key, value: object, problem: ProblemSpec, p: int) -> CFKey:
    name_value, i_value = key.split(value, 2, CF_FORM, "continuous function key")
    name = key.name(name_value, CF_NAMES, "continuous function")
    match name:
        case "f":
            count, what = problem.nx[p], Items.elements("dynamics", p)
        case "g":
            count, what = problem.nq[p], Items.elements("integrand", p)
        case "h":
            count, what = problem.nh[p], Items.elements("path", p)
        case _:
            assert_never(name)
    return name, key.index(i_value, count, what)


def objective_gradient_structure(problem: ProblemSpec, keys: Iterable[object]) -> OGS:
    """Parse the keys of ``arg.gradient`` set by ``functions.objective_gradient``."""
    structure: list[DVKey] = []
    for k in keys:
        key = _Key(k, "functions.objective_gradient")
        structure.append(_dv_key(key, k, problem))
    return tuple(structure)


def objective_hessian_structure(problem: ProblemSpec, keys: Iterable[object]) -> OHS:
    """Parse the keys of ``arg.hessian`` set by ``functions.objective_hessian``."""
    structure: list[tuple[DVKey, DVKey]] = []
    for k in keys:
        key = _Key(k, "functions.objective_hessian")
        v1, v2 = key.split(k, 2, "(variable key, variable key)")
        structure.append((_dv_key(key, v1, problem), _dv_key(key, v2, problem)))
    return tuple(structure)


def discrete_jacobian_structure(problem: ProblemSpec, keys: Iterable[object]) -> DJS:
    """Parse the keys of ``arg.jacobian`` set by ``functions.discrete_jacobian``."""
    structure: list[tuple[int, DVKey]] = []
    for k in keys:
        key = _Key(k, "functions.discrete_jacobian")
        d, v = key.split(k, 2, "(discrete function index, variable key)")
        d_index = key.index(d, problem.nd, DISCRETE)
        structure.append((d_index, _dv_key(key, v, problem)))
    return tuple(structure)


def discrete_hessian_structure(problem: ProblemSpec, keys: Iterable[object]) -> DHS:
    """Parse the keys of ``arg.hessian`` set by ``functions.discrete_hessian``."""
    structure: list[tuple[int, DVKey, DVKey]] = []
    for k in keys:
        key = _Key(k, "functions.discrete_hessian")
        d, v1, v2 = key.split(k, 3, "(discrete function index, variable key, variable key)")
        d_index = key.index(d, problem.nd, DISCRETE)
        structure.append((d_index, _dv_key(key, v1, problem), _dv_key(key, v2, problem)))
    return tuple(structure)


def continuous_jacobian_structure(
    problem: ProblemSpec,
    p: int,
    keys: Iterable[object],
) -> CJSPhase:
    """Parse the keys of ``arg.phase[p].jacobian`` set by ``functions.continuous_jacobian``."""
    structure: list[tuple[CFKey, CVKey]] = []
    for k in keys:
        key = _Key(k, f"functions.continuous_jacobian in phase {p}")
        f, v = key.split(k, 2, "(function key, variable key)")
        structure.append((_cf_key(key, f, problem, p), _cv_key(key, v, problem, p)))
    return tuple(structure)


def continuous_hessian_structure(
    problem: ProblemSpec,
    p: int,
    keys: Iterable[object],
) -> CHSPhase:
    """Parse the keys of ``arg.phase[p].hessian`` set by ``functions.continuous_hessian``."""
    structure: list[tuple[CFKey, CVKey, CVKey]] = []
    for k in keys:
        key = _Key(k, f"functions.continuous_hessian in phase {p}")
        f, v1, v2 = key.split(k, 3, "(function key, variable key, variable key)")
        structure.append(
            (
                _cf_key(key, f, problem, p),
                _cv_key(key, v1, problem, p),
                _cv_key(key, v2, problem, p),
            ),
        )
    return tuple(structure)
