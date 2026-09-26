"""What happens to a name that was never declared.

Every container a user touches answers for the names it has and refuses the rest, at the line
where the name is written. This is one rule with one message shape, stated here once rather
than on each page, and the value of it is that a typo is a refusal rather than a setting that
quietly does nothing.
"""

from __future__ import annotations

import pytest

from yapss.math import cos, sin

from ._api import G0, problem, raises, solvable


def test_a_setting_that_does_not_exist_is_refused() -> None:
    """Assigning an unknown name on the problem is a typo, not a new setting."""
    p = problem()
    with raises(AttributeError, "problem has no setting 'nope'", at="p.nope"):
        p.nope = 1


def test_a_phase_setting_that_does_not_exist_is_refused() -> None:
    """The same on a phase, and the message names which phase."""
    p = problem()
    with raises(AttributeError, "phase 'first' has no setting 'nope'", at="ph.nope"):
        ph = p.phases.first
        ph.nope = 1


def test_a_phase_that_was_not_declared_is_refused() -> None:
    """Phases answer for the names the declaration gave them."""
    p = problem()
    with raises(AttributeError, "Phases has no phase 'nope'", at="p.phases.nope"):
        _ = p.phases.nope


def test_a_removed_setting_is_refused_like_any_other() -> None:
    """`ipopt_source` was removed in 0.3.0 and is not a name this API has at all."""
    p = problem()
    with raises(AttributeError, "no setting 'ipopt_source'", at="p.ipopt_source"):
        p.ipopt_source = "x"


def test_a_misspelling_is_refused_where_it_is_written() -> None:
    """Not at the solve: at the assignment, which is the line a user has to change."""
    p = solvable()
    with raises(AttributeError, "no setting", at="p.derivatives.methodd"):
        p.derivatives.methodd = "auto"


# ------------------------------------------------------------------------- deletion


@pytest.mark.parametrize(
    "target",
    [
        "p.derivatives",
        "p.derivatives.method",
        "p.objective.sense",
        "p.spectral_method",
        "ph.time.final",
        "ph.mesh",
        "ph.state.x",
        "ph.state.x.bounds",
        "p.phases.slide",
    ],
)
def test_a_setting_cannot_be_deleted(target: str) -> None:
    """Deleted, a setting would leave the problem contradicting itself at every later read.

    "derivatives has no setting 'method'. Did you mean 'method'?" would be reported far from
    the line that caused it, so the deletion is refused at that line.
    """
    p = solvable()
    ph = p.phases.slide
    with raises(AttributeError, "cannot be deleted", at="exec"):
        exec(f"del {target}", {"p": p, "ph": ph})  # noqa: S102


@pytest.mark.parametrize("target", ["sol.objective", "sol.status", "ps.state", "sol.nlp.x"])
def test_a_solution_cannot_be_deleted_from(target: str) -> None:
    """A solution's names are fixed; deleting one would break its pickling and its repr."""
    p = solvable()
    p.ipopt_options.print_level = 0
    sol = p.solve()
    ps = sol[p.phases.slide]
    with raises(AttributeError, "cannot be deleted", "names are fixed", at="exec"):
        exec(f"del {target}", {"sol": sol, "ps": ps})  # noqa: S102


# ---------------------------------------------------------------------- the whole tree


def _reachable() -> list[object]:
    """Every object of YAPSS's front end reachable from a problem, its solution, and the
    arguments its callbacks are given, once each."""
    p = solvable("central-difference")
    p.ipopt_options.print_level = 0
    ph = p.phases.slide
    given: dict[str, tuple[object, ...]] = {}

    @ph.register.continuous
    def continuous(arg, out):
        given.setdefault("continuous", (arg, out))
        v, theta = arg.state.v, arg.control.theta
        out.dynamics.x = v * cos(theta)
        out.dynamics.y = v * sin(theta)
        out.dynamics.v = G0 * sin(theta)
        out.path.speed = v
        out.integrand.effort = theta**2

    @p.register.objective
    def objective(arg):
        given.setdefault("objective", (arg, arg[ph], arg[ph].initial, arg[ph].final))
        return arg[ph].final.time + 1e-3 * arg[ph].integral.effort

    @p.register.discrete
    def discrete(arg, out):
        given.setdefault("discrete", (arg, out))
        out.discrete.drop = arg[ph].final.y

    solution = p.solve()
    assert set(given) == {"continuous", "objective", "discrete"}
    seen: dict[int, object] = {}
    todo: list[object] = [p, solution, solution[ph], ph.state.x]
    todo += [obj for objects in given.values() for obj in objects]
    while todo:
        obj = todo.pop()
        if id(obj) in seen or not type(obj).__module__.startswith("yapss._api"):
            continue
        seen[id(obj)] = obj
        for name in dir(obj):
            if name.startswith("_"):
                continue
            try:
                value = getattr(obj, name)
            except Exception:  # noqa: BLE001, S112 -- a name that cannot be read is not walked
                continue
            if not callable(value) or type(value).__module__.startswith("yapss._api"):
                todo.append(value)
    return list(seen.values())


def test_every_reachable_object_refuses_an_undeclared_name_and_a_deletion() -> None:
    """A walk rather than a list, so that a new class cannot miss the protection unnoticed.

    Reached from a problem and its solution: every object whose class lives in YAPSS's front
    end refuses assignment to a name it does not declare, and refuses deleting a public name.
    """
    objects = _reachable()
    assert len(objects) > 20
    failures = []
    for obj in objects:
        kind = type(obj).__qualname__
        try:
            obj.__setattr__("zz_not_declared", 1)
        except (AttributeError, TypeError):
            pass
        else:
            failures.append(f"{kind} accepted an undeclared name")
        public = [n for n in dir(obj) if not n.startswith("_") and n != "zz_not_declared"]
        for name in public[:1]:
            try:
                obj.__delattr__(name)
            except (AttributeError, TypeError):
                pass
            else:
                failures.append(f"{kind} allowed 'del {name}'")
    assert not failures, "\n".join(failures)
