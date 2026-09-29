"""A solution's settings are the problem's setup as it was solved, under the problem's names.

Every name a user can set on the problem appears in ``solution.settings`` at the same path
(``callbacks`` where the problem has ``register``), and nothing else does. Each holds the value
the problem held when the solve began, and the value it holds can be written back to a
problem: the settings of one solve can configure the next.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from yapss._api.settings import SettingsGroup, settings_of

from ._api import problem, raises, solvable
from ._settable import settable_paths

AREA = "solution"


def _leaves(group: SettingsGroup, path: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in dir(group):
        value = getattr(group, name)
        here = f"{path}.{name}"
        # the options are one leaf, as the problem's options object is
        if isinstance(value, SettingsGroup) and name != "ipopt_options":
            out.update(_leaves(value, here))
        else:
            out[here] = value
    return out


def _paths(settings: SettingsGroup) -> set[str]:
    return set(_leaves(settings, "problem"))


@pytest.mark.parametrize("build", [problem, solvable], ids=["every kind", "solvable"])
def test_the_settings_are_the_problem_s_settable_names_exactly(build) -> None:
    """Walked from the problem as a user reaches it, and compared as sets of paths."""
    ocp = build()
    assert _paths(settings_of(ocp)) == set(settable_paths(ocp))


def test_each_setting_holds_the_value_the_problem_held() -> None:
    ocp = problem()
    ph = ocp.phases.first
    ph.state.x.bounds = (0.0, 1.0)
    ph.state.y.guess[1] = (2.0, 3.0)
    ph.dynamics.x.scale = 4.0
    ph.duration.bounds = (1.0, 2.0)
    ocp.parameter.t.scale[:] = (5.0, 6.0)
    ocp.comment = "a variant"
    settings = _leaves(settings_of(ocp), "problem")
    for path, value in settable_paths(ocp).items():
        if path.endswith("ipopt_options") or ".callbacks." in path:
            continue
        expected = tuple(value) if hasattr(value, "_rows") else value
        assert settings[path] == expected, path


def test_a_solution_records_the_settings_the_solve_began_with() -> None:
    """A snapshot: editing the problem afterwards leaves the solution's settings as they were."""
    ocp = solvable()
    ocp.comment = "first"
    ocp.phases.slide.state.v.scale = 2.0
    result = ocp.solve()
    ocp.comment = "second"
    ocp.phases.slide.state.v.scale = 3.0
    assert result.settings.comment == "first"
    assert result.settings.phases.slide.state.v.scale == 2.0
    assert result.settings.name == result.name == ocp.name
    assert result.settings.spectral_method == result.spectral_method


def test_the_settings_can_be_written_back_to_a_problem() -> None:
    """Every value is in a form its setter accepts, block fields' rows included.

    The exception is a time guess never set, which reads None; its setter refuses None.
    """
    source = problem()
    source.phases.first.state.y.bounds[:] = [(0.0, 1.0), (2.0, 3.0)]
    source.phases.first.control.w.guess[0] = (1.0, 2.0)
    source.parameter.t.scale[1] = 7.0
    settings = _leaves(settings_of(source), "problem")
    target = problem()
    for path, value in settings.items():
        if path.endswith("ipopt_options") or ".callbacks." in path:
            continue
        owner, name = path.removeprefix("problem").rsplit(".", 1)
        obj = target
        for part in filter(None, owner.split(".")):
            obj = getattr(obj, part)
        if value is None and name == "guess":
            continue  # a time guess never set reads None, which its setter refuses
        if hasattr(getattr(obj, name), "_rows"):  # a block field's setting: rows, not a value
            getattr(obj, name)[:] = value
        else:
            setattr(obj, name, value)
    assert _leaves(settings_of(target), "problem") == settings


def test_a_callback_is_recorded_by_identity() -> None:
    ocp = solvable()
    result = ocp.solve()
    objective = result.settings.callbacks.objective
    assert objective.module == ocp._objective_function.__module__
    assert objective.qualname == ocp._objective_function.__qualname__
    assert result.settings.callbacks.discrete is not None
    assert result.settings.phases.slide.callbacks.continuous.qualname.endswith("continuous")


def test_the_options_are_recorded_as_set() -> None:
    ocp = solvable()
    ocp.ipopt_options.max_iter = 77
    result = ocp.solve()
    assert result.settings.ipopt_options.max_iter == 77


def test_the_settings_pickle_as_values() -> None:
    ocp = solvable()
    result = ocp.solve()
    copy = pickle.loads(pickle.dumps(result))
    assert copy.settings.phases.slide.state.v.bounds == result.settings.phases.slide.state.v.bounds
    assert copy.settings.callbacks.objective == result.settings.callbacks.objective


def test_a_setting_cannot_be_assigned() -> None:
    result = solvable().solve()
    with raises(AttributeError, "cannot be assigned", at="settings.comment ="):
        result.settings.comment = "changed"


def test_a_misspelled_setting_is_refused_with_a_suggestion() -> None:
    result = solvable().solve()
    with raises(AttributeError, "has no 'coment'", "Did you mean 'comment'?", at="coment"):
        _ = result.settings.coment


def test_the_bounds_are_as_set_with_infinities_for_open_sides() -> None:
    ocp = problem()
    settings = settings_of(ocp)
    assert settings.phases.first.state.x.bounds == (-np.inf, np.inf)
    assert settings.phases.first.dynamics.x.scale is None
