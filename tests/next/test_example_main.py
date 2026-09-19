"""Every example runs as a script, and produces the figures its documentation shows.

The figure counts are not decoration. `docs/user_guide/scripts/<name>.rst` names each figure
with a caption, in order, and `plots/Makefile` names each PNG file, so an example that stops
producing one of them fails the documentation build rather than quietly dropping a plot. This
pins the count on the example side, where it is cheap to check.

`main` is called directly rather than through `runpy`, as `tests/examples/test_example_main.py`
does: re-executing an already-imported module warns, and the module guard is not what is at
stake.
"""

import importlib

import matplotlib.pyplot as plt
import pytest

FIGURES = {
    "brachistochrone": 5,
    "brachistochrone_minimal": 1,
    "brachistochrone_user_derivatives": 5,
    "delta_iii_ascent": 7,
    "dynamic_soaring": 7,
    "goddard_problem_1_phase": 5,
    "goddard_problem_3_phase": 5,
    "hs071": 0,
    "isoperimetric": 2,
    "minimum_time_to_climb": 7,
    "newton": 3,
    "orbit_raising": 5,
    "rosenbrock": 1,
}
"""How many figures each example's `main` leaves open, as its documentation page shows.

`brachistochrone_user_derivatives` has no page of its own and borrows `brachistochrone`'s
plotting, so its count is that one's rather than a page's.
"""


@pytest.fixture
def no_windows(monkeypatch):
    """Make `plt.show` a no-op, and start from no open figures."""
    plt.close("all")
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


@pytest.mark.parametrize("name", FIGURES)
def test_main_runs_and_plots_what_the_docs_show(name, no_windows, capsys):
    module = importlib.import_module(f"yapss.examples.{name}")
    module.main()
    assert len(plt.get_fignums()) == FIGURES[name]
    # an example that prints nothing at all has no text output for its page either
    if name in {"hs071", "rosenbrock", "newton"}:
        assert capsys.readouterr().out.strip()


def test_every_example_is_covered():
    """A new example must be added here, which is what keeps the docs pages honest."""
    import pkgutil

    from yapss import examples

    found = {
        info.name
        for info in pkgutil.iter_modules(examples.__path__)
        if not info.name.startswith("_")
    }
    assert found == set(FIGURES)
