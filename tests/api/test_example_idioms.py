"""The examples use only the public, typed access model.

Users copy the examples, so what an example does is what users do. Two things are kept out of
every example script and every notebook's code:

- A private name (``Discrete._fields``, ``problem._x``): the examples teach the public API only.
- ``getattr``, ``setattr`` and ``hasattr``: reaching for them means the access model is wrong
  for the job. A setting repeated over several fields loops over the fields themselves --
  ``for field in (ph.state.x, ph.state.y): field.scale = 1000.0`` -- which a type checker
  follows; a name held in a string it cannot.
"""

import ast
import json
from pathlib import Path

import pytest

import yapss.examples

SCRIPTS = sorted(Path(yapss.examples.__file__).parent.glob("*.py"))
NOTEBOOKS = sorted((Path(__file__).parents[2] / "examples" / "notebooks").glob("*.ipynb"))
REFLECTION = {"getattr", "setattr", "hasattr"}


def _sources(path: Path) -> list[tuple[str, str]]:
    """Return (where, source) for a script, or for each code cell of a notebook."""
    if path.suffix == ".py":
        return [(path.name, path.read_text(encoding="utf-8"))]
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [
        (f"{path.name} cell {index}", "".join(cell["source"]))
        for index, cell in enumerate(cells)
        if cell["cell_type"] == "code"
    ]


def _offences(source: str) -> list[str]:
    offences = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            if not (node.attr.startswith("__") and node.attr.endswith("__")):
                offences.append(f"line {node.lineno}: private name '.{node.attr}'")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in REFLECTION
        ):
            offences.append(f"line {node.lineno}: {node.func.id}()")
    return offences


def test_there_is_something_to_check():
    assert len(SCRIPTS) > 10
    assert len(NOTEBOOKS) > 10


@pytest.mark.parametrize("path", SCRIPTS + NOTEBOOKS, ids=lambda path: path.name)
def test_an_example_uses_only_the_public_typed_access_model(path):
    found = [
        f"{where}, {offence}" for where, source in _sources(path) for offence in _offences(source)
    ]
    assert not found, "\n".join(found)


def _undocumented_fields(source: str) -> list[str]:
    """Return each field declared with `scalar()` or `vector(n)` that has no docstring below it."""
    missing = []
    for cls in ast.parse(source).body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for statement, following in zip(cls.body, [*cls.body[1:], None], strict=True):
            if not (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr in {"scalar", "vector"}
            ):
                continue
            documented = (
                isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            )
            if not documented:
                name = ast.unparse(statement.targets[0])
                missing.append(f"line {statement.lineno}: {cls.name}.{name}")
    return missing


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.name)
def test_every_field_of_an_example_has_a_docstring(path):
    """A script documents itself: its page renders each vector class's fields from their
    docstrings, and a field without one would drop off the page without a word."""
    missing = _undocumented_fields(path.read_text(encoding="utf-8"))
    assert not missing, "\n".join(missing)
