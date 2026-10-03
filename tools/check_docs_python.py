"""Check that this interpreter can build the docs from this working tree.

`make quick-docs` builds the user guide with whatever `python` is first on PATH, instead of
tox's pinned environment. Two things go wrong with that interpreter, one loudly and one not.
Without Sphinx the build fails at once. With Sphinx but a yapss other than this tree's
(another checkout, a wheel), the build succeeds and documents the wrong code. This script
turns both into a message before the build starts.

Usage: python tools/check_docs_python.py
"""

import importlib.util
import sys
from pathlib import Path

TREE_SRC = (Path(__file__).resolve().parent.parent / "src").resolve()


def main() -> None:
    """Exit with a message unless Sphinx is importable and yapss comes from this tree."""
    advice = "Activate this tree's development virtualenv, or run `make develop` in it."
    if importlib.util.find_spec("sphinx") is None:
        sys.exit(f"{sys.executable} has no Sphinx. {advice}")
    spec = importlib.util.find_spec("yapss")
    if spec is None or spec.origin is None:
        sys.exit(f"{sys.executable} cannot import yapss. {advice}")
    origin = Path(spec.origin).resolve()
    if TREE_SRC not in origin.parents:
        sys.exit(
            f"{sys.executable} imports yapss from {origin.parent}, not from {TREE_SRC / 'yapss'}."
            f" {advice}"
        )


if __name__ == "__main__":
    main()
