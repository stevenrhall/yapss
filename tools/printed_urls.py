"""List every URL YAPSS can print, so that each can be checked against a release's docs.

A URL YAPSS prints is a string in its code, outside a docstring or a comment: a message, a
note, a constant a message uses. This lists each such string with its file and line, and the
address of every documentation page linked through ``config.docs_url``, for the version given
(the installed one by default). The release checklist asks that every one resolves against the
tagged version's docs before publication.

Usage::

    python tools/printed_urls.py            # for the installed version
    python tools/printed_urls.py 0.4.0      # as release 0.4.0 would print them
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "yapss"


def _docstrings(tree: ast.AST) -> set[int]:
    """Return the ids of the nodes that are docstrings."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                found.add(id(body[0].value))
    return found


def _text(node: ast.AST) -> str:
    """Return a string constant, or an f-string with its fields shown as ``{...}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    assert isinstance(node, ast.JoinedStr)
    return "".join(
        value.value if isinstance(value, ast.Constant) else "{...}" for value in node.values
    )


def printed_urls(package: Path = PACKAGE) -> list[tuple[str, int, str]]:
    """Return ``(file, line, text)`` for every string in the package that holds a URL."""
    out = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstrings(tree)
        inside_fstring = {
            id(part)
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            for part in node.values
        }
        for node in ast.walk(tree):
            if id(node) in docstrings or id(node) in inside_fstring:
                continue
            if isinstance(node, ast.Constant | ast.JoinedStr):
                if isinstance(node, ast.Constant) and not isinstance(node.value, str):
                    continue
                text = _text(node)
                if "http://" in text or "https://" in text:
                    out.append((str(path.relative_to(package.parent)), node.lineno, text))
    return out


def linked_pages(package: Path = PACKAGE) -> list[str]:
    """Return every page linked through ``docs_url``, by its path in the documentation."""
    pages = {
        node.args[0].value
        for path in sorted(package.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "docs_url"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    return sorted(pages)


def main(argv: list[str]) -> int:
    """Print the strings holding URLs, then the linked pages for the version given."""
    sys.path.insert(0, str(PACKAGE.parent))
    from yapss._backend.config import docs_url  # noqa: PLC0415 -- after the path is set

    installed = argv[1] if len(argv) > 1 else None
    for file, line, text in printed_urls():
        print(f"{file}:{line}: {text}")
    print()
    for page in linked_pages():
        print(docs_url(page, installed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
