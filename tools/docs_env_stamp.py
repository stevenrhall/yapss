"""Record the installed distributions of a docs environment, in a stamp file for make.

The docs Makefiles regenerate the executed notebooks and the script output whenever the
stamp is newer than what they generated, because a different CasADi, NumPy or matplotlib
can change that output. The stamp is rewritten only when its contents change, so its
modification time moves only when the environment does. YAPSS itself is left out: tox
reinstalls it on every run, and the Makefiles depend on its source files directly.

Usage: python docs_env_stamp.py STAMP_FILE
"""

import sys
from importlib.metadata import distributions
from pathlib import Path


def main(stamp: Path) -> None:
    """Write the environment's description to `stamp` if it differs from what is there."""
    names = {
        f"{dist.metadata['Name'].lower()}=={dist.version}"
        for dist in distributions()
        if dist.metadata["Name"].lower() != "yapss"
    }
    text = "\n".join([sys.version, *sorted(names)]) + "\n"
    try:
        if stamp.read_text() == text:
            return
    except FileNotFoundError:
        pass
    stamp.write_text(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
