"""

The module provides helper functions to configure the Ipopt source.

Locating the Ipopt library itself lives in `mseipopt.library`, not here.

"""

from __future__ import annotations

# standard library imports
import logging
import os
import platform
import sys
from pathlib import Path
from warnings import warn

# ANSI escape codes for colors
RED = "\033[31m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)
level = os.environ.get("YAPSS_LOGGING", None)
if level:
    try:
        logger.setLevel(level.upper())
    except ValueError:
        msg = (
            f"Invalid logging level: '{level}'. \n"
            f"    Valid levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
        warn(msg, stacklevel=2)
        logger.setLevel(logging.WARNING)

else:
    logger.setLevel(logging.WARNING)

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter(f"{RED}%(levelname)s %(name)s:%(lineno)d  %(message)s{RESET}")
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


def get_conda_prefix() -> Path | None:
    """Determine the Conda prefix associated with the current Python kernel.

    Could just get the environment variable CONDA_PREFIX, but that might fail in a
    JupyterLab notebook, since the CONDA_PREFIX variable corresponds to the JupyterLab
    server kernel, not the Python kernel running the notebook.

    Returns
    -------
    pathlib.Path | None
        The Conda environment prefix, or None if not in a Conda environment.
    """
    platform_str = platform.system()
    logger.debug(f"Platform: {platform_str}")

    # Get the directory containing the Python executable
    python_executable = Path(sys.executable)
    logger.debug(f"Python executable: {python_executable}")
    python_executable_dir = python_executable.parent
    conda_prefix = (
        python_executable_dir if platform_str == "Windows" else python_executable_dir.parent
    )
    logger.debug(f"Possible Conda prefix: {conda_prefix}")

    # Check if `conda-meta` exists in the current directory
    if (conda_prefix / "conda-meta").exists():
        logger.debug(f"Conda prefix confirmed: {conda_prefix}")
        return conda_prefix

    logger.debug("Not a Conda environment.")
    return None
