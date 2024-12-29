"""

The module provides helper functions to configure the Ipopt source.

"""

from __future__ import annotations

# standard library imports
import inspect
import logging
import os
import platform
import sys
from pathlib import Path
from warnings import warn

# third party imports
import casadi

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


def get_casadi_ipopt_library_path() -> str:
    """Find an Ipopt library for mseipopt to use."""
    # Set the library name depending on the platform
    platform_str = platform.system()
    if platform_str == "Windows":
        library_names = ["ipopt-3.dll", "ipopt.dll", "libipopt-3.dll", "libipopt.dll"]
    elif platform_str == "Darwin":
        library_names = ["libipopt.3.dylib", "libipopt.dylib"]
    else:  # Linux, or hope that systems uses the same naming convention as Linux
        library_names = ["libipopt.so.3", "libipopt.so"]

    # Check whether we are in a conda environment
    conda_default_env = get_conda_prefix()

    # determine the library directory
    if conda_default_env:
        conda_suffix = "Library/bin" if platform_str == "Windows" else "lib"
        library_directory = Path(conda_default_env) / conda_suffix
    else:
        library_directory = Path(inspect.getfile(casadi)).parent

    # Look for the library in the library directory
    logger.debug(f"Looking for Ipopt library in {library_directory}")
    for library_name in library_names:
        ipopt_path = library_directory / library_name
        logger.debug(f"Checking library_name: {library_name}")
        if ipopt_path.exists():
            logger.debug(f"Found casadi Ipopt library: {ipopt_path}")
            return ipopt_path.as_posix()

    msg = "Ipopt library not found."
    raise ValueError(msg)


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
