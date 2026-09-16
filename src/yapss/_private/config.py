"""

Package configuration: logging, Conda detection, and the `YAPSS_IPOPT_SOURCE` removal notice.

Imported for its side effect of configuring the ``yapss`` logger. Locating the Ipopt
library itself lives in `mseipopt.library`, not here.

"""

from __future__ import annotations

# standard library imports
import logging
import os
import platform
import sys
from pathlib import Path
from warnings import warn

from .exceptions import YapssDeprecationWarning

# ANSI escape codes for colors
RED = "\033[31m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)

# Configure the package root rather than this module, so every `yapss` submodule
# inherits the level and the handler -- notably `mseipopt.library`, where the Ipopt
# resolver logs which library it picked. Configuring `__name__` here left those
# messages invisible under YAPSS_LOGGING=DEBUG.
_package_logger = logging.getLogger("yapss")
level = os.environ.get("YAPSS_LOGGING", None)
if level:
    try:
        _package_logger.setLevel(level.upper())
    except ValueError:
        msg = (
            f"Invalid logging level: '{level}'. \n"
            f"    Valid levels are: DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )
        warn(msg, stacklevel=2)
        _package_logger.setLevel(logging.WARNING)

else:
    _package_logger.setLevel(logging.WARNING)

# Create a console handler and set the level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter(f"{RED}%(levelname)s %(name)s:%(lineno)d  %(message)s{RESET}")
console_handler.setFormatter(formatter)

# Add the handler to the package logger
_package_logger.addHandler(console_handler)


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


# --- `YAPSS_IPOPT_SOURCE` removal notice ---------------------------------------
#
# The variable selected the Ipopt backend until 0.3.0, when the cyipopt backend and the
# `ipopt_source` setting were removed. Setting it now has no effect, which is safe but
# silent, so a script or shell profile that still sets it is told once per process.
# Remove this notice in 0.4.0 or after 2027-09, whichever is later.

_ipopt_source_env_warned = False


def warn_if_ipopt_source_env_set() -> None:
    """Warn once per process that `YAPSS_IPOPT_SOURCE` no longer has any effect."""
    global _ipopt_source_env_warned  # noqa: PLW0603
    if _ipopt_source_env_warned or not os.environ.get("YAPSS_IPOPT_SOURCE"):
        return
    _ipopt_source_env_warned = True
    warn(
        "The YAPSS_IPOPT_SOURCE environment variable has no effect: it was removed in "
        "YAPSS 0.3.0, which always uses the Ipopt library that CasADi loads, after "
        "verifying it. It can be unset.",
        YapssDeprecationWarning,
        # warn_if_ipopt_source_env_set -> solver.solve -> Problem.solve -> the user's call
        stacklevel=4,
    )
