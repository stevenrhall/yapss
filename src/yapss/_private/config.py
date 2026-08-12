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


# --- `ipopt_source` deprecation -------------------------------------------------
#
# Removed in 0.3.0. Two distinct deprecations are in play and the messages differ
# accordingly: the attribute and environment variable are going away (which
# affects anyone who touches them at all, including someone setting "default"),
# and the *capability* of choosing a non-default backend is going away (which
# affects only the non-default values, and those users face a behavior change
# rather than merely an API change).
#
# Message text lives here, in one place, because it is emitted from two sites --
# the `Problem.ipopt_source` setter and the environment-variable branch of
# `solver.configure_ipopt_source()` -- and the two must not drift. Policy and
# rationale: IPOPT_BACKEND_POLICY.md §3.

_REMOVAL_VERSION = "0.3.0"

_SEE_ALSO = (
    'See "Sharp Edges" in the user guide for why YAPSS must control which Ipopt ' "it loads."
)


def ipopt_source_deprecation(value: str) -> tuple[type[Warning], str]:
    """Return the warning category and message for an `ipopt_source` value.

    `FutureWarning` for a custom library path, `DeprecationWarning` otherwise.
    The distinction is not cosmetic: `DeprecationWarning` is hidden by default
    unless triggered in ``__main__``, and the custom-path case can end in a
    SIGSEGV rather than an exception, so its warning must be visible from a
    notebook or an imported module as well.

    Parameters
    ----------
    value : str
        The requested `ipopt_source` value.

    Returns
    -------
    tuple[type[Warning], str]
        Warning category, and the message to emit.
    """
    tail = f"It is deprecated and will be removed in YAPSS {_REMOVAL_VERSION}."

    if value == "default":
        return DeprecationWarning, (
            f"'ipopt_source' is no longer used; the Ipopt backend is determined by "
            f"whether YAPSS is running in a Conda environment. {tail} Setting it to "
            f"'default' selects the behavior that is already in effect, so this line "
            f"can simply be deleted."
        )

    if value == "cyipopt":
        if get_conda_prefix():
            return DeprecationWarning, (
                f"'ipopt_source' is deprecated. In a Conda environment YAPSS already "
                f"uses cyipopt, so this line can simply be deleted. {tail}"
            )
        return DeprecationWarning, (
            f"'ipopt_source=\"cyipopt\"' outside a Conda environment loads a second, "
            f"independently built Ipopt alongside the one CasADi bundles. The two can "
            f"collide over their vendored OpenMP runtimes and crash. {tail} After that, "
            f"YAPSS will use its own bundled interface here. {_SEE_ALSO}"
        )

    if value == "casadi":
        return DeprecationWarning, (
            f"'ipopt_source' is deprecated. Outside a Conda environment 'casadi' is "
            f"already what YAPSS does, so this line can simply be deleted. {tail}"
        )

    return FutureWarning, (
        f"'ipopt_source' is set to an explicit Ipopt library path "
        f"({value!r}). YAPSS cannot verify a library it did not bundle: there is no "
        f"matching 'IpoptConfig.h' to check the ABI against, and a mismatched build "
        f"crashes the process rather than raising. {tail} {_SEE_ALSO}"
    )


def warn_ipopt_source_deprecated(value: str, stacklevel: int = 3) -> None:
    """Emit the deprecation warning for an `ipopt_source` value.

    Flushes `sys.stderr` after a `FutureWarning`, because that path may be
    followed by a hard crash and `stderr` is block-buffered when redirected to a
    file -- an unflushed warning would be lost precisely when it matters most.

    Parameters
    ----------
    value : str
        The requested `ipopt_source` value.
    stacklevel : int, default=3
        Passed through to `warnings.warn`, so the report points at user code.
    """
    category, message = ipopt_source_deprecation(value)
    warn(message, category=category, stacklevel=stacklevel)
    if category is FutureWarning:
        sys.stderr.flush()
