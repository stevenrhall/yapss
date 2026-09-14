"""Which Ipopt backend the solver will use in this process, for skipping tests.

Importable as ``_ipopt_backend`` both from test modules here (pytest puts this directory
on ``sys.path``) and from the subprocess workers run by path. Deleted with the cyipopt
backend in 0.3.0, along with every skip that uses it.
"""

from __future__ import annotations

import os

from yapss._private.config import get_conda_prefix

_SOURCE = os.environ.get("YAPSS_IPOPT_SOURCE", "")

CYIPOPT_ACTIVE = _SOURCE == "cyipopt" or (not _SOURCE and bool(get_conda_prefix()))
"""True when solves go through cyipopt, as `solver.configure_ipopt_source` decides it.

The override is honored so that a Conda run with ``YAPSS_IPOPT_SOURCE=casadi`` exercises
every mseipopt test, as it will once the cyipopt backend is removed.
"""
