"""A problem's setup with no annotation written, and the messages mypy prints for its mistakes.

The block between the markers is what docs/user_guide/reference/typing.rst shows. Each line's
trailing comment is the message mypy must print for that line -- in full, or up to "..." --
and ``# fine`` says it prints none. ``check_messages.py`` runs mypy and holds the comments to
it, so a mypy release that rewords a message fails CI instead of leaving the page stale.

This file is outside ``tests/typed`` on purpose: its mistakes are real errors, and every file
there must pass.
"""

import yapss
from yapss.examples.brachistochrone_minimal import Phases

# -- shown on the page --------------------------------------------------------------------------
problem = yapss.Problem("Brachistochrone", phases=Phases)
ph = problem.phases.slide

ph.state.x.bounds = (0, 10)  # fine
ph.state.xx.bounds = (0, 10)  # "State" has no attribute "xx"
ph.state.x.bond = (0, 10)  # "ScalarField" has no attribute "bond"; maybe "bounds"?
ph.state.x.bounds = 5.0  # Incompatible types in assignment (expression has type "float", ...
problem.phases.slid  # "Phases" has no attribute "slid"; maybe "slide"?
problem.ipopt_options.max_iters = 5000  # "IpoptOptions" has no attribute "max_iters"; maybe ...
problem.ipopt_options.max_iter = "5000"  # Incompatible types in assignment (expression has ...
# -- end of what the page shows ------------------------------------------------------------------
