"""A problem's setup with no annotation written, and the messages mypy prints for its mistakes.

The block between the markers is what docs/user_guide/reference/typing.rst shows. The comment
above each checked line is the message mypy must print for that line, in full or up to "...",
and may run over several comment lines, which are joined with a space. ``# fine`` says mypy
prints none. ``check_messages.py`` runs mypy and holds the comments to it, so a mypy release
that rewords a message fails CI instead of leaving the page stale.

This file is outside ``tests/typed`` on purpose: its mistakes are real errors, and every file
there must pass.
"""

from yapss.examples.brachistochrone_minimal import Brachistochrone

# -- shown on the page --------------------------------------------------------------------------
problem = Brachistochrone("Brachistochrone")
ph = problem.phases.phase

# fine
ph.state.x.bounds = (0, 10)

# "State" has no attribute "xx"
ph.state.xx.bounds = (0, 10)

# "ScalarField" has no attribute "bond"; maybe "bounds"?
ph.state.x.bond = (0, 10)

# Incompatible types in assignment (expression has type "float", variable has type
# "tuple[SupportsFloat | None, SupportsFloat | None] | list[SupportsFloat | None]")
ph.state.x.bounds = 5.0

# "Phases" has no attribute "phas"; maybe "phase"?
problem.phases.phas

# "IpoptOptions" has no attribute "max_iters"; maybe "max_iter"?
problem.ipopt_options.max_iters = 5000

# Incompatible types in assignment (expression has type "str", variable has type "int | None")
problem.ipopt_options.max_iter = "5000"
# -- end of what the page shows ------------------------------------------------------------------
