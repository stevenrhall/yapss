"""

Sphinx configuration for development documentation.

"""

import datetime
import sys
from importlib.metadata import version
from pathlib import Path

# Project information
project = "YAPSS"
copyright = f"2024-{datetime.datetime.now().year}, MIT"
release = version("yapss")
version = ".".join(release.split(".")[:4])

# General configuration
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Enable autosummary to pre-generate documentation stubs
autosummary_generate = True

# Autodoc configurations
autodoc_typehints = "both"
autodoc_member_order = "bysource"
add_module_names = False

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "special-members": "__init__",
    "ignore-module-all": True,
    "undoc-members": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True
source_suffix = {".rst": "restructuredtext"}
source_encoding = "utf-8-sig"
master_doc = "index"
exclude_patterns = ["build", "_build"]
templates_path = []
primary_domain = "py"
keep_warnings = False
# The public problem model and low-level mseipopt wrapper intentionally both expose
# a class named ``Problem``. Sphinx cannot resolve unqualified references uniquely,
# but either target is still available by its fully qualified name in these API docs.
suppress_warnings = ["ref.python"]
highlight_language = "python"
pygments_style = "manni"

# Options for HTML output
html_title = project + " " + version
html_last_updated_fmt = "%b %d, %Y"
html_static_path = ["_static"]
html_use_smartypants = True
html_use_index = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_css_files = ["css/custom.css"]

# HTML theme settings
html_theme = "furo"

highlight_options = {"linenos": True}
viewcode_line_numbers = False
