"""

Sphinx configuration for development documentation.

"""

import sys
from importlib.metadata import version
from pathlib import Path

# Project information
project = "YAPSS"
copyright = "2024, MIT"
release = version("yapss")
version = ".".join(release.split(".")[:4])

# General configuration
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
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
numpydoc_show_class_members = False
always_document_param_types = False
source_suffix = {".rst": "restructuredtext"}
source_encoding = "utf-8-sig"
master_doc = "index"
exclude_patterns = ["build", "_build"]
templates_path = []
primary_domain = "py"
keep_warnings = False
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
