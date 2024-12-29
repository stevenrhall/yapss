"""

Sphinx configuration.

"""

# standard library imports
import os
import subprocess
import sys
from pathlib import Path

# project imports
from yapss import __version__ as version

# project information
project = "YAPSS"
copyright = "2021-2024 MIT"  # noqa: A001
release = version = version.split("+")[0]

# general configuration
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.append(os.path.abspath("."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "nbsphinx",
    "numpydoc",
    "myst_parser",
    "link_modifier",  # custom extension to fix GitHub links
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

copybutton_exclude = ".linenos, .gp"

# inject environment variable READTHEDOCS into notebook environment
on_rtd = os.environ.get("READTHEDOCS") == "True"
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
    f"--os.environ['READTHEDOCS']={on_rtd}",
]

autosummary_generate = True
add_module_names = False
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
always_document_param_types = False
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
source_encoding = "utf-8-sig"
master_doc = "index"
exclude_patterns = ["build", "_build"]
templates_path = []
primary_domain = "py"
keep_warnings = False
highlight_language = "python"
pygments_style = "manni"

# options for HTML output
html_title = f"YAPSS {release}"
html_last_updated_fmt = "%b %d, %Y"
if not os.path.exists("_static"):
    print("Warning: _static directory not found, skipping.")
html_use_smartypants = True
html_use_index = True
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_css_files = ["css/custom.css"]
html_theme = "sphinx_rtd_theme"

# Display version in the menu
html_context = {
    "display_github": True,
    "github_user": "stevenrhall",
    "github_repo": "yapss",
    "github_version": "main",
    "conf_py_path": "/docs/user_guide/",
}

bibtex_bibfiles = ["references.bib"]


def setup(app):
    # run makefiles to generate content
    app.connect("config-inited", run_makefiles)
    # flag for conditional content if building on ReadTheDocs
    app.add_config_value("on_rtd", on_rtd, "html")


def run_makefiles(_app, _config):
    root_dir = os.path.abspath(os.path.dirname(__file__))  # docs/user_guide

    def run_command(command, cwd):
        try:
            subprocess.run(command, cwd=cwd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Makefile execution failed in {cwd} with error:\n{e}") from e

    # Run Makefile in root directory
    run_command(["make", "markdown"], cwd=root_dir)

    # Run Makefile in notebooks directory
    notebooks_path = os.path.join(root_dir, "notebooks")
    run_command(["make", "all"], cwd=notebooks_path)

    # Run Makefile in scripts/plots directory
    plots_path = os.path.join(root_dir, "scripts", "plots")
    run_command(["make", "all"], cwd=plots_path)
