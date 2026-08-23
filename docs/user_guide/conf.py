"""

Sphinx configuration.

"""

# standard library imports
import datetime
import doctest
import os
import subprocess
import sys
from pathlib import Path

# third party imports
from docutils import nodes
from nbsphinx import CodeAreaNode

# project imports
from yapss import __version__ as version

# project information
project = "YAPSS"
copyright = f"2021-{datetime.datetime.now().year} MIT"  # noqa: A001
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
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_llm.txt",
    "nbsphinx",
    "numpydoc",
    "myst_parser",
    "link_modifier",  # custom extension to fix GitHub links
]

# Sphinx's own default is DONT_ACCEPT_TRUE_FOR_1 | ELLIPSIS | IGNORE_EXCEPTION_DETAIL.
# IGNORE_EXCEPTION_DETAIL means a `.. doctest::` block that ends in a Traceback only
# checks that the right exception *type* was raised -- the message text after it is
# never compared, so it can drift from what the code actually raises without the
# doctest build ever catching it. (Confirmed the hard way: reference/bounds.rst had a
# `ValueError` message that didn't match any string in the codebase, and the doctest
# suite passed regardless -- even swapping in an obviously wrong message still passed.)
# Dropping IGNORE_EXCEPTION_DETAIL here makes the message text part of the check too.
doctest_default_flags = doctest.DONT_ACCEPT_TRUE_FOR_1 | doctest.ELLIPSIS

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

copybutton_exclude = ".linenos, .gp"

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
# Warns on unresolved :doc:/:ref:/:class: targets etc. Since `docs`/CI build with -W,
# this turns a renamed or deleted cross-reference target into a build failure instead
# of a silently dead link.
nitpicky = True

# The first `make docs` run under `nitpicky = True` surfaced two different kinds of
# unresolved reference, both left unfixed on purpose:
#
# - NumPy's typing generics. numpydoc renders `NDArray[np.float64]` field annotations
#   as cross-references, but numpy's intersphinx inventory does not index them under
#   these bare names. Standard, unfixable friction in numpydoc-based projects.
# - "Leaf" classes that are deliberately undocumented on their own. YAPSS documents
#   tree-structured results (Bounds, Guess, Solution, ...) so that the container's own
#   page is where a user looks, rather than needing to chase into each per-phase or
#   per-item child class individually -- `PhaseBounds` (a child of `Bounds`) has never
#   had its own doc page, and `PhaseGuess`, `SolutionPhases`, `NLPInfo`, and
#   `ContinuousPhase` (a child of `ContinuousArg`) follow the same rule. Their names
#   still appear in the rendered type of the containing attribute (for example
#   `phase : tuple[PhaseGuess, ...]`); they are just not cross-reference targets.
#   `LimitOptions` and `DVStructure` are pure internal machinery
#   (`catch_keyboard_interrupt: LimitOptions[bool]`, the NLP decision-variable
#   structure passed into `ContinuousArg`, etc.) that a user never constructs
#   directly, so they are in the same category. `T` is a bare TypeVar leaking through
#   `Generic[T]`-based classes; it was never going to be a cross-reference target.
nitpick_ignore = [
    ("py:class", "NDArray"),
    ("py:class", "np.float64"),
    ("py:class", "yapss._private.types_.LimitOptions"),
    ("py:class", "yapss._private.guess.PhaseGuess"),
    ("py:class", "SolutionPhases"),
    ("py:class", "NLPInfo"),
    ("py:class", "DVStructure"),
    ("py:class", "T"),
    ("py:class", "yapss._private.input_args.ContinuousPhase"),
    ("py:class", "yapss._private.input_args.T"),
]

# options for HTML output
html_title = f"YAPSS {release}"
html_last_updated_fmt = "%b %d, %Y"
# Required for anything in `_static` to reach the build. `html_css_files` below only
# emits the <link> tag; without this, nothing is copied and the stylesheet 404s
# silently -- Sphinx does not check that `html_css_files` entries resolve.
html_static_path = ["_static"]
# No `html_logo`: setting it replaces the sidebar title, which currently carries the
# version. `_static/yapss_logo.png` is kept, unused, for pages that reference it
# directly. Likewise no `html_favicon` -- maintaining a full favicon set is more work
# than it is worth.
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


# sphinx-llm uses sphinx-markdown-builder to generate the Markdown files that are
# assembled into llms.txt and llms-full.txt. nbsphinx wraps notebook input and
# plain-text output cells in its custom CodeAreaNode. The Markdown builder has no
# visitor for that node and skips unknown nodes (including their children), which
# causes notebook code cells to disappear from the LLM documentation.
#
# The Markdown translation handlers below make input CodeAreaNode wrappers
# transparent so their normal literal_block children are rendered as fenced code,
# and select only nbsphinx's plain-text representation for output cells. nbsphinx
# stores equivalent HTML, LaTeX, and text forms of textual output; allowing the
# Markdown builder to traverse all of them would duplicate each output three times.
# add_translation_handlers() is used instead of add_node() because nbsphinx has
# already registered CodeAreaNode; re-registering it produces a Sphinx warning.
#
# TODO: nbsphinx uses FancyOutputNode for non-plain-text notebook output (for
# example figures and rich HTML). sphinx-markdown-builder does not currently
# preserve those outputs. Decide whether any such outputs are useful enough for
# LLM context to justify adding a Markdown handler. In most YAPSS examples the
# figures are probably low-value LLM context, so omitting them may be preferable.
def _visit_code_area_markdown(translator, node):
    """Render nbsphinx code cells appropriately in Markdown."""

    # Input cell: make CodeAreaNode transparent. Its literal_block child
    # will be rendered normally by sphinx-markdown-builder.
    if any(isinstance(child, nodes.literal_block) for child in node.children):
        return

    # Text output: nbsphinx provides HTML, LaTeX, and text versions of the
    # same output. Keep only the plain-text representation.
    for raw in node.findall(nodes.raw):
        if raw.get("format") == "text":
            translator.add(
                f"```\n{raw.astext()}\n```",
                prefix_eol=2,
                suffix_eol=2,
            )
            raise nodes.SkipNode

    # Ignore output types we don't yet know how to represent.
    raise nodes.SkipNode


def _depart_code_area_markdown(_translator, _node):
    pass


def setup(app):
    # Connected to "builder-inited" rather than "config-inited" so that `app.builder`
    # is available for the markdown-subprocess guard in run_makefiles below; both
    # events fire once, before any document is read, so nothing regenerated here
    # arrives too late for the build.
    app.connect("builder-inited", run_makefiles)
    app.registry.add_translation_handlers(
        CodeAreaNode,
        markdown=(_visit_code_area_markdown, _depart_code_area_markdown),
    )


def run_makefiles(app):
    # sphinx-llm builds llms.txt/llms-full.txt by spawning a second, independent
    # `sphinx-build -b markdown` subprocess against this same source tree, running in
    # parallel with the primary build by default. That subprocess loads this conf.py
    # too, so without this guard the notebook- and plot-regenerating `make all` targets
    # below would run a second time concurrently with the primary build reading their
    # output -- a race, not just wasted work. The markdown subprocess only needs the
    # already-generated files, not to regenerate them itself.
    if app.builder is not None and app.builder.name == "markdown":
        return

    root_dir = os.path.abspath(os.path.dirname(__file__))  # docs/user_guide

    # Both sub-Makefiles regenerate content by running Python: nbconvert executes the
    # example notebooks, and make_plots.py imports yapss to produce the figures and the
    # printed output in the user guide. Left to find `python` on PATH they would use
    # whatever comes first there, which is the interpreter running this build only if a
    # venv is active. The loud failure -- an interpreter with no nbconvert -- is
    # harmless. The quiet one is not: an interpreter that can import *some* yapss
    # regenerates the whole user guide against a different installation than the one
    # being documented, and the build succeeds, so the docs describe the wrong version.
    # Passing PYTHON pins both Makefiles to this interpreter.
    def run_command(command, cwd):
        env = {**os.environ, "PYTHON": sys.executable}
        try:
            subprocess.run(command, cwd=cwd, check=True, env=env)
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
