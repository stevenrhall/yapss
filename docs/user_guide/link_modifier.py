"""

Sphinx extension to fix "Edit on GitHub" links in HTML files.

Because the Jupyter notebooks are located outside the `docs/user guide` directory and
are copied into the `docs/user guide/notebooks` directory during the build process,
the "Edit on GitHub" links in the HTML files are incorrect. This extension modifies
the links in the HTML files after the build process to point to the correct location
in the GitHub repository.

"""

import os
import re

from bs4 import BeautifulSoup
from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def modify_links(app, exception):
    """Modify links in the HTML files after the build process."""
    # Only proceed if the builder format is HTML
    if app.builder.format != "html" or exception:
        return

    # Iterate through all HTML files in the build directory
    build_dir = app.builder.outdir
    for root, _, files in os.walk(build_dir):
        for filename in files:
            if filename.endswith(".html"):
                file_path = os.path.join(root, filename)

                # Open each HTML file and modify links as needed
                with open(file_path, "r+", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                    # Find all links and modify them as needed
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]

                        # Modify the link if it contains the specified path
                        a_tag["href"] = fix_link(href)

                    # Write the modified HTML back to the file
                    f.seek(0)
                    f.write(str(soup))
                    f.truncate()


def fix_link(url):
    """Modify the URL if it matches the specified pattern."""
    # Fix links to example notebooks
    fixed_url = re.sub(
        r"/blob/main/docs/user_guide/notebooks/(.*\.ipynb)",
        r"/blob/main/examples/notebooks/\1",
        url,
    )

    # fix links to
    # CHANGELOG.md, CONTRIBUTING.md
    fixed_url = re.sub(
        r"/blob/main/docs/user_guide/(C.*G\.md)",
        r"/blob/main/\1",
        fixed_url,
    )

    if fixed_url != url:
        logger.info(f"Modified link: '{url}' -> '{fixed_url}'")

    return fixed_url


def setup(app: Sphinx):
    """Initialize the Sphinx extension."""
    app.connect("build-finished", modify_links)
    return {"version": "1.0", "parallel_read_safe": True}
