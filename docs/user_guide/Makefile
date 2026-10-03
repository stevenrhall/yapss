# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target for other Sphinx commands
%: Makefile $(MD)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

##############################################################################
# Custom targets

ROOT_DIR      = ../..

MD = CHANGELOG.md CONTRIBUTING.md

%.md: $(ROOT_DIR)/%.md
	@echo "Copying $< to current directory"
	cp $< $@

.PHONY: markdown
markdown: $(MD) ## copy markdown files

# Everything conf.py regenerates before Sphinx reads the sources: the copied Markdown, the
# executed notebooks, and the scripts' figures and output. The two sub-makes are
# prerequisites of one target, so under -j they run side by side and share one pool of
# jobs; conf.py chooses the pool's size.
.PHONY: generated generated-notebooks generated-scripts
generated: markdown generated-notebooks generated-scripts

generated-notebooks:
	$(MAKE) -C notebooks all

generated-scripts:
	$(MAKE) -C scripts/plots all

# Clean target
.PHONY: clean
clean:
	rm -f $(MD)
	cd notebooks && $(MAKE) clean
	cd scripts && $(MAKE) clean
	rm -rf build/
